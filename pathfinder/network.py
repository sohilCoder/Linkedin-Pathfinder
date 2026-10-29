"""The shared network: members' uploads -> resolved people -> weighted graph.

Flow:
    uploads (profile + Connections.csv per member)
      -> matching.resolve()          one node per real person
      -> tie strength per edge        P(this person will make the intro)
      -> search.hop_limited_dijkstra  strongest path within K hops
      -> ranking, privacy redaction, intro drafts
"""

from __future__ import annotations

import json
import math
import shutil
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .loader import (
    MEMBER_ID_PATTERN,
    read_text_file,
    ConnectionRow,
    MemberProfile,
    parse_connections_csv,
)
from .matching import Mention, Person, Resolution, resolve
from .normalize import normalize_company
from .search import cost_to_probability, hop_limited_dijkstra, probability_to_cost

# --- Tie strength ------------------------------------------------------------

# Logistic model: P(intro happens) = sigmoid(sum of weights of active signals).
# These are hand-set, explainable starting points; with real feedback
# ("did the intro happen?") they could be fit with logistic regression.
WEIGHTS = {
    "bias": -0.6,
    "same_company": 1.1,      # work together now
    "former_colleague": 0.8,  # one of them used to work at the other's company
    "same_school": 0.7,       # both members from the same school
    "mutuals": 0.45,          # per log(1 + mutual connections)
    "new_connection": -0.9,   # connected in the last 90 days
    "long_standing": 0.3,     # connected 3+ years ago
}
MIN_P, MAX_P = 0.05, 0.92


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Tie:
    a: int
    b: int
    connected_on: date | None
    reported_by: set[str] = field(default_factory=set)
    probability: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class Upload:
    profile: MemberProfile
    rows: list[ConnectionRow]


class Network:
    def __init__(self, *, as_of: date | None = None, data_dir: str | Path | None = None):
        self.as_of = as_of or date.today()
        self.data_dir = Path(data_dir) if data_dir else None
        self.uploads: dict[str, Upload] = {}
        self._lock = threading.RLock()
        self._dirty = True
        self.resolution: Resolution | None = None
        self.ties: dict[tuple[int, int], Tie] = {}
        self.neighbors: dict[int, set[int]] = {}
        self.adj: dict[int, list[tuple[int, float]]] = {}

    # --- loading and saving -------------------------------------------------
    @classmethod
    def load(cls, data_dir: str | Path, *, persist: bool = True) -> "Network":
        """Load every member folder under data_dir/members/."""
        data_dir = Path(data_dir)
        as_of = None
        manifest = data_dir / "manifest.json"
        if manifest.exists():
            as_of = date.fromisoformat(json.loads(read_text_file(manifest))["as_of"])
        net = cls(as_of=as_of, data_dir=data_dir if persist else None)
        members_dir = data_dir / "members"
        if members_dir.exists():
            for folder in sorted(members_dir.iterdir()):
                profile_file = folder / "profile.json"
                csv_file = folder / "Connections.csv"
                if not (profile_file.exists() and csv_file.exists()):
                    continue
                profile = MemberProfile.from_dict(json.loads(read_text_file(profile_file)))
                rows = parse_connections_csv(read_text_file(csv_file))
                net.uploads[profile.member_id] = Upload(profile, rows)
        return net

    def add_member(self, profile: MemberProfile, csv_text: str) -> MemberProfile:
        """Add (or replace) a member's upload. Raises ExportFormatError on bad CSV."""
        rows = parse_connections_csv(csv_text)
        with self._lock:
            self.uploads[profile.member_id] = Upload(profile, rows)
            self._dirty = True
            if self.data_dir:
                folder = self.data_dir / "members" / profile.member_id
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "profile.json").write_text(
                    json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
                (folder / "Connections.csv").write_text(csv_text, encoding="utf-8")
        return profile

    def remove_member(self, member_id: str) -> bool:
        """Delete a member and every row they uploaded."""
        if not MEMBER_ID_PATTERN.match(member_id):
            return False
        with self._lock:
            if member_id not in self.uploads:
                return False
            del self.uploads[member_id]
            self._dirty = True
            if self.data_dir:
                folder = self.data_dir / "members" / member_id
                if folder.is_dir():
                    shutil.rmtree(folder)
        return True

    # --- building -----------------------------------------------------------
    def _ensure_built(self) -> None:
        with self._lock:
            if self._dirty:
                self._build()
                self._dirty = False

    def _build(self) -> None:
        mentions: list[Mention] = []
        for mid, up in self.uploads.items():
            mentions.append(Mention.from_profile(up.profile))
            mentions.extend(Mention.from_row(mid, r) for r in up.rows)
        res = resolve(mentions)
        self.resolution = res

        ties: dict[tuple[int, int], Tie] = {}
        for mid, up in self.uploads.items():
            me = res.person_of[("member", mid)]
            for row in up.rows:
                other = res.person_of[("row", mid, row.row_index)]
                if other == me:
                    continue
                key = (min(me, other), max(me, other))
                tie = ties.get(key)
                if tie is None:
                    tie = ties[key] = Tie(key[0], key[1], row.connected_on)
                elif row.connected_on and (not tie.connected_on or row.connected_on < tie.connected_on):
                    tie.connected_on = row.connected_on
                tie.reported_by.add(mid)

        neighbors: dict[int, set[int]] = defaultdict(set)
        for a, b in ties:
            neighbors[a].add(b)
            neighbors[b].add(a)

        member_ctx = {
            res.person_of[("member", mid)]: up.profile for mid, up in self.uploads.items()
        }
        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), tie in ties.items():
            self._score(tie, res.people, neighbors, member_ctx)
            cost = probability_to_cost(tie.probability)
            adj[a].append((b, cost))
            adj[b].append((a, cost))

        self.ties, self.neighbors, self.adj = ties, dict(neighbors), dict(adj)

    def _score(self, tie: Tie, people: list[Person], neighbors, member_ctx) -> None:
        pa, pb = people[tie.a], people[tie.b]
        score = WEIGHTS["bias"]
        reasons: list[str] = []

        if pa.ncompany and pa.ncompany == pb.ncompany:
            score += WEIGHTS["same_company"]
            reasons.append(f"both at {pa.company or pb.company}")
        else:
            for x, y in ((pa, pb), (pb, pa)):
                prof = member_ctx.get(x.person_id)
                if prof and y.ncompany and y.ncompany in {
                    normalize_company(c) for c in prof.past_companies
                }:
                    score += WEIGHTS["former_colleague"]
                    reasons.append(f"{x.name.split()[0]} used to work at {y.company}")
                    break

        prof_a, prof_b = member_ctx.get(tie.a), member_ctx.get(tie.b)
        if prof_a and prof_b and prof_a.school and (
            normalize_company(prof_a.school) == normalize_company(prof_b.school)
        ):
            score += WEIGHTS["same_school"]
            reasons.append(f"both from {prof_a.school}")

        mutual = len(neighbors[tie.a] & neighbors[tie.b])
        if mutual:
            score += WEIGHTS["mutuals"] * math.log1p(mutual)
            reasons.append(f"{mutual} mutual connection{'s' if mutual != 1 else ''}")

        if tie.connected_on:
            age_days = (self.as_of - tie.connected_on).days
            if age_days < 90:
                score += WEIGHTS["new_connection"]
                reasons.append("connected recently")
            elif age_days > 3 * 365:
                score += WEIGHTS["long_standing"]
                reasons.append(f"connected since {tie.connected_on.year}")

        tie.probability = min(MAX_P, max(MIN_P, _sigmoid(score)))
        tie.reasons = reasons or ["connected on LinkedIn"]

    # --- queries ------------------------------------------------------------
    def members(self) -> list[dict]:
        self._ensure_built()
        res = self.resolution
        out = []
        for mid, up in sorted(self.uploads.items(), key=lambda kv: kv[1].profile.full_name):
            pid = res.person_of[("member", mid)]
            out.append({
                "member_id": mid,
                "name": up.profile.full_name,
                "company": up.profile.company,
                "position": up.profile.position,
                "school": up.profile.school,
                "connections": len(up.rows),
                "reachable_people": len(self.neighbors.get(pid, ())),
            })
        return out

    def stats(self) -> dict:
        self._ensure_built()
        res = self.resolution
        companies = Counter()
        display: dict[str, Counter] = defaultdict(Counter)
        for p in res.people:
            if p.ncompany:
                companies[p.ncompany] += 1
                display[p.ncompany][p.company] += 1
        top = [
            {"company": display[key].most_common(1)[0][0], "people": n}
            for key, n in companies.most_common(40)
        ]
        return {
            "as_of": self.as_of.isoformat(),
            "members": len(self.uploads),
            "rows_uploaded": sum(len(u.rows) for u in self.uploads.values()),
            "people": len(res.people),
            "ties": len(self.ties),
            "matching": res.stats,
            "top_companies": top,
        }

    def search(
        self,
        member_id: str,
        company: str,
        *,
        title: str = "",
        max_hops: int = 3,
        limit: int = 10,
        max_per_first_hop: int = 3,
    ) -> dict:
        """Ranked intro paths from a member to people at `company`."""
        self._ensure_built()
        if member_id not in self.uploads:
            raise KeyError(f"unknown member: {member_id}")
        max_hops = max(1, min(int(max_hops), 4))
        limit = max(1, min(int(limit), 25))   # cap so nobody can dump a member's list
        res = self.resolution
        source = res.person_of[("member", member_id)]
        ncompany = normalize_company(company)
        title_lc = title.strip().lower()

        targets = [
            p for p in res.people
            if p.ncompany and p.ncompany == ncompany and p.person_id != source
            and (not title_lc or title_lc in p.position.lower())
        ]
        best = hop_limited_dijkstra(self.adj, source, max_hops)

        candidates = []
        for t in targets:
            if t.person_id in best:
                cost, path = best[t.person_id]
                candidates.append((cost, len(path), t, path))
        candidates.sort(key=lambda c: (c[0], c[1], c[2].name))

        per_first_hop: Counter = Counter()
        results = []
        for cost, _, target, path in candidates:
            first_hop = path[1]
            if len(path) > 2 and per_first_hop[first_hop] >= max_per_first_hop:
                continue   # don't route every request through one person
            per_first_hop[first_hop] += 1
            results.append(self._describe(path, cost, target, company, title))
            if len(results) == limit:
                break

        return {
            "query": {"member_id": member_id, "company": company, "title": title, "max_hops": max_hops},
            "people_at_company": len(targets),
            "reachable": len(candidates),
            "results": results,
        }

    def _describe(self, path: list[int], cost: float, target: Person, company: str, title: str) -> dict:
        people = self.resolution.people
        nodes = []
        prev_visible = None
        for i, pid in enumerate(path):
            p = people[pid]
            if i == 0:
                nodes.append({"kind": "you", "label": "You"})
            elif i == len(path) - 1:
                nodes.append({"kind": "target", "label": p.name, "position": p.position,
                              "company": p.company})
            elif p.is_member:
                nodes.append({"kind": "member", "label": p.name, "member_id": p.member_id})
            elif i == 1:
                # Your own direct connection: it's from your own export.
                nodes.append({"kind": "connection", "label": p.name})
            else:
                # Privacy: non-members deeper in the chain stay anonymous.
                # Only the person next to them knows who they are.
                nodes.append({"kind": "hidden", "label": f"a contact of {prev_visible.split()[0]}"})
            if i > 0 and nodes[-1]["kind"] != "hidden":
                prev_visible = p.name
        hops = []
        for a, b in zip(path, path[1:]):
            tie = self.ties[(min(a, b), max(a, b))]
            hops.append({"probability": round(tie.probability, 3), "reasons": tie.reasons})
        return {
            "target": {"name": target.name, "position": target.position, "company": target.company},
            "probability": round(cost_to_probability(cost), 4),
            "hops": len(path) - 1,
            "path": nodes,
            "hop_details": hops,
            "draft": self._draft(path, nodes, company, title),
        }

    def _draft(self, path: list[int], nodes: list[dict], company: str, title: str) -> dict:
        """Template-based intro request (no LLM needed)."""
        people = self.resolution.people
        me = people[path[0]]
        my_first = me.name.split()[0]
        target = nodes[-1]
        interest = f" (I'm most interested in {title.strip()} roles)" if title.strip() else ""
        if len(path) == 2:
            to = target["label"].split()[0]
            body = (
                f"Hi {to},\n\nI hope you're doing well! I'm applying to "
                f"{target['company']}{interest} and would really value hearing about your "
                f"experience on the team. Would you be open to a 15-minute chat in "
                f"the next couple of weeks?\n\nThanks so much,\n{my_first}"
            )
            return {"to": target["label"], "body": body}

        to = nodes[1]["label"]
        if len(path) == 3:
            via = f"{target['label']} ({target['position']})"
        else:
            middle = nodes[2]
            via_person = middle["label"] if middle["kind"] != "hidden" else "someone you know"
            via = f"{target['label']} ({target['position']}) through {via_person}"
        body = (
            f"Hi {to.split()[0]},\n\nHope things are going well! I'm applying to "
            f"{target['company']}{interest}, and I noticed you might be able to connect me "
            f"with {via}. Would you be comfortable making an introduction? I'm happy "
            f"to send a short blurb about me that you can forward.\n\n"
            f"No worries at all if not. Thanks either way!\n{my_first}"
        )
        return {"to": to, "body": body}
