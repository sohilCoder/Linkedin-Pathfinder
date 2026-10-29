"""Entity resolution: deciding which rows across exports are the same person.

Every member's export mentions people independently, so "Jon Smith" in one
file and "Jonathan Smith, PhD" in another may be one person, or two. The
matcher works in two passes:

1. **Exact key.** Mentions sharing a normalized profile URL are merged.
2. **Fuzzy key.** Mentions without a URL are compared inside small "blocks"
   (same normalized last name *and* company) and merged when their first
   names are compatible (equal, a known nickname, or a bare initial).

Two safety rules stop bad merges:

* **Cannot-link:** two clusters that carry *different* profile URLs are
  never merged. That is how namesakes at the same company stay separate.
* **No guessing:** if a URL-less mention could belong to more than one known
  person, it is left on its own rather than merged with an arbitrary one.

Blocking keeps the comparison cost close to linear: we only compare within
blocks, which are tiny in practice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .loader import ConnectionRow, MemberProfile
from .normalize import (
    first_names_compatible,
    normalize_company,
    normalize_first_name,
    normalize_last_name,
    normalize_url,
)

MentionKey = tuple  # ("member", member_id) or ("row", member_id, row_index)


@dataclass
class Mention:
    key: MentionKey
    first: str
    last: str
    url: str
    company: str
    position: str
    nfirst: str = ""
    nlast: str = ""
    nurl: str = ""
    ncompany: str = ""

    def __post_init__(self) -> None:
        self.nfirst = normalize_first_name(self.first)
        self.nlast = normalize_last_name(self.last)
        self.nurl = normalize_url(self.url)
        self.ncompany = normalize_company(self.company)

    @classmethod
    def from_row(cls, member_id: str, row: ConnectionRow) -> "Mention":
        return cls(("row", member_id, row.row_index), row.first_name,
                   row.last_name, row.url, row.company, row.position)

    @classmethod
    def from_profile(cls, profile: MemberProfile) -> "Mention":
        return cls(("member", profile.member_id), profile.first_name,
                   profile.last_name, profile.url, profile.company, profile.position)


class ConstrainedUnionFind:
    """Union-find whose clusters remember their URLs and first names, and
    refuse merges that would put two different identities in one cluster."""

    def __init__(self, mentions: list[Mention]):
        self.parent = list(range(len(mentions)))
        self.size = [1] * len(mentions)
        self.urls: list[set[str]] = [{m.nurl} if m.nurl else set() for m in mentions]
        self.names: list[set[str]] = [
            {m.nfirst} if len(m.nfirst) > 1 else set() for m in mentions
        ]
        self.refused = 0

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]   # path halving
            i = self.parent[i]
        return i

    def can_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.urls[ra] and self.urls[rb] and self.urls[ra] != self.urls[rb]:
            return False   # cannot-link: different profile URLs
        names_a, names_b = self.names[ra], self.names[rb]
        if not names_a or not names_b:
            return True
        if self.urls[ra] or self.urls[rb]:
            # A URL-backed cluster is a verified identity whose names are
            # known aliases ("Elizabeth", "Libby"): matching any one is enough.
            return any(first_names_compatible(x, y) for x in names_a for y in names_b)
        # Two unverified clusters: every name must fit, which stops chains
        # like "John" ~ "J." ~ "Jane" from collapsing into one person.
        return all(first_names_compatible(x, y) for x in names_a for y in names_b)

    def union(self, a: int, b: int, *, force: bool = False) -> bool:
        """Merge two clusters. `force` skips the name check (same URL)."""
        if not force and not self.can_union(a, b):
            self.refused += 1
            return False
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.urls[ra] |= self.urls[rb]
        self.names[ra] |= self.names[rb]
        return True


@dataclass
class Person:
    """One resolved real-world person."""

    person_id: int
    name: str
    url: str
    company: str          # display form, e.g. "Meta"
    ncompany: str         # matching key, e.g. "meta"
    position: str
    member_id: str | None = None
    mentions: list[MentionKey] = field(default_factory=list)

    @property
    def is_member(self) -> bool:
        return self.member_id is not None


@dataclass
class Resolution:
    people: list[Person]
    person_of: dict[MentionKey, int]       # mention key -> person_id
    stats: dict


def resolve(mentions: list[Mention], *, fuzzy: bool = True) -> Resolution:
    """Cluster mentions into people. `fuzzy=False` gives the URL-only baseline."""
    uf = ConstrainedUnionFind(mentions)

    # Pass 1: identical profile URLs are the same person.
    by_url: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        if m.nurl:
            by_url[m.nurl].append(i)
    for idxs in by_url.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j, force=True)   # same URL is authoritative

    ambiguous = 0
    if fuzzy:
        # Pass 2: compare URL-less mentions within (last name, company)
        # blocks. Companies are pooled per cluster first, so a URL-backed
        # mention with a blank company still joins the blocks of the
        # companies its other mentions list.
        cluster_companies: dict[int, set[str]] = defaultdict(set)
        for i, m in enumerate(mentions):
            if m.ncompany:
                cluster_companies[uf.find(i)].add(m.ncompany)
        blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, m in enumerate(mentions):
            if m.nlast and m.nfirst:
                for company in cluster_companies.get(uf.find(i), ()):
                    blocks[(m.nlast, company)].append(i)

        for idxs in blocks.values():
            if len(idxs) < 2:
                continue
            for i in idxs:
                if mentions[i].nurl:
                    continue
                own = uf.find(i)
                if uf.urls[own]:
                    continue            # already attached to a known identity
                # Known identities (clusters with a URL) this mention could be.
                url_roots = {
                    uf.find(j) for j in idxs
                    if uf.find(j) != own and uf.urls[uf.find(j)]
                    and first_names_compatible(mentions[i].nfirst, mentions[j].nfirst)
                }
                if len(url_roots) == 1:
                    uf.union(i, next(iter(url_roots)))
                elif len(url_roots) > 1:
                    ambiguous += 1      # could be either namesake: don't guess
            # URL-less mentions that matched no known identity: merge with
            # each other when compatible (same unknown person seen twice).
            loose = [i for i in idxs if not uf.urls[uf.find(i)]]
            for a_pos, a in enumerate(loose):
                for b in loose[a_pos + 1:]:
                    if first_names_compatible(mentions[a].nfirst, mentions[b].nfirst):
                        uf.union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(mentions)):
        clusters[uf.find(i)].append(i)

    people: list[Person] = []
    person_of: dict[MentionKey, int] = {}
    for idxs in clusters.values():
        pid = len(people)
        group = [mentions[i] for i in idxs]
        member = next((m for m in group if m.key[0] == "member"), None)
        people.append(_summarize(pid, group, member))
        for m in group:
            person_of[m.key] = pid

    stats = {
        "mentions": len(mentions),
        "people": len(people),
        "merges_refused": uf.refused,
        "ambiguous_left_unmerged": ambiguous,
    }
    return Resolution(people, person_of, stats)


def _summarize(pid: int, group: list[Mention], member: Mention | None) -> Person:
    """Pick the best display values for a cluster: the member's own profile
    wins, otherwise the most common spelling across exports."""

    def most_common(values: list[str]) -> str:
        values = [v for v in values if v]
        return Counter(values).most_common(1)[0][0] if values else ""

    if member is not None:
        name = f"{member.first} {member.last}".strip()
        company, position = member.company, member.position
    else:
        full = [f"{m.first} {m.last}".strip() for m in group if len(m.nfirst) > 1]
        name = most_common(full) or f"{group[0].first} {group[0].last}".strip()
        company = most_common([m.company for m in group])
        position = most_common([m.position for m in group])
    return Person(
        person_id=pid,
        name=name,
        url=most_common([m.url for m in group]),
        company=company,
        ncompany=normalize_company(company),
        position=position,
        member_id=member.key[1] if member else None,
        mentions=[m.key for m in group],
    )


def pairwise_scores(predicted: dict[MentionKey, int], truth: dict[MentionKey, int]) -> dict:
    """Pairwise precision/recall of a clustering against ground truth.

    A "pair" is two mentions placed in the same cluster. Precision: of the
    pairs we merged, how many are truly the same person. Recall: of the
    true same-person pairs, how many we merged.
    """
    keys = [k for k in truth if k in predicted]
    joint = Counter((predicted[k], truth[k]) for k in keys)
    pred_sizes = Counter(predicted[k] for k in keys)
    true_sizes = Counter(truth[k] for k in keys)

    def pairs(n: int) -> int:
        return n * (n - 1) // 2

    tp = sum(pairs(n) for n in joint.values())
    pred_pairs = sum(pairs(n) for n in pred_sizes.values())
    true_pairs = sum(pairs(n) for n in true_sizes.values())
    precision = tp / pred_pairs if pred_pairs else 1.0
    recall = tp / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_pairs": true_pairs, "predicted_pairs": pred_pairs, "correct_pairs": tp,
    }
