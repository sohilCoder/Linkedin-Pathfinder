"""Score the matcher against a simulated dataset's ground truth."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .loader import read_text_file
from .matching import Mention, pairwise_scores, resolve
from .network import Network


def load_truth(data_dir: Path) -> dict:
    manifest = json.loads(read_text_file(data_dir / "manifest.json"))
    truth = {("member", mid): pid for mid, pid in manifest["members"].items()}
    with open(data_dir / "ground_truth.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            truth[("row", row["member_id"], int(row["row_index"]))] = int(row["person_id"])
    return truth


def evaluate(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    if not (data_dir / "ground_truth.csv").exists():
        raise FileNotFoundError(
            f"{data_dir} has no ground_truth.csv; evaluation needs simulated data"
        )
    net = Network.load(data_dir, persist=False)
    mentions: list[Mention] = []
    for mid, up in net.uploads.items():
        mentions.append(Mention.from_profile(up.profile))
        mentions.extend(Mention.from_row(mid, r) for r in up.rows)
    truth = load_truth(data_dir)

    report = {"mentions": len(mentions), "true_people": len(set(truth.values()))}
    for label, fuzzy in (("url_only_baseline", False), ("pathfinder_matcher", True)):
        res = resolve(mentions, fuzzy=fuzzy)
        report[label] = {
            "people_found": len(res.people),
            **pairwise_scores(res.person_of, truth),
        }
    return report


def format_report(report: dict) -> str:
    lines = [
        f"Mentions across all exports: {report['mentions']}",
        f"Distinct real people:        {report['true_people']}",
        "",
        f"{'':22}{'people':>8}{'precision':>11}{'recall':>9}{'F1':>8}",
    ]
    for label in ("url_only_baseline", "pathfinder_matcher"):
        r = report[label]
        lines.append(
            f"{label:22}{r['people_found']:>8}{r['precision']:>11.1%}{r['recall']:>9.1%}{r['f1']:>8.3f}"
        )
    return "\n".join(lines)
