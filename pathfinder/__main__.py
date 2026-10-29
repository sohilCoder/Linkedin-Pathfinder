"""Command line: python -m pathfinder <command>

    simulate   write a fake network to a folder
    eval       score the matcher on simulated data
    search     find intro paths from the terminal
    serve      run the web app
"""

from __future__ import annotations

import argparse
import json
import sys

from . import evaluate, simulate
from .network import Network


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathfinder", description="Warm Intro Pathfinder")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="generate a simulated network")
    p.add_argument("--out", default="data/demo")
    p.add_argument("--people", type=int, default=2500)
    p.add_argument("--members", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("eval", help="score person matching against ground truth")
    p.add_argument("--data", default="data/demo")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("search", help="find intro paths")
    p.add_argument("--data", default="data/demo")
    p.add_argument("--member", help="member id to search as (omit to list members)")
    p.add_argument("--company")
    p.add_argument("--title", default="")
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--data", default="data/demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "simulate":
        manifest = simulate.generate(args.out, n_people=args.people,
                                     n_members=args.members, seed=args.seed)
        print(f"Wrote {len(manifest['members'])} member exports "
              f"({manifest['rows_written']} rows) to {args.out}")
        print("Members:", ", ".join(manifest["members"]))
        return 0

    if args.command == "eval":
        report = evaluate.evaluate(args.data)
        print(json.dumps(report, indent=2) if args.json else evaluate.format_report(report))
        return 0

    if args.command == "search":
        net = Network.load(args.data, persist=False)
        if not args.member or not args.company:
            print("Members:", ", ".join(m["member_id"] for m in net.members()))
            print("Usage: --member <id> --company <name> [--title keyword]")
            return 1
        try:
            result = net.search(args.member, args.company, title=args.title,
                                max_hops=args.hops, limit=args.limit)
        except KeyError:
            print(f"Unknown member {args.member!r}. Members:",
                  ", ".join(m["member_id"] for m in net.members()))
            return 1
        print(f"{result['people_at_company']} people at {args.company} in the network, "
              f"{result['reachable']} reachable within {args.hops} hops\n")
        for i, r in enumerate(result["results"], 1):
            chain = "  ->  ".join(n["label"] for n in r["path"])
            print(f"{i}. {r['probability']:.0%}  {chain}")
            print(f"   target: {r['target']['position']} at {r['target']['company']}")
            for hop in r["hop_details"]:
                print(f"   - {hop['probability']:.0%}: {', '.join(hop['reasons'])}")
            print()
        return 0

    if args.command == "serve":
        from .server import serve
        serve(Network.load(args.data), args.host, args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
