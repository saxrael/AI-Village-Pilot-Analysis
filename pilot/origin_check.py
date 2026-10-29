"""Did a goal, a human or a system message use a term before the agents did?

Out: origin_check.csv (one row per term). Message context is printed only with --show-context.
"""
import argparse
import csv
import datetime as dt
import os
import re

import common
from term_diffusion import SEEDS, load_goal_text

EXTRA = {"independent_verification": r"independent(?:ly)?[- ](?:verif\w*|check\w*|replicat\w*)"}


def first_uses(msgs, rx, n=12):
    return [m for m in msgs if rx.search(m[3])][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    names, msgs = common.load_chat(args.local_dir, token)
    goal_text = load_goal_text(args.local_dir, token)
    patterns = {k: re.compile(v, re.I) for k, v in {**SEEDS, **EXTRA}.items()}

    rows = []
    for term, rx in patterns.items():
        hits = [m for m in msgs if rx.search(m[3])]
        agent_hits = [m for m in hits if m[2] != "HUMAN"]
        human_hits = [m for m in hits if m[2] == "HUMAN"]
        if not agent_hits:
            continue
        t0, _, who, _ = agent_hits[0]
        day_before = [m for m in msgs if m[2] == "HUMAN" and t0 - dt.timedelta(hours=24) <= m[0] <= t0]
        rows.append({
            "term": term, "first_agent_use": t0.isoformat(), "first_agent": names.get(who, who),
            "family": common.family(names.get(who, who)), "agent_messages": len(agent_hits),
            "human_messages": len(human_hits),
            "human_used_before_agents": bool(human_hits) and human_hits[0][0] <= t0,
            "in_goal_text": bool(rx.search(goal_text)),
            "human_messages_24h_before": len(day_before),
        })
        print(f"{term:26}{names.get(who, who):22}{t0.isoformat()[:16]}  human-first={rows[-1]['human_used_before_agents']}  in-goals={rows[-1]['in_goal_text']}")
        if args.show_context:
            for t, _, w, c in first_uses(msgs, rx):
                m = rx.search(c)
                print("   ", t.isoformat()[:16], names.get(w, w)[:22].ljust(22), "..." + " ".join(c[max(0, m.start() - 100):m.end() + 100].split()) + "...")

    if rows:
        with open(os.path.join(args.out_dir, "origin_check.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
