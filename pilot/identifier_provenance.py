"""Where did each quoted identifier come from?

For every identifier from identifier_census.py, find when each agent's own tools first showed it.
Labels (first mentions): own_tool_before, other_tool_before, own_tool_after_only, none.
Later quotes of the same token: echo_own, echo_other.

Out: identifier_provenance.jsonl, identifier_provenance_by_agent_month.csv, untraced_windows.jsonl
"""
import argparse
import bisect
import collections
import csv
import datetime as dt
import json
import os
import re

import common

RUN_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,64}(?![0-9A-Za-z])", re.I)
LABELS = ["own_tool_before", "other_tool_before", "own_tool_after_only", "none", "echo_own", "echo_other"]


def label_claim(claim_time, agent_id, seen, tolerance):
    """Label a first mention from {agent_id: first time that agent's tools showed the token}."""
    own = seen.get(agent_id)
    others = [t for a, t in seen.items() if a != agent_id]
    if own is not None and own <= claim_time + tolerance:
        return "own_tool_before"
    if any(t <= claim_time + tolerance for t in others):
        return "other_tool_before"
    return "own_tool_after_only" if own is not None else "none"


def write_totals(rows, path):
    """Counts by label for all identifiers and for those next to a commit word."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "first_mentions"] + LABELS)
        for name, keep in (("all", lambda r: True), ("near_commit_word", lambda r: r["near_commit_word"])):
            sub = [r for r in rows if keep(r)]
            firsts = collections.Counter(r["label"] for r in sub if r["kind"] == "first")
            w.writerow([name, sum(firsts.values())] + [sum(1 for r in sub if r["label"] == label) for label in LABELS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="identifier_claims.jsonl")
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--tolerance-min", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=5, help="turns listed before each untraced quote")
    ap.add_argument("--skip-claude-code", action="store_true")
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    tol = dt.timedelta(minutes=args.tolerance_min)

    claims, seen_keys = [], set()
    with open(args.claims, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["message_id"], r["token"])
            if key not in seen_keys:
                seen_keys.add(key)
                r["t"] = common.parse_time(r["created_at"])
                claims.append(r)
    claims.sort(key=lambda r: (r["t"], r["message_id"]))
    wanted = {r["token"].lower() for r in claims}
    lens = sorted({len(t) for t in wanted})

    session_agent = {r["id"]: r["agent_id"]
                     for r in common.read_rows("computer_use_sessions.jsonl.gz", args.local_dir, token)}
    first_seen = collections.defaultdict(dict)

    def note(text, agent, t):
        for m in RUN_RE.finditer(text):
            run = m.group(0).lower()
            for n in lens:
                if len(run) >= n and run[:n] in wanted:
                    prev = first_seen[run[:n]].get(agent)
                    if prev is None or t < prev:
                        first_seen[run[:n]][agent] = t

    index = collections.defaultdict(list)
    n_turns = 0
    for r in common.read_rows("computer_use_turns.jsonl.gz", args.local_dir, token):
        n_turns += 1
        if args.max_turns and n_turns > args.max_turns:
            break
        if n_turns % 100000 == 0:
            print(f"  {n_turns} turns", flush=True)
        agent = session_agent.get(r.get("session_id"))
        if agent is None:
            continue
        t = common.parse_time(r["created_at"])
        out, err = common.as_text(r.get("output")), common.as_text(r.get("error"))
        index[agent].append((t, r.get("id"), r.get("session_id"),
                             bool(r.get("screenshot_is_redacted")), out[:300], err[:300]))
        if out or err:
            note(f"{out}\n{err}", agent, t)
    print(f"{n_turns} turns read")

    if not args.skip_claude_code:
        for r in common.read_rows("claude_code_messages.jsonl.gz", args.local_dir, token):
            if r.get("message_type") == "user":
                text = "\n".join(common.tool_result_texts(r.get("content")))
                if text:
                    note(text, r.get("agent_id"), common.parse_time(r["created_at"]))

    first_poster, rows = {}, []
    for r in claims:
        tok, aid = r["token"].lower(), r["agent_id"]
        if tok in first_poster:
            kind = label = "echo_own" if first_poster[tok] == aid else "echo_other"
        else:
            first_poster[tok] = aid
            kind = "first"
            label = label_claim(r["t"], aid, first_seen.get(tok, {}), tol)
        rows.append({"message_id": r["message_id"], "agent": r["agent"], "agent_id": aid,
                     "created_at": r["created_at"], "token": r["token"],
                     "near_commit_word": r["near_commit_word"], "kind": kind, "label": label,
                     "context": r["context"]})

    with open(os.path.join(args.out_dir, "identifier_provenance.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for lst in index.values():
        lst.sort(key=lambda x: x[0])
    times = {a: [x[0] for x in lst] for a, lst in index.items()}
    with open(os.path.join(args.out_dir, "untraced_windows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            if r["kind"] != "first" or r["label"] != "none":
                continue
            lst = index.get(r["agent_id"], [])
            i = bisect.bisect_left(times.get(r["agent_id"], []), common.parse_time(r["created_at"]))
            f.write(json.dumps({
                "message_id": r["message_id"], "agent": r["agent"], "created_at": r["created_at"],
                "token": r["token"], "context": r["context"],
                "turns_before": [{"turn_id": x[1], "created_at": str(x[0]), "session_id": x[2],
                                  "screenshot_is_redacted": x[3], "output_start": x[4], "error_start": x[5]}
                                 for x in lst[max(0, i - args.window):i]],
            }, ensure_ascii=False) + "\n")

    by = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["near_commit_word"]:
            by[(r["agent"], r["created_at"][:7])][r["label"]] += 1
    with open(os.path.join(args.out_dir, "identifier_provenance_by_agent_month.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "month"] + LABELS)
        for key in sorted(by, key=lambda k: (k[0] or "", k[1])):
            w.writerow([*key] + [by[key][label] for label in LABELS])
    write_totals(rows, os.path.join(args.out_dir, "identifier_provenance_totals.csv"))
    firsts = collections.Counter(r["label"] for r in rows if r["kind"] == "first")
    total = sum(firsts.values())
    print(dict(firsts))
    if total:
        print(f"first mentions: {total}, found in own tool output before the quote: {firsts['own_tool_before'] / total:.1%}")


if __name__ == "__main__":
    main()
