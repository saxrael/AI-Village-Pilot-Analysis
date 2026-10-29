"""Do "pushed" claims have a successful push in the agent's own tool logs?

Reads identifier_provenance.jsonl. For each first-mention quote whose context says "pushed",
looks for a git push result in the agent's tool output in the window before the quote.
Labels: push_confirmed_hash, push_success_other, push_failed_only, no_push_evidence.
Also re-matches untraced identifiers by their start or end (long hashes are often quoted in part).

Out: push_check.jsonl, push_check_by_agent.csv, rematch_untraced.csv
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
PUSHED_RE = re.compile(r"(?<!not )(?<!n't )(?<!never )(?<!un)\bpushed\b", re.I)
TO_LINE_RE = re.compile(r"^To\s+(https?://|git@|ssh://|\S+\.git)", re.M)
FETCH_REF_RE = re.compile(r"^(origin|upstream)/", re.I)
PUSH_OK_RE = re.compile(r"\b([0-9a-f]{7,40})\.\.\.?([0-9a-f]{7,40})\s+(\S+)\s*->\s*(\S+)", re.I)
NEW_REF_RE = re.compile(r"\*\s*\[new (?:branch|tag)\]\s+(\S+)\s*->\s*(\S+)", re.I)
PUSH_FAIL_RE = re.compile(
    r"(!\s*\[(remote )?rejected\]|error: failed to push|non-fast-forward|\[remote rejected\]"
    r"|fatal: .{0,80}\b(push|remote)\b|Permission denied \(publickey\)|403)", re.I)
LABELS = ["push_confirmed_hash", "push_success_other", "push_failed_only", "no_push_evidence"]


def hash_match(a, b):
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def push_evidence(text):
    """(new commit hashes pushed, new branch or tag pushed, failure seen). Fetch lines do not count."""
    if not TO_LINE_RE.search(text):
        return [], False, bool(PUSH_FAIL_RE.search(text))
    ok = [m.group(2) for m in PUSH_OK_RE.finditer(text) if not FETCH_REF_RE.match(m.group(4))]
    new_ref = any(not FETCH_REF_RE.match(m.group(2)) for m in NEW_REF_RE.finditer(text))
    return ok, new_ref, bool(PUSH_FAIL_RE.search(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", default="identifier_provenance.jsonl")
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--tolerance-min", type=float, default=5.0)
    ap.add_argument("--skip-claude-code", action="store_true")
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    window = dt.timedelta(minutes=args.window_min)
    tol = dt.timedelta(minutes=args.tolerance_min)

    prov = []
    with open(args.provenance, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["t"] = common.parse_time(r["created_at"])
            prov.append(r)
    pushed = [r for r in prov if r["kind"] == "first" and PUSHED_RE.search(r["context"])]
    untraced = [r for r in prov if r["kind"] == "first" and r["label"] in ("none", "own_tool_after_only")]

    by_agent = collections.defaultdict(list)
    for i, r in enumerate(pushed):
        by_agent[r["agent_id"]].append((r["t"], i))
    for lst in by_agent.values():
        lst.sort()
    agent_times = {a: [x[0] for x in lst] for a, lst in by_agent.items()}
    ev = [{"confirmed": None, "other": 0, "failed": 0, "line": None} for _ in pushed]

    rem_tokens = {r["token"].lower() for r in untraced}
    rem_lens = sorted({len(t) for t in rem_tokens})
    rem_seen = collections.defaultdict(dict)

    def handle(text, agent, t):
        for m in RUN_RE.finditer(text):
            run = m.group(0).lower()
            for n in rem_lens:
                if len(run) >= n:
                    for key in (run[:n], run[-n:]):
                        if key in rem_tokens and (agent not in rem_seen[key] or t < rem_seen[key][agent]):
                            rem_seen[key][agent] = t
        if agent not in by_agent:
            return
        lo = bisect.bisect_left(agent_times[agent], t - tol)
        hi = bisect.bisect_right(agent_times[agent], t + window)
        if lo >= hi or not any(k in text.lower() for k in ("->", "rejected", "failed to push", "fatal")):
            return
        ok, new_ref, failed = push_evidence(text)
        if not (ok or new_ref or failed):
            return
        line = next((ln.strip()[:200] for ln in text.splitlines()
                     if "->" in ln or "rejected" in ln.lower() or "failed to push" in ln.lower()), None)
        for _, ci in by_agent[agent][lo:hi]:
            e = ev[ci]
            if any(hash_match(pushed[ci]["token"], h) for h in ok):
                if e["confirmed"] is None or t < e["confirmed"]:
                    e["confirmed"], e["line"] = t, line
            elif ok or new_ref:
                e["other"] += 1
                e["line"] = e["line"] or line
            elif failed:
                e["failed"] += 1
                e["line"] = e["line"] or line

    session_agent = {r["id"]: r["agent_id"]
                     for r in common.read_rows("computer_use_sessions.jsonl.gz", args.local_dir, token)}
    n_turns = 0
    for r in common.read_rows("computer_use_turns.jsonl.gz", args.local_dir, token):
        n_turns += 1
        if args.max_turns and n_turns > args.max_turns:
            break
        if n_turns % 200000 == 0:
            print(f"  {n_turns} turns", flush=True)
        agent = session_agent.get(r.get("session_id"))
        out, err = common.as_text(r.get("output")), common.as_text(r.get("error"))
        if agent and (out or err):
            handle(f"{out}\n{err}", agent, common.parse_time(r["created_at"]))
    if not args.skip_claude_code:
        for r in common.read_rows("claude_code_messages.jsonl.gz", args.local_dir, token):
            if r.get("message_type") == "user":
                text = "\n".join(common.tool_result_texts(r.get("content")))
                if text:
                    handle(text, r.get("agent_id"), common.parse_time(r["created_at"]))

    rows, per = [], collections.defaultdict(collections.Counter)
    for r, e in zip(pushed, ev):
        label = ("push_confirmed_hash" if e["confirmed"] is not None else
                 "push_success_other" if e["other"] else
                 "push_failed_only" if e["failed"] else "no_push_evidence")
        per[r["agent"]][label] += 1
        rows.append({"message_id": r["message_id"], "agent": r["agent"], "created_at": r["created_at"],
                     "token": r["token"], "push_label": label, "evidence_line": e["line"], "context": r["context"]})
    with open(os.path.join(args.out_dir, "push_check.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "push_check_by_agent.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "pushed_claims"] + LABELS)
        for a, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
            w.writerow([a, sum(c.values())] + [c[label] for label in LABELS])

    rem_rows = []
    for r in untraced:
        seen = rem_seen.get(r["token"].lower(), {})
        own = seen.get(r["agent_id"])
        others = [t for a, t in seen.items() if a != r["agent_id"]]
        if own is not None and own <= r["t"] + tol:
            label = "rescued_own_before"
        elif any(t <= r["t"] + tol for t in others):
            label = "rescued_other_before"
        elif own is not None:
            label = "own_after_only"
        else:
            label = "still_untraced"
        rem_rows.append((r["agent"], r["created_at"], r["token"], r["label"], label))
    with open(os.path.join(args.out_dir, "rematch_untraced.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "created_at", "token", "provenance_label", "rematch_label"])
        w.writerows(rem_rows)
    total = collections.Counter()
    for c in per.values():
        total.update(c)
    print(f"{len(rows)} pushed claims:", dict(total))
    print("rematch:", dict(collections.Counter(x[4] for x in rem_rows)))


if __name__ == "__main__":
    main()
