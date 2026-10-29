"""Match cold-verification claims in chat to the same agent's computer-use logs.

For each message reporting a cold verification, look in that agent's own turns from shortly before
the post for the quoted check count, the quoted hash and the script it names.

Out: claim_evidence.jsonl, claim_evidence_by_family.csv, claim_evidence_by_agent.csv, hand_check_sample.csv
"""
import argparse
import bisect
import collections
import csv
import datetime as dt
import json
import os
import random
import re

import common
from term_diffusion import SEEDS

CLAIM_RE = re.compile(SEEDS["cold_verification"], re.I)
COMPLETE_RE = re.compile(
    r"cold[- ]?verified|cold[- ]verif\w*\s+(?:complete\w*|done|passed|confirmed|result)"
    r"|cold[- ](?:run|check)\s+(?:done|complete\w*|passed|confirmed)"
    r"|(?:did|completed|ran|finished|performed)\s+(?:a|the|my|an)\s+(?:independent\s+)?cold[- ]?(?:run|verif\w*|check)"
    r"|cold[- ]?(?:run|verification)\s?:", re.I)
SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(\[])|\n+|;\s+|\s[—–]\s|•")
HEX_RE = re.compile(r"(?<![0-9A-Za-z_/\-])([0-9a-f]{7,64})(?![0-9A-Za-z_])", re.I)
FRAC_RE = re.compile(r"(?<![\d/.,])(\d{1,3}(?:,\d{3})+|\d+)\s?/\s?(\d{1,3}(?:,\d{3})+|\d+)(?![\d/])")
COUNT_RE = re.compile(r"(?<![\d,.])(\d{1,3}(?:,\d{3})+|\d{2,})\s+(?:checks?|assertions?|tests?|invariants?|cases?)\b", re.I)
SCRIPT_RE = re.compile(r"([\w.\-]+(?:/[\w.\-]+)*\.(?:py|sh|js|mjs|ts))\b")
SCRIPT_STOP = {"setup.py", "main.py", "index.js", "app.js", "test.py", "__init__.py", "config.py"}
CHECKOUT_RE = re.compile(r"git\s+(?:clone|checkout|fetch|pull|worktree|reset\s+--hard)|cloning into", re.I)
FAIL_RE = re.compile(r"traceback|assertionerror|\bFAIL\b|[1-9]\d*\s+(?:failed|failures)|exit(?: code)?[: ]+[1-9]\b", re.I)
PASS_RE = re.compile(r"pass|\bok\b|✓|✔|success|checks|exit|verified|all ", re.I)
LABELS = ["verified_strong", "verified_weak", "no_log_evidence", "no_log_text", "no_specifics"]


def num(s):
    return int(s.replace(",", ""))


def hex_heads(text):
    out = []
    for m in HEX_RE.finditer(text):
        h = m.group(1).lower()
        if any(c.isdigit() for c in h) and any(c.isalpha() for c in h):
            out.append(h)
    return out


def snippet(text, pos, width=90):
    return " ".join(text[max(0, pos - width):pos + width].split())


def extract_claims(msgs, names, min_count):
    claims = []
    for idx, (ts, mid, actor, content) in enumerate(msgs):
        if actor == "HUMAN" or not CLAIM_RE.search(content):
            continue
        sents = [s for s in SPLIT_RE.split(content) if s and s.strip()]
        hit = {i for i, s in enumerate(sents) if CLAIM_RE.search(s)}
        span = sorted({j for i in hit for j in (i, i + 1) if j < len(sents)})
        text = " ".join(sents[j].strip() for j in span)
        fracs, counts = set(), set()
        for m in FRAC_RE.finditer(text):
            a, b = num(m.group(1)), num(m.group(2))
            if a == b and a >= min_count:
                fracs.add(f"{a}/{a}")
                counts.add(a)
        for m in COUNT_RE.finditer(text):
            if num(m.group(1)) >= min_count:
                counts.add(num(m.group(1)))
        scripts = sorted({os.path.basename(m.group(1)).lower() for m in SCRIPT_RE.finditer(text)
                          if len(os.path.basename(m.group(1))) >= 8 and os.path.basename(m.group(1)).lower() not in SCRIPT_STOP})
        claims.append({
            "idx": idx, "message_id": mid, "agent_id": actor, "agent": names.get(actor, actor),
            "family": common.family(names.get(actor, actor)), "t": ts, "created_at": ts.isoformat(),
            "completion_phrase": bool(COMPLETE_RE.search(text)), "text": text[:500],
            "fracs": sorted(fracs), "counts": sorted(counts),
            "heads": sorted({h[:7] for h in hex_heads(text)}), "scripts": scripts,
        })
    return claims


def mark_prior_posts(claims, msgs, names, hours):
    """Flag claims whose fraction or 7-char hash another agent posted earlier within `hours`."""
    wanted = collections.defaultdict(list)
    for c in claims:
        for tkn in c["fracs"] + c["heads"]:
            wanted[tkn].append(c)
    seen = collections.defaultdict(list)
    for idx, (ts, _, actor, content) in enumerate(msgs):
        if actor == "HUMAN" or not wanted:
            continue
        toks = {f"{num(m.group(1))}/{num(m.group(2))}" for m in FRAC_RE.finditer(content)
                if m.group(1).replace(",", "") == m.group(2).replace(",", "")}
        toks |= {h[:7] for h in hex_heads(content)}
        for tkn in toks & wanted.keys():
            seen[tkn].append((idx, ts, actor))
    window = dt.timedelta(hours=hours)
    for c in claims:
        found = {}
        for tkn in c["fracs"] + c["heads"]:
            for idx, ts, actor in seen.get(tkn, []):
                if actor != c["agent_id"] and idx < c["idx"] and c["t"] - ts <= window and (actor not in found or ts < found[actor]):
                    found[actor] = ts
        c["prior_other_agents"] = [names.get(a, a) for a, _ in sorted(found.items(), key=lambda x: x[1])]
        c["prior_other"] = bool(found)


def scan_turns(claims, local_dir, token, window_min, tolerance_min, max_turns=0):
    window, tol = dt.timedelta(minutes=window_min), dt.timedelta(minutes=tolerance_min)
    by_agent = collections.defaultdict(list)
    for ci, c in enumerate(claims):
        by_agent[c["agent_id"]].append((c["t"], ci))
    for lst in by_agent.values():
        lst.sort()
    times = {a: [x[0] for x in lst] for a, lst in by_agent.items()}
    ev = [{"turns": 0, "text_turns": 0, "count": None, "frac": None, "hash": None, "script": None,
           "checkout": False, "fail": False} for _ in claims]

    def handle(cmd, outp, agent, t):
        if agent not in times:
            return
        lo = bisect.bisect_left(times[agent], t - tol)
        hi = bisect.bisect_right(times[agent], t + window)
        if lo >= hi:
            return
        has_pass = bool(outp) and bool(PASS_RE.search(outp))
        fails = bool(outp) and bool(FAIL_RE.search(outp))
        checkout = bool(CHECKOUT_RE.search(cmd) or CHECKOUT_RE.search(outp[:4000]))
        low_c, low_o = cmd.lower(), outp.lower()
        for _, ci in by_agent[agent][lo:hi]:
            c, e = claims[ci], ev[ci]
            e["turns"] += 1
            if not (cmd or outp):
                continue
            e["text_turns"] += 1
            e["checkout"] |= checkout
            e["fail"] |= fails
            if outp and e["frac"] is None:
                for fr in c["fracs"]:
                    n = int(fr.split("/")[0])
                    for form in (str(n), format(n, ",")):
                        m = re.search(rf"(?<![\d,.]){re.escape(form)}\s?/\s?{re.escape(form)}(?![\d])", outp)
                        if m:
                            e["frac"] = snippet(outp, m.start())
                            break
                    if e["frac"]:
                        break
            if outp and has_pass and e["count"] is None:
                for n in c["counts"]:
                    m = re.search(rf"(?<![\d,.])(?:{n}|{format(n, ',')})(?![\d]|,\d)", outp)
                    if m:
                        e["count"] = snippet(outp, m.start())
                        break
            if e["hash"] is None:
                e["hash"] = next((h for h in c["heads"] if h in low_o or h in low_c), None)
            if e["script"] is None and cmd:
                for s in c["scripts"]:
                    if s in low_c:
                        e["script"] = snippet(cmd, low_c.find(s), 70)
                        break

    session_agent = {r["id"]: r["agent_id"] for r in common.read_rows("computer_use_sessions.jsonl.gz", local_dir, token)}
    n = 0
    for r in common.read_rows("computer_use_turns.jsonl.gz", local_dir, token):
        n += 1
        if max_turns and n > max_turns:
            break
        if n % 200000 == 0:
            print(f"  {n} turns", flush=True)
        agent = session_agent.get(r.get("session_id"))
        if agent is None:
            continue
        handle(common.as_text(r.get("agent_action")),
               f"{common.as_text(r.get('output'))}\n{common.as_text(r.get('error'))}".strip(),
               agent, common.parse_time(r["created_at"]))
    print(f"read {n} turns")
    return ev


def assign_labels(claims, ev):
    for c, e in zip(claims, ev):
        c.update(window_turns=e["turns"], window_text_turns=e["text_turns"],
                 count_in_own_output=e["count"] or e["frac"], hash_in_own_logs=e["hash"],
                 script_run=e["script"], fresh_checkout_seen=e["checkout"], failure_text_seen=e["fail"])
        if not (c["counts"] or c["heads"] or c["scripts"]):
            label = "no_specifics"
        elif e["text_turns"] == 0:
            label = "no_log_text"
        else:
            count_ok, other_ok = bool(e["count"] or e["frac"]), bool(e["script"] or e["hash"])
            label = ("verified_strong" if count_ok and other_ok
                     else "verified_weak" if count_ok or other_ok else "no_log_evidence")
        c["label"] = label
        c["reproduced_peer"] = label in ("verified_strong", "verified_weak") and c["prior_other"]


def summarise(claims, key, path):
    per = collections.defaultdict(collections.Counter)
    for c in claims:
        per[c[key]][c["label"]] += 1
        per[c[key]]["reproduced_peer"] += int(c["reproduced_peer"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([key, "claims"] + LABELS + ["reproduced_peer"])
        for g, cnt in sorted(per.items(), key=lambda kv: -sum(kv[1][label] for label in LABELS)):
            w.writerow([g, sum(cnt[label] for label in LABELS)] + [cnt[label] for label in LABELS] + [cnt["reproduced_peer"]])


def write_hand_check(claims, path, seed):
    rng = random.Random(seed)
    sample = []
    for label, k in (("verified_strong", 12), ("verified_weak", 8), ("no_log_evidence", 12), ("no_log_text", 6)):
        pool = [c for c in claims if c["label"] == label]
        sample += rng.sample(pool, min(k, len(pool)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "agent", "date", "claim_text", "counts", "hashes", "scripts", "own_output_snippet",
                    "own_command_snippet", "window_text_turns", "label_correct"])
        for c in sample:
            w.writerow([c["label"], c["agent"], c["created_at"][:10], c["text"][:400], c["counts"], c["heads"], c["scripts"],
                        c["count_in_own_output"] or "", c["script_run"] or "", c["window_text_turns"], ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--window-min", type=float, default=90.0)
    ap.add_argument("--tolerance-min", type=float, default=5.0)
    ap.add_argument("--prior-hours", type=float, default=72.0)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    names, msgs = common.load_chat(args.local_dir, token)

    claims = extract_claims(msgs, names, args.min_count)
    mark_prior_posts(claims, msgs, names, args.prior_hours)
    print(f"{len(claims)} messages with a cold-verification term, {sum(c['completion_phrase'] for c in claims)} with a completion phrase")
    assign_labels(claims, scan_turns(claims, args.local_dir, token, args.window_min, args.tolerance_min, args.max_turns))

    main_set = [c for c in claims if c["completion_phrase"]]
    with open(os.path.join(args.out_dir, "claim_evidence.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps({k: v for k, v in c.items() if k not in ("t", "idx")}, ensure_ascii=False) + "\n")
    summarise(main_set, "family", os.path.join(args.out_dir, "claim_evidence_by_family.csv"))
    summarise(main_set, "agent", os.path.join(args.out_dir, "claim_evidence_by_agent.csv"))
    write_hand_check(main_set, os.path.join(args.out_dir, "hand_check_sample.csv"), args.seed)
    counts = collections.Counter(c["label"] for c in main_set)
    for label in LABELS:
        print(f"{label:18}{counts[label]:>6}")


if __name__ == "__main__":
    main()
