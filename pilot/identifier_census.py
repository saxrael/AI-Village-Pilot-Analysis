"""Census of commit-hash-like identifiers that agents quote in chat.

Out: identifier_census_by_agent_month.csv, identifier_claims.jsonl (input to identifier_provenance.py).
"""
import argparse
import collections
import csv
import json
import os
import re

import common

HASH_RE = re.compile(r"(?<![0-9A-Za-z_\-])([0-9a-f]{7,12}|[0-9a-f]{40})(?![0-9A-Za-z_\-])")
CONTEXT_RE = re.compile(r"(commit|sha|hash|pushed|push|merged|revert|cherry-pick|rebase)", re.I)


def looks_like_hash(tok):
    return any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)


def find_identifiers(text):
    """(token, near_commit_word, context) for each hash-like token in a message."""
    out = []
    for m in HASH_RE.finditer(text):
        if not looks_like_hash(m.group(1)):
            continue
        near = bool(CONTEXT_RE.search(text[max(0, m.start() - 40):m.start()]))
        context = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
        out.append((m.group(1), near, context))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    names = common.agent_names(args.local_dir, token)

    msgs, with_id, ids, near_ids = (collections.Counter() for _ in range(4))
    claims_path = os.path.join(args.out_dir, "identifier_claims.jsonl")
    total = 0
    with open(claims_path, "w", encoding="utf-8") as out:
        for r in common.read_rows("chat_messages.jsonl.gz", args.local_dir, token):
            total += 1
            if r.get("speaker_type") != "agent":
                continue
            aid = r.get("agent_speaker_id")
            name = names.get(aid, aid or "unknown")
            created = str(r.get("created_at") or "")
            key = (name, created[:7])
            msgs[key] += 1
            found = find_identifiers(r.get("content") or "")
            if not found:
                continue
            with_id[key] += 1
            for tok, near, context in found:
                ids[key] += 1
                near_ids[key] += int(near)
                out.write(json.dumps({"message_id": r.get("id"), "agent_id": aid, "agent": name,
                                      "created_at": created, "token": tok,
                                      "near_commit_word": near, "context": context},
                                     ensure_ascii=False) + "\n")

    csv_path = os.path.join(args.out_dir, "identifier_census_by_agent_month.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "month", "agent_messages", "messages_with_identifier",
                    "identifiers", "identifiers_near_commit_word"])
        for key in sorted(msgs, key=lambda k: (k[0] or "", k[1])):
            w.writerow([*key, msgs[key], with_id[key], ids[key], near_ids[key]])
    print(f"{total} chat rows, {sum(ids.values())} identifiers -> {csv_path}, {claims_path}")


if __name__ == "__main__":
    main()
