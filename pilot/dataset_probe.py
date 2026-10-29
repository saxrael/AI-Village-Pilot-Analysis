"""List the dataset's files and show the shape of each table without downloading the large ones.

Out: dataset_probe.md
"""
import argparse
import gzip
import json
import os

import common

SMALL = ["agents", "villages", "village_goals", "agent_goals", "chat_rooms"]
LARGE = ["chat_messages", "events", "computer_use_sessions", "computer_use_turns", "agent_memories", "summaries"]
DOCS = ["CHANGELOG.md", "SCHEMA.md", "README.md"]
HIDDEN = ("token", "key", "password", "secret", "authorization")


def size_text(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def shape(obj, depth=0):
    if isinstance(obj, dict):
        if depth >= 2:
            return "{...}"
        return {k: "<hidden>" if any(h in k.lower() for h in HIDDEN) else shape(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    if isinstance(obj, str):
        return f"str({len(obj)})"
    return type(obj).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_probe.md")
    args = ap.parse_args()
    token = common.require_token(None)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    sizes = {e.path: e.size for e in api.list_repo_tree(common.REPO, repo_type="dataset", recursive=True)
             if getattr(e, "size", None) is not None}
    lines = [f"# {common.REPO}", "", "## Files"]
    lines += [f"- {p}: {size_text(s)}" for p, s in sorted(sizes.items()) if not p.startswith("images/")]
    images = [s for p, s in sizes.items() if p.startswith("images/")]
    if images:
        lines.append(f"- images/ ({len(images)} files): {size_text(sum(images))}")

    for name in DOCS:
        if name in sizes:
            text = open(hf_hub_download(common.REPO, name, repo_type="dataset", token=token), encoding="utf-8").read()
            lines += ["", f"## {name}", "```", text[:30000], "```"]

    lines += ["", "## Tables"]
    for table in SMALL + LARGE:
        name = f"{table}.jsonl.gz"
        if name not in sizes:
            continue
        with common.open_stream(name, None, token) as f:
            first = json.loads(f.readline())
        lines += [f"### {name} ({size_text(sizes[name])})", "```json", json.dumps(shape(first), indent=1), "```"]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
