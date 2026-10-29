"""Was an untraced identifier visible on the agent's screen before it was quoted?

Samples quotes from untraced_windows.jsonl (minus those rescued by push_check.py), downloads only the
needed day archives of screenshots, OCRs the agent's last screenshots and searches for the identifier.
A feasibility test on a few days: the hit rate shows whether the method works, not how common it is.
Needs tesseract (apt-get install tesseract-ocr) and pytesseract, pillow.

Out: screenshot_check.csv, screenshot_samples.zip (candidate hits and last screenshots, for visual review)
"""
import argparse
import collections
import csv
import io
import json
import os
import random
import re
import tarfile
import zipfile
import zoneinfo

import common
import stats

PT = zoneinfo.ZoneInfo("America/Los_Angeles")
CONFUSE = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1", "|": "1", "i": "1", "S": "5", "s": "5", "B": "8", "g": "9"})
RUN_RE = re.compile(r"[0-9a-f]{5,}")


def pt_day(s):
    return common.parse_time(s).astimezone(PT).date().isoformat()


def near(key, run):
    """'exact' if key is in run, 'fuzzy' if it matches with one substituted character."""
    if key in run:
        return "exact"
    k = len(key)
    for i in range(len(run) - k + 1):
        if sum(a != b for a, b in zip(key, run[i:i + k])) <= 1:
            return "fuzzy"
    return None


def find_token(ocr_text, token):
    """Look for the first or last 8 characters of token in OCR text, allowing common OCR confusions."""
    token = token.lower()
    keys = {token[:8], token[-8:]} if len(token) >= 8 else {token}
    runs = set(RUN_RE.findall(ocr_text.lower().translate(CONFUSE))) | set(RUN_RE.findall(ocr_text.lower()))
    best = None
    for key in keys:
        for run in runs:
            hit = near(key, run)
            if hit == "exact":
                return "exact"
            best = best or hit
    return best


def pick_sample(rows, n, max_days, seed):
    day_counts = collections.Counter(r["_day"] for r in rows)
    days = {d for d, _ in day_counts.most_common(max_days)}
    pool = [r for r in rows if r["_day"] in days]
    random.Random(seed).shuffle(pool)
    by_agent = collections.defaultdict(list)
    for r in pool:
        by_agent[r["agent"]].append(r)
    sample = []
    while len(sample) < n and any(by_agent.values()):
        for agent in list(by_agent):
            if by_agent[agent] and len(sample) < n:
                sample.append(by_agent[agent].pop())
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="untraced_windows.jsonl")
    ap.add_argument("--rematch", help="rematch_untraced.csv from push_check.py")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-days", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--local-tar-dir", help="folder with <YYYY-MM-DD>.tar, for testing")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    with open(args.windows, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.rematch:
        with open(args.rematch, newline="", encoding="utf-8") as f:
            rescued = {(r["agent"], r["created_at"], r["token"])
                       for r in csv.DictReader(f) if r["rematch_label"].startswith("rescued")}
        rows = [r for r in rows if (r["agent"], r["created_at"], r["token"]) not in rescued]
    rows = [r for r in rows if r["turns_before"]]
    for r in rows:
        r["_days"] = {pt_day(t["created_at"]) for t in r["turns_before"]}
        r["_day"] = pt_day(r["turns_before"][-1]["created_at"])
    sample = pick_sample(rows, args.n, args.max_days, args.seed)
    need_days = sorted({d for r in sample for d in r["_days"]})
    print(f"{len(rows)} untraced quotes; sample {len(sample)}; day archives {need_days}")

    import pytesseract
    from PIL import Image

    tars = {}
    if args.local_tar_dir:
        for d in need_days:
            path = os.path.join(args.local_tar_dir, f"{d}.tar")
            if os.path.exists(path):
                tars[d] = tarfile.open(path)
    else:
        from huggingface_hub import HfFileSystem, hf_hub_download
        token = common.require_token(None)
        fs = HfFileSystem(token=token)
        for d in need_days:
            path = f"images/computer-use-turns/{d}.tar"
            gb = fs.info(f"datasets/{common.REPO}/{path}")["size"] / 1e9
            if gb > args.max_gb:
                print(f"  {d}: {gb:.1f} GB over limit, skipped")
                continue
            print(f"  downloading {d} ({gb:.2f} GB)", flush=True)
            tars[d] = tarfile.open(hf_hub_download(common.REPO, path, repo_type="dataset", token=token))

    summary = collections.Counter()
    with open(os.path.join(args.out_dir, "screenshot_check.csv"), "w", newline="", encoding="utf-8") as fcsv, \
            zipfile.ZipFile(os.path.join(args.out_dir, "screenshot_samples.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        w = csv.writer(fcsv)
        w.writerow(["message_id", "agent", "created_at", "token", "turns_in_window", "redacted", "missing",
                    "checked", "result", "hit_turn_id"])
        for r in sample:
            redacted = missing = checked = 0
            result, hit_turn, last_png = "none", "", None
            for t in r["turns_before"]:
                if t.get("screenshot_is_redacted"):
                    redacted += 1
                    continue
                tar = tars.get(pt_day(t["created_at"]))
                try:
                    png = tar.extractfile(f"{t['turn_id']}.png").read()
                except Exception:  # noqa: BLE001
                    missing += 1
                    continue
                checked += 1
                last_png = (t["turn_id"], png)
                img = Image.open(io.BytesIO(png)).convert("L")
                if img.width < 1800:
                    img = img.resize((int(img.width * 1.6), int(img.height * 1.6)))
                hit = find_token(pytesseract.image_to_string(img, config="--psm 6"), r["token"])
                if hit and (result == "none" or (hit == "exact" and result == "fuzzy")):
                    result, hit_turn = hit, t["turn_id"]
                    z.writestr(f"{r['message_id']}/HIT_{t['turn_id']}.png", png)
            if last_png:
                z.writestr(f"{r['message_id']}/last_{last_png[0]}.png", last_png[1])
            if checked == 0:
                result = "uncheckable"
            summary[result] += 1
            w.writerow([r["message_id"], r["agent"], r["created_at"], r["token"], len(r["turns_before"]),
                        redacted, missing, checked, result, hit_turn])
    print(dict(summary))
    found = summary["exact"] + summary["fuzzy"]
    n = sum(summary.values()) - summary["uncheckable"]
    lo, hi = stats.wilson_ci(found, n)
    print(f"found on screen: {found}/{n}  (95% interval {lo:.1%} to {hi:.1%})")


if __name__ == "__main__":
    main()
