"""Who introduced a term, how fast it spread, and which model families used it (chat only).

Seed terms are listed in SEEDS. New terms are found automatically: words first seen on or after
--novelty-start that reach several agents in several families. A term used by a human or found in
a goal text before the first agent used it is flagged, since the agents did not introduce it.

Out: term_diffusion_terms.csv, term_adoption_order.csv, term_weekly_by_family.csv, term_adoption.png
"""
import argparse
import collections
import csv
import os
import re

import common

SEEDS = {
    "cold_verification": r"cold[- ]?(?:verif\w*|clone\w*|run\b|ran\b|check\w*|replay\w*)",
    "first_latch": r"first[- ]latch\w*",
    "prelog": r"\bpre[- ]?log(?:s|ged|ging)?\b",
    "receipt": r"\breceipts?\b",
}
TOKEN_RE = re.compile(r"(?<![\w/.@#:$%-])([a-z][a-z0-9]*(?:[-_][a-z0-9]+)*)(?![\w/@:]|\.\w)")
URL_RE = re.compile(r"https?://\S+")


def tokens(text):
    """Distinct lowercase word-like tokens, 4 to 32 characters, not hex strings, at most one digit."""
    out = set()
    for m in TOKEN_RE.finditer(URL_RE.sub(" ", text.lower())):
        w = m.group(1)
        if 4 <= len(w) <= 32 and not re.fullmatch(r"[0-9a-f]+", w) and sum(c.isdigit() for c in w) <= 1:
            out.add(w)
    return out


def load_goal_text(local_dir, token):
    parts = []
    for name, keys in (("village_goals.jsonl.gz", ("goal",)), ("agent_goals.jsonl.gz", ("name", "description"))):
        try:
            for r in common.read_rows(name, local_dir, token):
                parts.extend(str(r.get(k) or "") for k in keys)
        except Exception as e:  # noqa: BLE001
            print(f"note: could not read {name}: {e}")
    return "\n".join(parts).lower()


def find_candidates(msgs, names, bit, fam_of, novelty, args):
    """Terms new since `novelty` used by enough agents and families. Returns {term: (count, first, agents, families)}."""
    seed_rx = [re.compile(v, re.I) for v in SEEDS.values()]
    name_tokens = {w for n in names.values() for w in re.findall(r"[a-z]+", (n or "").lower())}
    stat = {}
    for ts, _, actor, content in msgs:
        b = bit.get(actor, 0)
        for w in tokens(content):
            s = stat.get(w)
            if s is None:
                stat[w] = [1, ts, b]
            else:
                s[0] += 1
                s[2] |= b
    agents_by_bit = {bit[a]: a for a in bit}
    out = {}
    for w, (count, first, mask) in stat.items():
        if first < novelty or count < args.min_msgs or w in name_tokens or any(rx.fullmatch(w) for rx in seed_rx):
            continue
        members = [a for b, a in agents_by_bit.items() if mask & b]
        fams = {fam_of[a] for a in members}
        if len(members) >= args.min_agents and len(fams) >= args.min_families:
            out[w] = (count, first, len(members), len(fams))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--novelty-start", default="2026-07-01")
    ap.add_argument("--min-agents", type=int, default=5)
    ap.add_argument("--min-families", type=int, default=3)
    ap.add_argument("--min-msgs", type=int, default=20)
    ap.add_argument("--top-terms", type=int, default=150)
    args = ap.parse_args()
    token = common.require_token(args.local_dir)
    names, msgs = common.load_chat(args.local_dir, token)
    goal_text = load_goal_text(args.local_dir, token)

    agent_ids = sorted({m[2] for m in msgs if m[2] != "HUMAN"})
    bit = {a: 1 << i for i, a in enumerate(agent_ids)}
    fam_of = {a: common.family(names.get(a, a)) for a in agent_ids}
    cand = find_candidates(msgs, names, bit, fam_of, common.parse_time(args.novelty_start), args)
    ranked = sorted(cand, key=lambda w: (-cand[w][3], -cand[w][2], -cand[w][0]))[:args.top_terms]
    print(f"{len(cand)} new terms reach {args.min_agents}+ agents in {args.min_families}+ families; keeping {len(ranked)}")

    kind = {k: "seed" for k in SEEDS}
    kind.update({k: "discovered" for k in ranked})
    seed_rx = {k: re.compile(v, re.I) for k, v in SEEDS.items()}
    keep = set(ranked)
    events = collections.defaultdict(list)
    for ts, _, actor, content in msgs:
        here = [k for k, rx in seed_rx.items() if rx.search(content)]
        here += list(tokens(content) & keep)
        for k in here:
            events[k].append((ts, actor))

    terms, order, weekly = [], [], collections.defaultdict(lambda: [0, set()])
    for k in list(SEEDS) + ranked:
        ev = events.get(k)
        if not ev:
            continue
        first, last = {}, {}
        for ts, a in ev:
            first.setdefault(a, ts)
            last[a] = ts
            fam = "Human" if a == "HUMAN" else fam_of[a]
            cell = weekly[(k, common.week_of(ts), fam)]
            cell[0] += 1
            cell[1].add(a)
        adopters = sorted(((a, t) for a, t in first.items() if a != "HUMAN"), key=lambda x: x[1])
        if not adopters:
            continue
        origin, t0 = adopters[0]
        days = [(t - t0).total_seconds() / 86400 for _, t in adopters]
        nth = lambda n: round(days[n - 1], 1) if len(days) >= n else ""  # noqa: E731
        in_goals = bool(re.search(r"\b" + re.escape(k.replace("_", " ")) + r"\b", goal_text)) if kind[k] == "discovered" else ""
        terms.append({
            "term": k, "kind": kind[k], "origin_agent": names.get(origin, origin), "origin_family": fam_of[origin],
            "first_use": t0.isoformat(), "messages": len(ev), "agents": len(adopters),
            "families": len({fam_of[a] for a, _ in adopters}),
            "days_to_3rd_adopter": nth(3), "days_to_5th_adopter": nth(5),
            "adopters_within_7d": sum(d <= 7 for d in days), "adopters_within_30d": sum(d <= 30 for d in days),
            "uses_by_others_after_origin_last_use": sum(1 for ts, a in ev if a not in (origin, "HUMAN") and ts > last[origin]),
            "human_used_first": "HUMAN" in first and first["HUMAN"] <= t0, "in_goal_text": in_goals,
        })
        order += [[k, i + 1, names.get(a, a), fam_of[a], t.isoformat(), round(d, 2)]
                  for i, ((a, t), d) in enumerate(zip(adopters, days))]

    def write(name, header, rows):
        with open(os.path.join(args.out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    if terms:
        header = list(terms[0])
        write("term_diffusion_terms.csv", header, [[t[h] for h in header] for t in terms])
    write("term_adoption_order.csv", ["term", "rank", "agent", "family", "first_use", "days_after_origin"], order)
    write("term_weekly_by_family.csv", ["term", "week", "family", "messages", "distinct_agents"],
          [[k, wk, fam, v[0], len(v[1])] for (k, wk, fam), v in sorted(weekly.items())])
    for t in terms:
        if t["kind"] == "seed":
            print(f"{t['term']:20}{t['origin_agent']:22}{t['first_use'][:10]}  agents {t['agents']:>3}  families {t['families']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        seeds = [k for k in SEEDS if events.get(k)]
        fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 3.6), squeeze=False)
        for ax, k in zip(axes[0], seeds):
            per_fam = collections.defaultdict(dict)
            for ts, a in events[k]:
                if a != "HUMAN":
                    per_fam[fam_of[a]].setdefault(a, ts)
            for fam, d in sorted(per_fam.items()):
                times = sorted(d.values())
                ax.step(times, range(1, len(times) + 1), where="post", label=fam)
            ax.set_title(k.replace("_", " "), fontsize=10)
            ax.set_ylabel("agents using it (cumulative)")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "term_adoption.png"), dpi=140)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
