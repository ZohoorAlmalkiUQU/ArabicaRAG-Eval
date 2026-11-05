#!/usr/bin/env python3
"""
Check the second-judge results of the manuscript (appendix "Sensitivity of the
RAGAS Scores to the Judge Model" and Table judge_agreement) against
second_judge_per_question.csv. Standard library only: no numpy or pandas.

Usage
    python check_second_judge.py second_judge_per_question.csv
    python check_second_judge.py second_judge_per_question.csv --flags flags.csv

The optional flags file needs the columns question_id, model_key, flexible and,
if available, strict (1 or True = the matcher flags the output as an
abstention). It is used only for the Faithfulness breakdown by refusal type.

Not checkable from this CSV: the Context Recall / Context Precision rows of
Table judge_agreement (including the GPT-4.1-mini test-retest values), the
sub-sample selection itself, and anything computed on the full 1,000 questions.

Permutation p-values and bootstrap intervals are Monte Carlo estimates, so they
can differ slightly from the authors' run; Holm decisions near 0.05 may differ.
"""
import argparse
import csv
import math
import random

METRICS = {"AR": "answer_relevancy", "F": "faithfulness", "AC": "answer_correctness"}
NAMES = {"AR": "Answer Relevancy", "F": "Faithfulness", "AC": "Answer Correctness"}
MODELS = ("command", "llama", "mistral")
PAIRS = (("command", "llama"), ("command", "mistral"), ("llama", "mistral"))
SUBSETS = ("all", "ans", "unans")
B_PERM = 10_000
B_BOOT = 2_000
SEED = 42
EXACT_REFUSAL_AS = 0.9887479692489857  # Answer Similarity of the exact prescribed refusal

# Full-sample Holm decisions from Tables 6-8, per pair in PAIRS order:
# sign of (A - B) if significant, 0 if not significant.
FULL = {
    ("AR", "all"): (-1, +1, +1),
    ("AR", "ans"): (-1, +1, +1),
    ("AR", "unans"): (-1, 0, +1),
    ("AC", "all"): (+1, +1, -1),
    ("AC", "ans"): (0, +1, +1),
    ("AC", "unans"): (+1, +1, -1),
    ("F", "all"): (0, 0, 0),
    ("F", "ans"): (0, 0, 0),
    ("F", "unans"): (0, 0, 0),
}

# Values stated in the manuscript
CLAIM_RHO = {"AR": (0.884, 0.854, 0.912), "F": (0.618, 0.555, 0.678), "AC": (0.928, 0.905, 0.949)}
CLAIM_DBAR = {"AR": -0.017, "F": -0.004, "AC": -0.046}
CLAIM_NAN = {"AR": "0.0 / 3.4", "F": "0.3 / 2.7", "AC": "0.0 / 0.3"}
CLAIM_ORDER = {"AR": 2, "F": 1, "AC": 2}
CLAIM_SIG = {"AR": "6/7/6", "F": "0/4/0", "AC": "7/7/7"}


# ---------------------------------------------------------------- helpers
def num(x):
    x = (x or "").strip()
    if x == "" or x.lower() == "nan":
        return None
    return float(x)


def flag(x):
    x = (x or "").strip().lower()
    if x in ("1", "1.0", "true"):
        return 1
    if x in ("0", "0.0", "false"):
        return 0
    return None


def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else float("nan")


def sign(x):
    return (x > 0) - (x < 0)


def ranks(v):
    order = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x, y):
    return pearson(ranks(list(x)), ranks(list(y)))


def percentile(values, q):
    v = sorted(x for x in values if not math.isnan(x))
    if not v:
        return float("nan")
    pos = (len(v) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def boot_rho(groups, rng):
    """Spearman CI from resampling questions; groups = list of [(x, y), ...] per question."""
    k = len(groups)
    out = []
    for _ in range(B_BOOT):
        xs, ys = [], []
        for _ in range(k):
            for x, y in groups[rng.randrange(k)]:
                xs.append(x)
                ys.append(y)
        out.append(spearman(xs, ys))
    return percentile(out, 2.5), percentile(out, 97.5)


def perm_p(d, rng):
    """Two-sided paired sign-flip permutation test of mean(d) = 0."""
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    total = sum(d)
    obs = total / n
    thr = abs(obs) - 1e-12
    count = 0
    for _ in range(B_PERM):
        bits = format(rng.getrandbits(n), f"0{n}b")
        flipped = sum(x for x, c in zip(d, bits) if c == "1")
        if abs((total - 2 * flipped) / n) >= thr:
            count += 1
    return obs, (count + 1) / (B_PERM + 1)


def holm(pvals):
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for k, i in enumerate(sorted(range(m), key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- data
def load(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        d = {"q": r["question_id"].strip(), "m": r["model_key"].strip(),
             "ans": r["is_answerable"].strip().lower() == "true"}
        for col in list(METRICS.values()) + ["answer_similarity"]:
            d[col + "_2"] = num(r[col])
            d[col + "_1"] = num(r[col + "_original"])
        rows.append(d)
    return rows


def run_tests(by, qsets, judge, rng):
    res = []
    for mk, col in METRICS.items():
        key = f"{col}_{judge}"
        for s in SUBSETS:
            for a, b in PAIRS:
                d = []
                for q in qsets[s]:
                    xa, xb = by[(q, a)][key], by[(q, b)][key]
                    if xa is not None and xb is not None:
                        d.append(xa - xb)
                obs, p = perm_p(d, rng)
                res.append({"metric": mk, "subset": s, "pair": (a, b), "n": len(d), "delta": obs, "p": p})
    for r, pa in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = pa
        r["sig"] = pa < 0.05
    return res


def judged_part(r, judge):
    """Answer Correctness without its 0.25-weighted Answer Similarity term."""
    ac, as_ = r[f"answer_correctness_{judge}"], r[f"answer_similarity_{judge}"]
    return None if ac is None or as_ is None else (ac - 0.25 * as_) / 0.75


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--flags", help="matcher flags per question_id and model_key (optional)")
    args = ap.parse_args()

    rows = load(args.csv)
    n = len(rows)
    by = {(r["q"], r["m"]): r for r in rows}
    qans = {}
    for r in rows:
        qans.setdefault(r["q"], r["ans"])
    qsets = {"all": list(qans),
             "ans": [q for q, a in qans.items() if a],
             "unans": [q for q, a in qans.items() if not a]}

    print("== 1. Structure")
    complete = all((q, m) in by for q in qans for m in MODELS)
    print(f"rows {n}; unique (question, model) {len(by)}; questions {len(qans)}; "
          f"every question has the three models: {complete}")
    print(f"answerable {len(qsets['ans'])}, unanswerable {len(qsets['unans'])}   (claimed 150 / 150)")
    if not complete or len(by) != n:
        raise SystemExit("structure problem: stopping")

    print("\n== 2. Missing scores")
    tot1 = tot2 = 0
    for mk, col in METRICS.items():
        n1 = sum(r[col + "_1"] is None for r in rows)
        n2 = sum(r[col + "_2"] is None for r in rows)
        tot1 += n1
        tot2 += n2
        per = {m: sum(r[col + "_2"] is None for r in rows if r["m"] == m) for m in MODELS}
        print(f"{NAMES[mk]:<19} J1 {n1:>3} = {100 * n1 / n:5.2f}%   J2 {n2:>3} = {100 * n2 / n:5.2f}%"
              f"   (Table: {CLAIM_NAN[mk]})   J2 by model {per}")
    print(f"pooled over the {3 * n} re-scored items: J1 {tot1} = {100 * tot1 / (3 * n):.2f}%, "
          f"J2 {tot2} = {100 * tot2 / (3 * n):.2f}%   (text: 0.1% / 2.1%)")
    print("claimed Faithfulness J2 missing by model: command 8, llama 10, mistral 6")

    print("\n== 3. Item-level agreement (complete pairs)")
    rng = random.Random(SEED)
    for mk, col in METRICS.items():
        pairs = [(r[col + "_1"], r[col + "_2"], r["q"]) for r in rows
                 if r[col + "_1"] is not None and r[col + "_2"] is not None]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        groups = {}
        for x, y, q in pairs:
            groups.setdefault(q, []).append((x, y))
        lo, hi = boot_rho(list(groups.values()), rng)
        dbar = mean(y - x for x, y, _ in pairs)
        c = CLAIM_RHO[mk]
        print(f"{NAMES[mk]:<19} n={len(pairs)}  rho={rho:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"(claimed {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}])   dbar={dbar:+.3f}   (claimed {CLAIM_DBAR[mk]:+.3f})")
    jp = [(judged_part(r, 1), judged_part(r, 2)) for r in rows]
    jp = [p for p in jp if None not in p]
    print(f"Answer Correctness, judged part only (without the 0.25 similarity term): "
          f"rho={spearman([p[0] for p in jp], [p[1] for p in jp]):.3f}")
    diffs = [(abs(r["answer_similarity_2"] - r["answer_similarity_1"]), r["q"], r["m"]) for r in rows
             if r["answer_similarity_1"] is not None and r["answer_similarity_2"] is not None]
    dmax = max(diffs)
    print(f"Answer Similarity max |J2 - J1| = {dmax[0]:.2e} (question {dmax[1]}, {dmax[2]})   (claimed 8.8e-08)")

    print("\n== 4. Model ordering by subset mean (Order column)")
    for mk, col in METRICS.items():
        same = 0
        for s in SUBSETS:
            keep = set(qsets[s])
            sub = [r for r in rows if r["q"] in keep]
            o = {}
            for j in (1, 2):
                means = {m: mean(r[f"{col}_{j}"] for r in sub if r["m"] == m) for m in MODELS}
                o[j] = tuple(sorted(MODELS, key=lambda m: -means[m]))
            same += o[1] == o[2]
            print(f"{NAMES[mk]:<19} {s:<5} J1 {' > '.join(o[1]):<28} J2 {' > '.join(o[2])}")
        print(f"{'':<19} same order in {same}/3   (claimed {CLAIM_ORDER[mk]}/3)")

    print("\n== 5. Paired permutation tests on the sub-sample (Holm within 27 tests per judge)")
    r1 = run_tests(by, qsets, 1, random.Random(SEED))
    r2 = run_tests(by, qsets, 2, random.Random(SEED + 1))
    comb = []
    for a, b in zip(r1, r2):
        full = FULL[(a["metric"], a["subset"])][PAIRS.index(a["pair"])]
        comb.append({"metric": a["metric"], "subset": a["subset"], "pair": a["pair"], "full": full,
                     "d1": a["delta"], "p1": a["p_holm"], "s1": a["sig"], "n1": a["n"],
                     "d2": b["delta"], "p2": b["p_holm"], "s2": b["sig"], "n2": b["n"]})
    for mk in METRICS:
        x = [c for c in comb if c["metric"] == mk]
        j1 = sum(c["s1"] for c in x)
        j2 = sum(c["s2"] for c in x)
        both = sum(c["s1"] and c["s2"] and sign(c["d1"]) == sign(c["d2"]) for c in x)
        print(f"{NAMES[mk]:<19} Sig J1/J2/both = {j1}/{j2}/{both}   (claimed {CLAIM_SIG[mk]})")

    sig = [c for c in comb if c["full"] != 0]
    ns = [c for c in comb if c["full"] == 0]
    print(f"\nfull-sample significant AR/AC differences: {len(sig)}   (claimed 16)")
    print(f"  same direction under J2:               {sum(sign(c['d2']) == c['full'] for c in sig)}   (claimed 15)")
    print(f"  significant, same direction, J2:       "
          f"{sum(c['s2'] and sign(c['d2']) == c['full'] for c in sig)}   (claimed 14)")
    print(f"  significant, same direction, J1:       "
          f"{sum(c['s1'] and sign(c['d1']) == c['full'] for c in sig)}   (claimed 13)")
    print(f"  significant, opposite direction:       J1 {sum(c['s1'] and sign(c['d1']) == -c['full'] for c in sig)}, "
          f"J2 {sum(c['s2'] and sign(c['d2']) == -c['full'] for c in sig)}   (claimed 0)")
    for c in sig:
        if sign(c["d2"]) != c["full"]:
            print(f"  direction change: {c['metric']} {c['subset']} {c['pair'][0]} vs {c['pair'][1]}: "
                  f"J1 {c['d1']:+.3f} (p_Holm {c['p1']:.3f}), J2 {c['d2']:+.3f} (p_Holm {c['p2']:.3f})"
                  f"   (claimed: AC all llama vs mistral, separated by neither judge)")
    print(f"\nfull-sample non-significant: {len(ns)} (claimed 11, of which 9 Faithfulness); "
          f"significant under J2: {sum(c['s2'] for c in ns)} (claimed 4, all Faithfulness); "
          f"under J1: {sum(c['s1'] for c in ns)}")
    for c in ns:
        if c["s2"] or c["s1"]:
            print(f"  {c['metric']} {c['subset']} {c['pair'][0]} vs {c['pair'][1]}: "
                  f"J1 {c['d1']:+.3f} (p_Holm {c['p1']:.3f}), J2 {c['d2']:+.3f} (p_Holm {c['p2']:.3f})")
    print("  claimed J2 gaps, Faithfulness: all command-mistral 0.148, llama-mistral 0.122; "
          "ans 0.167, 0.153")
    print("\nFaithfulness, all nine comparisons (A - B):")
    for c in comb:
        if c["metric"] == "F":
            print(f"  {c['subset']:<5} {c['pair'][0]:>7} vs {c['pair'][1]:<7} "
                  f"J1 n={c['n1']:>3} {c['d1']:+.3f} (p_Holm {c['p1']:.3f})   "
                  f"J2 n={c['n2']:>3} {c['d2']:+.3f} (p_Holm {c['p2']:.3f})")

    print("\n== 6. Exact prescribed refusal on unanswerable questions "
          "(proxy: original Answer Similarity = 0.98875)")
    for m in MODELS:
        e = [r for r in rows if r["m"] == m and not r["ans"] and r["answer_similarity_1"] is not None
             and abs(r["answer_similarity_1"] - EXACT_REFUSAL_AS) < 1e-6]
        if not e:
            continue
        df_ = mean(r["faithfulness_2"] - r["faithfulness_1"] for r in e
                   if r["faithfulness_2"] is not None and r["faithfulness_1"] is not None)
        dac = mean(r["answer_correctness_2"] - r["answer_correctness_1"] for r in e
                   if r["answer_correctness_2"] is not None and r["answer_correctness_1"] is not None)
        zero = sum(1 for r in e if None not in (judged_part(r, 1), judged_part(r, 2))
                   and judged_part(r, 2) < 0.01 and judged_part(r, 1) > 0.99)
        print(f"{m:<8} items {len(e):>3}   mean J2 - J1: Faithfulness {df_:+.3f}, Answer Correctness {dac:+.3f}"
              f"   judged part 0 under J2 but 1 under J1: {zero}")
    print("claimed: J2 scores the exact refusal of Command-R7B-12-2024 0.205 higher on Faithfulness")

    print("\n== 7. Faithfulness offset by refusal type")
    if not args.flags:
        print("skipped: pass --flags with question_id, model_key, flexible[, strict]")
    else:
        with open(args.flags, newline="", encoding="utf-8-sig") as fh:
            fl = {(r["question_id"].strip(), r["model_key"].strip()):
                  (flag(r.get("flexible")), flag(r.get("strict"))) for r in csv.DictReader(fh)}
        missing = sum((r["q"], r["m"]) not in fl for r in rows)
        if missing:
            print(f"warning: {missing} rows have no flag")
        has_strict = any(v[1] is not None for v in fl.values())
        for m in MODELS:
            for s, want in (("ans", True), ("unans", False)):
                g = [r for r in rows if r["m"] == m and r["ans"] == want and (r["q"], r["m"]) in fl]
                if has_strict:
                    groups = {"attempted answers": lambda f: f[0] == 0,
                              "paraphrased refusals": lambda f: f[0] == 1 and f[1] == 0,
                              "exact refusals": lambda f: f[1] == 1}
                else:
                    groups = {"attempted answers": lambda f: f[0] == 0,
                              "flagged refusals": lambda f: f[0] == 1}
                for name, test in groups.items():
                    h = [r for r in g if test(fl[(r["q"], r["m"])])
                         and r["faithfulness_2"] is not None and r["faithfulness_1"] is not None]
                    if h:
                        print(f"{m:<8} {s:<5} {name:<21} n={len(h):>3}  Faithfulness J2 - J1 "
                              f"{mean(r['faithfulness_2'] - r['faithfulness_1'] for r in h):+.3f}")
        print("claimed: mistral, ans: about -0.09 to -0.10 for attempted answers and for paraphrased "
              "refusals; command, unans, exact refusals: +0.205")


if __name__ == "__main__":
    main()
