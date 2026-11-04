# -*- coding: utf-8 -*-
"""
second_judge_compare.py
=======================

Compares the second judge (second_judge_ragas.py) with the original judge
(openai/gpt-4.1-mini) on the same sub-sample, for Reviewer 1, Concern 4.
No API calls: the original scores come from the archived per-question RAGAS
file, the second-judge scores from the second-judge output directory.

The headline question is whether the second judge reproduces the paper's
conclusions (which model differences are significant, and in which
direction); matching absolute scores is not the goal.

What is reported
----------------
1. Per-item agreement for each re-judged metric (answer_relevancy,
   faithfulness, answer_correctness): Spearman rho between the two judges
   over (question, model) items, and the mean difference (second - original)
   to show any systematic offset; 95% CIs by bootstrap over questions.
2. context_recall / context_precision (judged once per question by the second
   judge): cross-judge rho against the three GPT-4.1-mini scores of each
   question. Because the three models give these metrics identical inputs,
   those three scores are replicate judgments, so their mutual rho is
   GPT-4.1-mini's own test-retest agreement and serves as a reference.
3. answer_similarity: recomputed in the second-judge run with the same
   embeddings; it is judge-independent, so only the identity check is shown.
4. Model ordering on each metric and subset under each judge.
5. For each model pair x metric x subset: direction and significance of the
   difference under each judge on the SAME sub-sample, with the paired
   sign-flip permutation test of 04_RAGAS_statistical-analysis.ipynb
   (B = 10,000, two-sided, seed 42) and Holm correction within each judge's
   family of 27 tests (3 metrics x 3 pairs x 3 subsets). The paper's
   full-sample Holm decisions are shown alongside. A sub-sample has less
   power than the full 1,000 questions, which is why the like-for-like
   comparison is second judge vs GPT-4.1-mini on the same questions.
6. NaN (parse-failure) rate of each judge.

Usage (from the repository root)
--------------------------------
python second_judge_compare.py
python second_judge_compare.py --self-test     # first reproduce the paper's 54 p-values

Outputs (in <second-dir>/comparison)
------------------------------------
judge_item_agreement.csv      per metric (and subset): rho, mean difference, NaN
judge_context_agreement.csv   context metrics: cross-judge vs test-retest rho
judge_model_means.csv         means and orderings per judge, metric, subset
judge_pairwise_tests.csv      27 tests per judge + the paper's decisions
judge_comparison_summary.json headline numbers
judge_agreement_table.tex     the appendix table with numbers filled in
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONFIG
# ============================================================
PER_QUESTION_CSV = os.path.join(
    "arabicaqa_rag_results", "ragas_full_llama_mistral_command",
    "ragas_per_question_all_models.csv",
)
PAPER_PVALUES_CSV = os.path.join(
    "arabicaqa_rag_results", "ragas_full_llama_mistral_command", "ci_analysis",
    "paired_permutation_pvalues_all_corrections.csv",
)
SECOND_DIR = os.path.join("arabicaqa_rag_results", "second_judge",
                          "qwen__qwen3-235b-a22b-2507")

MODEL_PAIRS = [("command", "llama"), ("command", "mistral"), ("llama", "mistral")]
MODEL_KEYS = ["command", "llama", "mistral"]
SHORT = {"command": "C", "llama": "L", "mistral": "M"}
SUBSETS = ["ALL", "ANSWERABLE", "UNANSWERABLE"]
REJUDGED_METRICS = ["answer_relevancy", "faithfulness", "answer_correctness"]
CONTEXT_METRICS = ["context_recall", "context_precision"]
ALL_RAGAS_METRICS = ["answer_relevancy", "faithfulness", "context_recall",
                     "context_precision", "answer_correctness", "answer_similarity"]
METRIC_LABEL = {
    "answer_relevancy": "Answer Relevancy", "faithfulness": "Faithfulness",
    "answer_correctness": "Answer Correctness", "context_recall": "Context Recall",
    "context_precision": "Context Precision", "answer_similarity": "Answer Similarity",
}
# answer_correctness = 0.75 * LLM statement F1 + 0.25 * answer_similarity (RAGAS 0.4.1)
AC_WEIGHTS = (0.75, 0.25)

N_PERMUTATIONS = 10000
RANDOM_SEED = 42
N_BOOTSTRAP = 2000
ALPHA = 0.05
ID_COL = "question_id"


# ============================================================
# STATISTICS (copied from 04_RAGAS_statistical-analysis.ipynb)
# ============================================================
def filter_subset(df, subset):
    if subset == "ALL":
        return df.copy()
    elif subset == "ANSWERABLE":
        return df[df["is_answerable"] == True].copy()
    elif subset == "UNANSWERABLE":
        return df[df["is_answerable"] == False].copy()
    raise ValueError(f"Unknown subset: {subset}")


def paired_permutation_test(df, model_a, model_b, metric, subset="ALL",
                            id_col=ID_COL, n_permutations=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Paired sign-flip permutation test, two-sided, as in the notebook."""
    df_work = filter_subset(df, subset)
    pivot = df_work[[id_col, "model", metric]].pivot_table(
        index=id_col, columns="model", values=metric, aggfunc="mean")
    out = {"model_a": model_a, "model_b": model_b, "comparison": f"{model_a} - {model_b}",
           "subset": subset, "metric": metric, "n_paired": 0, "mean_a": np.nan,
           "mean_b": np.nan, "mean_diff": np.nan, "p_value": np.nan}
    if model_a not in pivot.columns or model_b not in pivot.columns:
        return out
    paired = pivot[[model_a, model_b]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(paired)
    if n == 0:
        return out
    a = paired[model_a].to_numpy()
    b = paired[model_b].to_numpy()
    diffs = a - b
    observed_diff = float(np.mean(diffs))
    out.update(n_paired=int(n), mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
               mean_diff=observed_diff)
    if n < 2:
        return out
    rng = np.random.default_rng(seed)
    permuted_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted_diffs[i] = np.mean(diffs * signs)
    out["p_value"] = float((np.sum(np.abs(permuted_diffs) >= abs(observed_diff)) + 1)
                           / (n_permutations + 1))
    return out


def holm_correction(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid_mask = ~np.isnan(pvalues)
    valid_p = pvalues[valid_mask]
    m = len(valid_p)
    if m == 0:
        return adjusted
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    adjusted_sorted = np.array([(m - i) * sorted_p[i] for i in range(m)])
    adjusted_sorted = np.minimum(np.maximum.accumulate(adjusted_sorted), 1.0)
    adjusted_valid = np.empty(m)
    adjusted_valid[order] = adjusted_sorted
    adjusted[valid_mask] = adjusted_valid
    return adjusted


def test_family(df_long, metrics):
    rows = [paired_permutation_test(df_long, a, b, m, s)
            for s in SUBSETS for m in metrics for a, b in MODEL_PAIRS]
    t = pd.DataFrame(rows)
    t["p_value_holm"] = holm_correction(t["p_value"].to_numpy())
    t["significant_holm_0.05"] = t["p_value_holm"] < ALPHA
    return t


def decision(sig, diff):
    if not sig or pd.isna(diff):
        return "ns"
    return "+" if diff > 0 else "-"


# ============================================================
# AGREEMENT HELPERS
# ============================================================
def spearman(x, y):
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def cluster_bootstrap(groups, fn, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """95% percentile CI of fn over questions resampled with replacement.
    groups: list of (x_array, y_array) per question."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, size=k)
        x = np.concatenate([groups[i][0] for i in idx])
        y = np.concatenate([groups[i][1] for i in idx])
        v = fn(x, y)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def item_agreement(j1, j2, metric, subset):
    a = filter_subset(j1, subset)[[ID_COL, "model", metric]].rename(columns={metric: "j1"})
    b = filter_subset(j2, subset)[[ID_COL, "model", metric]].rename(columns={metric: "j2"})
    m = a.merge(b, on=[ID_COL, "model"])
    n_items = len(m)
    nan1, nan2 = int(m["j1"].isna().sum()), int(m["j2"].isna().sum())
    m = m.dropna()
    groups = [(g["j1"].to_numpy(), g["j2"].to_numpy()) for _, g in m.groupby(ID_COL)]
    x, y = m["j1"].to_numpy(), m["j2"].to_numpy()
    rho = spearman(x, y)
    rho_lo, rho_hi = cluster_bootstrap(groups, spearman)
    md = float(np.mean(y - x)) if len(x) else np.nan
    md_lo, md_hi = cluster_bootstrap(groups, lambda u, v: float(np.mean(v - u)))
    return {"metric": metric, "subset": subset, "n_items": n_items, "n_complete": int(len(m)),
            "spearman": rho, "spearman_ci_lower": rho_lo, "spearman_ci_upper": rho_hi,
            "pearson": float(stats.pearsonr(x, y).statistic) if len(x) > 2 else np.nan,
            "mean_original": float(np.mean(x)) if len(x) else np.nan,
            "mean_second": float(np.mean(y)) if len(y) else np.nan,
            "mean_diff_second_minus_original": md, "mean_diff_ci_lower": md_lo,
            "mean_diff_ci_upper": md_hi,
            "mae": float(np.mean(np.abs(y - x))) if len(x) else np.nan,
            "nan_original": nan1, "nan_second": nan2,
            "nan_rate_original": nan1 / n_items if n_items else np.nan,
            "nan_rate_second": nan2 / n_items if n_items else np.nan}


def ordering(means):
    """'L > C > M' from {model: mean}; equal means are joined with '='."""
    items = sorted(means.items(), key=lambda kv: -kv[1])
    out = SHORT[items[0][0]]
    for (_, prev), (k, v) in zip(items, items[1:]):
        out += (" = " if v == prev else " > ") + SHORT[k]
    return out


# ============================================================
# LOADING
# ============================================================
def load_original(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["is_answerable"] = df["is_answerable"].astype(bool)
    df["model"] = df["model_key"]
    return df


def load_second(second_dir, original):
    path = os.path.join(second_dir, "second_judge_per_question.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}: run second_judge_ragas.py first.")
    s = pd.read_csv(path, encoding="utf-8-sig")
    # Keep questions scored for all three models (paired design).
    counts = s.groupby(ID_COL)["model_key"].nunique()
    keep = counts[counts == len(MODEL_KEYS)].index
    s = s[s[ID_COL].isin(keep)].copy()
    s["model"] = s["model_key"]
    s = s.drop(columns=["is_answerable"], errors="ignore").merge(
        original.drop_duplicates(ID_COL)[[ID_COL, "is_answerable"]], on=ID_COL, how="left")
    ctx_path = os.path.join(second_dir, "second_judge_context_once.csv")
    ctx = pd.read_csv(ctx_path, encoding="utf-8-sig") if os.path.exists(ctx_path) else pd.DataFrame()
    if len(ctx):
        ctx = ctx[ctx[ID_COL].isin(keep)]
    summary_path = os.path.join(second_dir, "run_summary.json")
    run_summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}
    return s, ctx, run_summary


# ============================================================
# SELF-TEST: reproduce the paper's 54 p-values
# ============================================================
def self_test(original, paper):
    t = test_family(original, ALL_RAGAS_METRICS)
    k = ["subset", "metric", "model_a", "model_b"]
    m = t.merge(paper[k + ["p_value", "significant_holm_0.05"]], on=k, suffixes=("", "_paper"))
    p_diff = float(np.max(np.abs(m["p_value"] - m["p_value_paper"])))
    same = int((m["significant_holm_0.05"] == m["significant_holm_0.05_paper"]).sum())
    print(f"SELF-TEST: {len(m)} tests; max |p - p_paper| = {p_diff:.2e}; "
          f"Holm decisions identical: {same}/{len(m)}; "
          f"significant: {int(m['significant_holm_0.05'].sum())} (paper "
          f"{int(m['significant_holm_0.05_paper'].sum())})")
    return p_diff, same, len(m)


# ============================================================
# MAIN ANALYSIS
# ============================================================
def analyse(args):
    original = load_original(args.per_question_csv)
    paper = pd.read_csv(args.paper_pvalues, encoding="utf-8-sig")
    out_dir = os.path.join(args.second_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    self_test_result = self_test(original, paper) if args.self_test else None

    j2, ctx2, run_summary = load_second(args.second_dir, original)
    qids = sorted(j2[ID_COL].unique())
    j1 = original[original[ID_COL].isin(qids)].copy()
    n_q = len(qids)
    n_ans = int(j1.drop_duplicates(ID_COL)["is_answerable"].sum())
    print(f"Sub-sample: {n_q} questions ({n_ans} answerable, {n_q - n_ans} unanswerable), "
          f"{len(j2)} items")

    # Judge-dependent part of answer_correctness, for both judges.
    for d in (j1, j2):
        d["answer_correctness_llm_part"] = (
            (d["answer_correctness"] - AC_WEIGHTS[1] * d["answer_similarity"]) / AC_WEIGHTS[0])

    # 1) Item agreement
    agree = pd.DataFrame([item_agreement(j1, j2, m, s)
                          for m in REJUDGED_METRICS + ["answer_correctness_llm_part"]
                          for s in SUBSETS])
    agree.to_csv(os.path.join(out_dir, "judge_item_agreement.csv"), index=False,
                 encoding="utf-8-sig")

    # 2) Context metrics: cross-judge vs GPT-4.1-mini test-retest
    ctx_rows = []
    if len(ctx2):
        for m in CONTEXT_METRICS:
            w = j1.pivot(index=ID_COL, columns="model_key", values=m).reindex(ctx2[ID_COL])
            y = ctx2.set_index(ID_COL)[m]
            retest = [spearman(*w[[a, b]].dropna().to_numpy().T)
                      for a, b in itertools.combinations(MODEL_KEYS, 2)]
            cross = []
            for mk in MODEL_KEYS:
                pair = pd.concat([w[mk], y], axis=1).dropna().to_numpy()
                cross.append(spearman(pair[:, 0], pair[:, 1]))
            gpt_mean = w[MODEL_KEYS].mean(axis=1)
            both = pd.concat([gpt_mean, y], axis=1).dropna()
            ctx_rows.append({
                "metric": m, "n_questions": int(len(y)),
                "spearman_cross_judge_mean": float(np.nanmean(cross)),
                "spearman_original_retest_mean": float(np.nanmean(retest)),
                "mean_diff_second_minus_original": float((both.iloc[:, 1] - both.iloc[:, 0]).mean()),
                "nan_original": int(w[MODEL_KEYS].isna().sum().sum()),
                "nan_original_items": int(w[MODEL_KEYS].size),
                "nan_second": int(y.isna().sum()),
            })
    ctx_agree = pd.DataFrame(ctx_rows)
    ctx_agree.to_csv(os.path.join(out_dir, "judge_context_agreement.csv"), index=False,
                     encoding="utf-8-sig")

    # 3) Answer similarity identity check
    as_diff = float((j2["answer_similarity"] - j2["answer_similarity_original"]).abs().max())

    # 4) Means and orderings
    mean_rows = []
    orders = {}
    for m in REJUDGED_METRICS:
        for s in SUBSETS:
            row = {"metric": m, "subset": s}
            for label, d in [("original_sub", j1), ("second_sub", j2), ("original_full", original)]:
                means = filter_subset(d, s).groupby("model_key")[m].mean().to_dict()
                for mk in MODEL_KEYS:
                    row[f"{label}_{mk}"] = means.get(mk, np.nan)
                row[f"{label}_ordering"] = ordering(means)
            orders[(m, s)] = row
            mean_rows.append(row)
    means_df = pd.DataFrame(mean_rows)
    means_df["ordering_same_sub"] = means_df["original_sub_ordering"] == means_df["second_sub_ordering"]
    means_df["ordering_same_full"] = means_df["original_full_ordering"] == means_df["second_sub_ordering"]
    means_df.to_csv(os.path.join(out_dir, "judge_model_means.csv"), index=False,
                    encoding="utf-8-sig")

    # 5) Pairwise tests: 27 per judge on the same sub-sample
    t1 = test_family(j1, REJUDGED_METRICS)
    t2 = test_family(j2, REJUDGED_METRICS)
    k = ["subset", "metric", "model_a", "model_b"]
    cols = ["n_paired", "mean_a", "mean_b", "mean_diff", "p_value", "p_value_holm",
            "significant_holm_0.05"]
    tests = t1[k + cols].merge(t2[k + cols], on=k, suffixes=("_original_sub", "_second_sub"))
    tests = tests.merge(
        paper[k + ["mean_diff", "p_value_holm", "significant_holm_0.05"]].rename(columns={
            "mean_diff": "mean_diff_paper", "p_value_holm": "p_value_holm_paper",
            "significant_holm_0.05": "significant_holm_0.05_paper"}), on=k, how="left")
    for lab in ["original_sub", "second_sub", "paper"]:
        tests[f"decision_{lab}"] = [decision(s, d) for s, d in
                                    zip(tests[f"significant_holm_0.05_{lab}"], tests[f"mean_diff_{lab}"])]
    tests["same_direction_second_vs_paper"] = (
        np.sign(tests["mean_diff_second_sub"]) == np.sign(tests["mean_diff_paper"]))
    tests["same_decision_second_vs_original_sub"] = (
        tests["decision_second_sub"] == tests["decision_original_sub"])
    tests.to_csv(os.path.join(out_dir, "judge_pairwise_tests.csv"), index=False,
                 encoding="utf-8-sig")

    # Headline counts
    def counts(g):
        s1 = g["significant_holm_0.05_original_sub"]
        s2 = g["significant_holm_0.05_second_sub"]
        same_dir = np.sign(g["mean_diff_original_sub"]) == np.sign(g["mean_diff_second_sub"])
        return {
            "tests": int(len(g)),
            "sig_original_sub": int(s1.sum()), "sig_second_sub": int(s2.sum()),
            "sig_both_same_direction": int((s1 & s2 & same_dir).sum()),
            "sig_both_opposite_direction": int((s1 & s2 & ~same_dir).sum()),
            "sig_second_only": int((s2 & ~s1).sum()), "sig_original_only": int((s1 & ~s2).sum()),
            "same_decision": int(g["same_decision_second_vs_original_sub"].sum()),
            "same_direction": int(same_dir.sum()),
        }

    per_metric = {m: counts(tests[tests["metric"] == m]) for m in REJUDGED_METRICS}
    overall = counts(tests)
    paper_sig = tests[tests["significant_holm_0.05_paper"] == True]
    vs_paper = {
        "paper_significant": int(len(paper_sig)),
        "same_direction_second_sub": int(paper_sig["same_direction_second_vs_paper"].sum()),
        "holm_sig_same_direction_second_sub": int(
            (paper_sig["significant_holm_0.05_second_sub"]
             & paper_sig["same_direction_second_vs_paper"]).sum()),
        "holm_sig_same_direction_original_sub": int(
            (paper_sig["significant_holm_0.05_original_sub"]
             & (np.sign(paper_sig["mean_diff_original_sub"]) == np.sign(paper_sig["mean_diff_paper"]))).sum()),
        "holm_sig_opposite_direction_second_sub": int(
            (paper_sig["significant_holm_0.05_second_sub"]
             & ~paper_sig["same_direction_second_vs_paper"]).sum()),
        "paper_nonsignificant": int((tests["significant_holm_0.05_paper"] == False).sum()),
        "paper_ns_but_second_sig": int(((tests["significant_holm_0.05_paper"] == False)
                                        & tests["significant_holm_0.05_second_sub"]).sum()),
        "faithfulness_sig_second_sub": int(tests.loc[tests["metric"] == "faithfulness",
                                                     "significant_holm_0.05_second_sub"].sum()),
    }
    ord_same = {m: int(means_df.loc[means_df["metric"] == m, "ordering_same_sub"].sum())
                for m in REJUDGED_METRICS}

    a_all = agree[agree["subset"] == "ALL"].set_index("metric")
    summary = {
        "judge_second": run_summary.get("judge"), "judge_original": "openai/gpt-4.1-mini",
        "providers_seen": run_summary.get("providers_seen"),
        "n_questions": n_q, "n_answerable": n_ans, "n_items": int(len(j2)),
        "cost_cumulative_usd": run_summary.get("cost_cumulative_usd"),
        "self_test": None if self_test_result is None else {
            "max_abs_p_diff": self_test_result[0], "holm_decisions_identical": self_test_result[1],
            "tests": self_test_result[2]},
        "item_agreement_all": {m: {
            "spearman": a_all.loc[m, "spearman"],
            "spearman_ci": [a_all.loc[m, "spearman_ci_lower"], a_all.loc[m, "spearman_ci_upper"]],
            "mean_diff": a_all.loc[m, "mean_diff_second_minus_original"],
            "mean_diff_ci": [a_all.loc[m, "mean_diff_ci_lower"], a_all.loc[m, "mean_diff_ci_upper"]],
            "nan_rate_original": a_all.loc[m, "nan_rate_original"],
            "nan_rate_second": a_all.loc[m, "nan_rate_second"],
        } for m in REJUDGED_METRICS + ["answer_correctness_llm_part"]},
        "context_agreement": ctx_agree.to_dict(orient="records"),
        "answer_similarity_max_abs_diff": as_diff,
        "ordering_same_as_original_sub_of_3_subsets": ord_same,
        "pairwise_per_metric": per_metric, "pairwise_overall": overall,
        "vs_paper": vs_paper,
        "holm_family": "27 tests per judge (3 metrics x 3 pairs x 3 subsets)",
    }
    with open(os.path.join(out_dir, "judge_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    tex = latex_table(summary, agree, ctx_agree, per_metric, ord_same)
    with open(os.path.join(out_dir, "judge_agreement_table.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    print_summary(summary, tests, means_df)
    print(f"\nSaved outputs in {out_dir}")


# ============================================================
# REPORTING
# ============================================================
def f3(v):
    return "--" if v is None or pd.isna(v) else f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")


def f2(v):
    return "--" if v is None or pd.isna(v) else f"{v:.2f}".replace("-", "$-$")


def pct(v):
    return "--" if v is None or pd.isna(v) else f"{100 * v:.1f}"


def latex_table(s, agree, ctx_agree, per_metric, ord_same):
    a = agree[agree["subset"] == "ALL"].set_index("metric")
    lines = []
    for m in REJUDGED_METRICS:
        r = a.loc[m]
        pm = per_metric[m]
        lines.append(
            f"\\rowcolor{{yellow}} {METRIC_LABEL[m]} & {f2(r.spearman)} "
            f"[{f2(r.spearman_ci_lower)}, {f2(r.spearman_ci_upper)}] & "
            f"{f3(r.mean_diff_second_minus_original)} & "
            f"{pct(r.nan_rate_original)} / {pct(r.nan_rate_second)} & "
            f"{ord_same[m]}/3 & {pm['sig_original_sub']} / {pm['sig_second_sub']} / "
            f"{pm['sig_both_same_direction']} \\\\")
    c = ctx_agree.set_index("metric") if len(ctx_agree) else None
    for m in CONTEXT_METRICS:
        if c is not None and m in c.index:
            r = c.loc[m]
            nan1 = r.nan_original / r.nan_original_items if r.nan_original_items else np.nan
            nan2 = r.nan_second / r.n_questions if r.n_questions else np.nan
            lines.append(
                f"\\rowcolor{{yellow}} {METRIC_LABEL[m]}$^{{a}}$ & {f2(r.spearman_cross_judge_mean)} "
                f"({f2(r.spearman_original_retest_mean)}) & {f3(r.mean_diff_second_minus_original)} & "
                f"{pct(nan1)} / {pct(nan2)} & -- & -- \\\\")
    n = s["n_questions"]
    body = "\n".join(lines)
    return f"""% Generated by second_judge_compare.py; paste into the appendix table.
% Sub-sample: {n} questions ({s['n_answerable']} answerable), {s['n_items']} items.
% Answer Similarity max |second - archived| = {s['answer_similarity_max_abs_diff']:.2e}
% Paper-significant AR/AC differences: same direction {s['vs_paper']['same_direction_second_sub']}/{s['vs_paper']['paper_significant']}; Holm-significant same direction: second judge {s['vs_paper']['holm_sig_same_direction_second_sub']}, GPT-4.1-mini on the same sub-sample {s['vs_paper']['holm_sig_same_direction_original_sub']}; significant in the opposite direction: {s['vs_paper']['holm_sig_opposite_direction_second_sub']}; new significant among paper-non-significant: {s['vs_paper']['paper_ns_but_second_sig']}.
{body}
"""


def print_summary(s, tests, means_df):
    print("\n" + "=" * 28 + " SECOND-JUDGE COMPARISON SUMMARY (paste this back) " + "=" * 28)
    print(f"judges: second={s['judge_second']}  original={s['judge_original']}  "
          f"providers={s['providers_seen']}  cost so far=${(s['cost_cumulative_usd'] or 0):.4f}")
    print(f"sub-sample: {s['n_questions']} questions ({s['n_answerable']} answerable), "
          f"{s['n_items']} question x model items")
    if s["self_test"]:
        st = s["self_test"]
        print(f"self-test (paper's 54 tests): max|dp|={st['max_abs_p_diff']:.2e}, "
              f"Holm decisions identical {st['holm_decisions_identical']}/{st['tests']}")
    print("\nPer-item agreement (ALL): Spearman [95% CI]; mean diff second-original [95% CI]; NaN% orig/second")
    for m, v in s["item_agreement_all"].items():
        print(f"  {m:28s} rho={v['spearman']:.3f} [{v['spearman_ci'][0]:.3f}, {v['spearman_ci'][1]:.3f}]"
              f"  d={v['mean_diff']:+.3f} [{v['mean_diff_ci'][0]:+.3f}, {v['mean_diff_ci'][1]:+.3f}]"
              f"  NaN {100 * v['nan_rate_original']:.1f}/{100 * v['nan_rate_second']:.1f}")
    for r in s["context_agreement"]:
        print(f"  {r['metric']:28s} cross-judge rho={r['spearman_cross_judge_mean']:.3f}  "
              f"GPT test-retest rho={r['spearman_original_retest_mean']:.3f}  "
              f"d={r['mean_diff_second_minus_original']:+.3f}  NaN second={r['nan_second']}/{r['n_questions']}")
    print(f"  answer_similarity max |second - archived| = {s['answer_similarity_max_abs_diff']:.2e}")
    print("\nModel ordering (original sub | second sub | paper full):")
    for _, r in means_df.iterrows():
        flag = "same" if r["ordering_same_sub"] else "DIFFERENT"
        print(f"  {r['metric']:20s} {r['subset']:12s} {r['original_sub_ordering']} | "
              f"{r['second_sub_ordering']} | {r['original_full_ordering']}   [{flag}]")
    print("\nPairwise tests (Holm within 27 per judge). Decision: + / - significant, ns:")
    print("  metric               subset        pair               orig_sub(d, pH)       second_sub(d, pH)     paper")
    for _, r in tests.iterrows():
        print(f"  {r['metric']:20s} {r['subset']:12s} {r['model_a']:>7s}-{r['model_b']:<8s}  "
              f"{r['decision_original_sub']:>2s} ({r['mean_diff_original_sub']:+.3f}, {r['p_value_holm_original_sub']:.3f})"
              f"   {r['decision_second_sub']:>2s} ({r['mean_diff_second_sub']:+.3f}, {r['p_value_holm_second_sub']:.3f})"
              f"   {r['decision_paper']:>2s}")
    vp = s["vs_paper"]
    print(f"\nOf the {vp['paper_significant']} paper-significant differences (AR, AC):")
    print(f"  same direction under second judge on sub-sample: {vp['same_direction_second_sub']}")
    print(f"  Holm-significant, same direction: second judge {vp['holm_sig_same_direction_second_sub']} "
          f"| GPT-4.1-mini on the same sub-sample {vp['holm_sig_same_direction_original_sub']}")
    print(f"  Holm-significant in the OPPOSITE direction under second judge: "
          f"{vp['holm_sig_opposite_direction_second_sub']}")
    print(f"Of the {vp['paper_nonsignificant']} paper-non-significant: significant under second judge: "
          f"{vp['paper_ns_but_second_sig']} (Faithfulness: {vp['faithfulness_sig_second_sub']})")
    o = s["pairwise_overall"]
    print(f"Second vs original on the same sub-sample (27 tests): same decision {o['same_decision']}, "
          f"same direction {o['same_direction']}, sig both same dir {o['sig_both_same_direction']}, "
          f"second only {o['sig_second_only']}, original only {o['sig_original_only']}, "
          f"opposite {o['sig_both_opposite_direction']}")
    print("=" * 108)


def main():
    ap = argparse.ArgumentParser(description="Compare second judge with GPT-4.1-mini.")
    ap.add_argument("--per-question-csv", default=PER_QUESTION_CSV)
    ap.add_argument("--paper-pvalues", default=PAPER_PVALUES_CSV)
    ap.add_argument("--second-dir", default=SECOND_DIR)
    ap.add_argument("--self-test", action="store_true",
                    help="first reproduce the paper's 54 p-values from the archive")
    analyse(ap.parse_args())


if __name__ == "__main__":
    main()
