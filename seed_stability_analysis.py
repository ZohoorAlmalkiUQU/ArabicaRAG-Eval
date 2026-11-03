# -*- coding: utf-8 -*-
"""
seed_stability_analysis.py
==========================

Sampling-seed sensitivity analysis for Reviewer 1, Comment 1:

    "The evaluation is limited to 1,000 questions selected using a single
     random seed. Testing with additional samples or different random seeds
     would improve the reliability of the findings."

No new generation or RAGAS judging is performed. The analysis re-uses the
archived per-question outputs and scores of the 1,000-question sample.

Design: split-half replication
------------------------------
For each of S seeds (default 100, seeds 1..S), the 1,000 questions are
randomly partitioned into two DISJOINT halves of 500, stratified by
answerability (250 answerable + 250 unanswerable per half). Each half is a
random 500-question sample that shares no question with the other half.
Every reported metric and every pairwise model comparison is recomputed on
each half, and we record

  * whether both halves reproduce the DIRECTION of the full-sample
    difference (sign agreement);
  * whether the difference is significant (paired sign-flip permutation
    test, uncorrected p < 0.05) in both halves;
  * whether both halves reproduce the full-sample ORDERING of the three
    models on each metric;
  * the spread of each model's score across the 2*S half-sample estimates.

Full-sample significance is computed with the same paired permutation test
as the manuscript (10,000 sign-flips, two-sided, Holm correction within the
task-metric family and within the RAGAS family). When the manuscript's own
p-value file is supplied, the Holm decisions are cross-checked against it.

Metric definitions are copied from 03_EM_EM25_ans-abstention-rate.ipynb
(EM, EM-25, strict/flexible abstention) and the abstention matchers are
imported from the canonical abstention_matcher.py. A reproduction check
confirms that the full-sample values equal the published ones before any
split-half result is written.

Usage (from the repository root)
--------------------------------
python seed_stability_analysis.py \
    --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
    --ragas arabicaqa_rag_results/ragas_full_llama_mistral_command/ragas_per_question_all_models.csv \
    --reference arabicaqa_rag_results/predictions/final_scores_strict_and_flexible_abstention_corrected.csv \
    --ragas-reference arabicaqa_rag_results/ragas_full_llama_mistral_command/ragas_summary_means.csv \
    --paper-pvalues arabicaqa_rag_results/ragas_full_llama_mistral_command/ci_analysis/paired_permutation_pvalues_all_corrections.csv \
    --out-dir arabicaqa_rag_results/seed_stability

Outputs (in --out-dir)
----------------------
seed_stability_pairwise.csv   one row per (subset, metric, model pair)
seed_stability_ranking.csv    one row per (subset, metric)
seed_stability_per_model.csv  one row per (subset, metric, model)
seed_stability_summary.json   headline numbers for the manuscript
"""

import argparse
import ast
import itertools
import json
import math
import os
import string
import sys

import numpy as np
import pandas as pd

from abstention_matcher import (
    NO_ANSWER,
    normalize_arabic_text,
    is_flexible_no_answer,
    is_strict_no_answer,
)

RAGAS_METRICS = ["answer_relevancy", "faithfulness", "context_recall",
                 "context_precision", "answer_correctness", "answer_similarity"]
SUBSETS = ["ALL", "ANSWERABLE", "UNANSWERABLE"]
ALPHA = 0.05


# ----------------------------------------------------------------------
# Metric definitions (ported verbatim from notebook 03)
# ----------------------------------------------------------------------

def normalize_pred(pred):
    if pred is None:
        return ""
    if isinstance(pred, float) and math.isnan(pred):
        return ""
    return str(pred).strip()


def parse_maybe_list(x):
    if x is None:
        return x
    if isinstance(x, float) and math.isnan(x):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                pass
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
        return x
    return x


def force_list(x):
    if x is None:
        return [NO_ANSWER]
    if isinstance(x, float) and math.isnan(x):
        return [NO_ANSWER]
    if isinstance(x, (list, tuple)):
        cleaned = []
        for item in x:
            if item is None:
                continue
            if isinstance(item, float) and math.isnan(item):
                continue
            item = str(item).strip()
            if item:
                cleaned.append(item)
        return cleaned if cleaned else [NO_ANSWER]
    s = str(x).strip()
    return [s] if s else [NO_ANSWER]


def normalize_refs(refs):
    if refs is None:
        return [NO_ANSWER]
    if isinstance(refs, float) and math.isnan(refs):
        return [NO_ANSWER]
    if isinstance(refs, str):
        s = refs.strip()
        return [s] if s else [NO_ANSWER]
    if isinstance(refs, (list, tuple)):
        if len(refs) == 1 and isinstance(refs[0], (list, tuple)):
            refs = refs[0]
        cleaned = []
        for r in refs:
            if r is None:
                continue
            if isinstance(r, float) and math.isnan(r):
                continue
            r = str(r).strip()
            if r:
                cleaned.append(r)
        return cleaned if cleaned else [NO_ANSWER]
    s = str(refs).strip()
    return [s] if s else [NO_ANSWER]


# HuggingFace evaluate's exact_match with ignore_case=True and
# ignore_punctuation=True: lower-case, delete string.punctuation, compare.
# Re-implemented here so the script needs no network access to load it.
_PUNCT = str.maketrans("", "", string.punctuation)


def hf_exact_match(pred_norm, ref_norm):
    return float(pred_norm.lower().translate(_PUNCT) ==
                 ref_norm.lower().translate(_PUNCT))


def compute_em25_single(pred_norm, ref_norm, max_strip=25):
    if not ref_norm:
        return 0
    for k in range(0, max_strip + 1):
        if pred_norm[k:] == ref_norm:
            return 1
    return 0


def em_and_em25(pred, refs):
    pred_norm = normalize_arabic_text(normalize_pred(pred))
    refs_norm = [normalize_arabic_text(r) for r in normalize_refs(refs)]
    em = max(hf_exact_match(pred_norm, r) for r in refs_norm)
    em25 = max(compute_em25_single(pred_norm, r) for r in refs_norm)
    return float(em), float(em25)


def to_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    if s in ("false", "0", "0.0"):
        return False
    raise ValueError(f"cannot parse is_impossible value {x!r}")


# ----------------------------------------------------------------------
# Per-question value matrices
# ----------------------------------------------------------------------

def build_values(pred_path, ragas_path):
    """Return (row_idx, is_unans, models, values) where values maps
    (family, subset, metric) -> float array of shape (N, n_models), NaN
    where the metric is undefined for that question (outside the subset,
    or an undefined RAGAS score)."""
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    if pred["row_idx"].duplicated().any():
        raise ValueError("duplicate row_idx in predictions file")
    pred = pred.sort_values("row_idx").reset_index(drop=True)

    models = sorted(c.replace("predicted_answer_", "")
                    for c in pred.columns if c.startswith("predicted_answer_"))
    if len(models) < 2:
        raise ValueError("need at least two predicted_answer_* columns")

    is_unans = pred["is_impossible"].map(to_bool).to_numpy()
    refs = pred["correct_answers"].apply(parse_maybe_list).apply(force_list)
    N, M = len(pred), len(models)
    ans_mask = ~is_unans

    em = np.zeros((N, M))
    em25 = np.zeros((N, M))
    flex = np.zeros((N, M))
    strict = np.zeros((N, M))
    for j, m in enumerate(models):
        col = pred[f"predicted_answer_{m}"]
        for i in range(N):
            p = col.iat[i]
            flex[i, j] = float(is_flexible_no_answer("" if pd.isna(p) else p))
            strict[i, j] = float(is_strict_no_answer(p))
            if ans_mask[i]:
                em[i, j], em25[i, j] = em_and_em25(p, refs.iat[i])

    nan = np.full((N, M), np.nan)
    a, u = ans_mask[:, None], is_unans[:, None]
    values = {
        ("task", "ANSWERABLE", "EM"): np.where(a, em, nan),
        ("task", "ANSWERABLE", "EM-25"): np.where(a, em25, nan),
        ("task", "ANSWERABLE", "false_abstention"): np.where(a, flex, nan),
        ("task", "UNANSWERABLE", "flexible_abstention"): np.where(u, flex, nan),
        ("task", "UNANSWERABLE", "strict_abstention"): np.where(u, strict, nan),
        ("task", "ALL", "flexible_overall_EM-25"): np.where(a, em25, flex),
        ("task", "ALL", "strict_overall_EM-25"): np.where(a, em25, strict),
    }

    rg = pd.read_csv(ragas_path, encoding="utf-8-sig")
    if rg.duplicated(["row_idx", "model_key"]).any():
        raise ValueError("duplicate (row_idx, model_key) in RAGAS file")
    missing = set(models) - set(rg["model_key"].unique())
    if missing:
        raise ValueError(f"models missing from RAGAS file: {sorted(missing)}")
    for m in models:
        r = (rg[rg["model_key"] == m].set_index("row_idx")
             .reindex(pred["row_idx"]))
        if r["question_id"].isna().any():
            raise ValueError(f"RAGAS file lacks some row_idx for model {m}")
        if not (r["question_id"].to_numpy() == pred["question_id"].to_numpy()).all():
            raise ValueError(f"question_id mismatch between files for model {m}")
        if not (r["is_impossible"].map(to_bool).to_numpy() == is_unans).all():
            raise ValueError(f"is_impossible mismatch between files for model {m}")
    for metric in RAGAS_METRICS:
        mat = np.column_stack([
            pd.to_numeric(rg[rg["model_key"] == m].set_index("row_idx")
                          .reindex(pred["row_idx"])[metric], errors="coerce")
            .to_numpy(dtype=float)
            for m in models])
        values[("ragas", "ALL", metric)] = mat
        values[("ragas", "ANSWERABLE", metric)] = np.where(a, mat, nan)
        values[("ragas", "UNANSWERABLE", metric)] = np.where(u, mat, nan)

    return pred["row_idx"].to_numpy(), is_unans, models, values


# ----------------------------------------------------------------------
# Reproduction checks against the published numbers
# ----------------------------------------------------------------------

TASK_REFERENCE_COLUMNS = {
    ("task", "ANSWERABLE", "EM"): "answerable_em",
    ("task", "ANSWERABLE", "EM-25"): "answerable_em25",
    ("task", "ANSWERABLE", "false_abstention"): "answerable_abstention_rate",
    ("task", "UNANSWERABLE", "flexible_abstention"): "flexible_abstention_accuracy",
    ("task", "UNANSWERABLE", "strict_abstention"): "strict_abstention_accuracy",
    ("task", "ALL", "flexible_overall_EM-25"): "flexible_overall_em25",
    ("task", "ALL", "strict_overall_EM-25"): "strict_overall_em25",
}


def check_reproduction(values, models, ref_path, ragas_ref_path):
    ok = True
    if ref_path:
        ref = pd.read_csv(ref_path, encoding="utf-8-sig").set_index("model")
        print("\nreproduction check: task metrics (%)")
        for key, col in TASK_REFERENCE_COLUMNS.items():
            for j, m in enumerate(models):
                ours = round(float(np.nanmean(values[key][:, j])) * 100, 2)
                pub = float(ref.loc[m, col])
                good = abs(ours - pub) < 0.006
                ok &= good
                if not good:
                    print(f"  MISMATCH {key[1]}/{key[2]} {m}: ours={ours} published={pub}")
        print("  task metrics:", "PASS" if ok else "FAIL")
    if ragas_ref_path:
        ref = pd.read_csv(ragas_ref_path, encoding="utf-8-sig")
        ragas_ok = True
        for _, r in ref.iterrows():
            if r["model_key"] not in models:
                continue
            j = models.index(r["model_key"])
            for metric in RAGAS_METRICS:
                ours = float(np.nanmean(values[("ragas", r["subset"], metric)][:, j]))
                pub = float(r[metric])
                if not abs(ours - pub) < 1e-5:
                    ragas_ok = False
                    print(f"  MISMATCH RAGAS {r['subset']}/{metric} {r['model_key']}: "
                          f"ours={ours:.6f} published={pub:.6f}")
        print("  RAGAS means:", "PASS" if ragas_ok else "FAIL")
        ok &= ragas_ok
    return ok


# ----------------------------------------------------------------------
# Paired sign-flip permutation test, vectorised over comparisons
# ----------------------------------------------------------------------

def perm_test(D, n_perm, rng, chunk=2000):
    """D: (n, C) paired differences with NaN for missing pairs.
    Returns (mean_diff, p_value, n_valid), each of length C. Missing pairs
    are set to 0, which is equivalent to dropping them, because a flipped
    zero contributes nothing and the mean divides by n_valid only."""
    valid = ~np.isnan(D)
    n_valid = valid.sum(axis=0)
    Z = np.where(valid, D, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = Z.sum(axis=0) / n_valid
    thr = np.abs(obs) - 1e-12
    exceed = np.zeros(D.shape[1])
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        S = rng.integers(0, 2, size=(b, D.shape[0])).astype(np.float64) * 2 - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            perm = (S @ Z) / n_valid
        exceed += (np.abs(perm) >= thr).sum(axis=0)
        done += b
    p = (exceed + 1) / (n_perm + 1)
    p[n_valid == 0] = np.nan
    return obs, p, n_valid


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def stratified_halves(is_unans, rng):
    halves = ([], [])
    for stratum in (False, True):
        idx = np.flatnonzero(is_unans == stratum)
        idx = rng.permutation(idx)
        k = len(idx) // 2
        halves[0].append(idx[:k])
        halves[1].append(idx[k:2 * k])  # drops one item if the stratum is odd
    return np.concatenate(halves[0]), np.concatenate(halves[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ragas", required=True)
    ap.add_argument("--reference", help="final_scores_strict_and_flexible_abstention_corrected.csv")
    ap.add_argument("--ragas-reference", help="ragas_summary_means.csv")
    ap.add_argument("--paper-pvalues", help="paired_permutation_pvalues_all_corrections.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n-perm-full", type=int, default=10000)
    ap.add_argument("--n-perm-half", type=int, default=5000)
    ap.add_argument("--perm-seed", type=int, default=42)
    ap.add_argument("--skip-check", action="store_true",
                    help="continue even if the reproduction check fails")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading and scoring per-question outputs ...")
    row_idx, is_unans, models, values = build_values(args.predictions, args.ragas)
    N = len(row_idx)
    print(f"  {N} questions ({int((~is_unans).sum())} answerable, "
          f"{int(is_unans.sum())} unanswerable), models: {', '.join(models)}")

    if args.reference or args.ragas_reference:
        if not check_reproduction(values, models, args.reference, args.ragas_reference):
            if not args.skip_check:
                print("\nReproduction check FAILED: the full-sample values differ from the "
                      "published ones, so the split-half results would not describe the "
                      "paper. Fix the inputs, or rerun with --skip-check to inspect anyway.")
                sys.exit(1)
    else:
        print("WARNING: no --reference given; full-sample values are not checked.")

    keys = list(values.keys())
    pairs = list(itertools.combinations(range(len(models)), 2))
    comps = [(k, a, b) for k in keys for a, b in pairs]
    D = np.column_stack([values[k][:, a] - values[k][:, b] for k, a, b in comps])

    # ---- full sample ---------------------------------------------------
    rng = np.random.default_rng(args.perm_seed)
    full_diff, full_p, full_n = perm_test(D, args.n_perm_full, rng)
    full_holm = np.empty(len(comps))
    for fam in ("task", "ragas"):
        idx = [i for i, (k, _, _) in enumerate(comps) if k[0] == fam]
        full_holm[idx] = holm(full_p[idx])
    full_sig = full_holm < ALPHA

    # ---- split halves --------------------------------------------------
    S = args.seeds
    half_diff = np.empty((S, 2, len(comps)))
    half_p = np.empty((S, 2, len(comps)))
    half_means = {k: np.empty((S, 2, len(models))) for k in keys}
    print(f"\nrunning {S} split-half seeds ...")
    for s in range(S):
        seed = s + 1
        rng_split = np.random.default_rng(seed)
        rng_perm = np.random.default_rng(10_000 + seed)
        for h, idx in enumerate(stratified_halves(is_unans, rng_split)):
            d, p, _ = perm_test(D[idx], args.n_perm_half, rng_perm)
            half_diff[s, h], half_p[s, h] = d, p
            for k in keys:
                half_means[k][s, h] = np.nanmean(values[k][idx], axis=0)

    # ---- pairwise table ------------------------------------------------
    full_sign = np.sign(full_diff)
    agree = (np.sign(half_diff) == full_sign).all(axis=1)          # (S, C)
    both_sig = agree & (half_p < ALPHA).all(axis=1)
    any_sig = (half_p < ALPHA).any(axis=1)
    scale = np.array([100.0 if k[0] == "task" else 1.0 for k, _, _ in comps])

    paper = None
    if args.paper_pvalues:
        pp = pd.read_csv(args.paper_pvalues, encoding="utf-8-sig")
        paper = {(r["subset"], r["metric"], r["comparison"]): to_bool(r["significant_holm_0.05"])
                 for _, r in pp.iterrows()}

    rows = []
    for i, (k, a, b) in enumerate(comps):
        comparison = f"{models[a]} - {models[b]}"
        hd = half_diff[:, :, i].ravel() * scale[i]
        rows.append({
            "family": k[0], "subset": k[1], "metric": k[2], "comparison": comparison,
            "n_paired_full": int(full_n[i]),
            "full_diff": full_diff[i] * scale[i],
            "full_p": full_p[i], "full_p_holm": full_holm[i],
            "full_significant_holm": bool(full_sig[i]),
            "paper_significant_holm": (paper.get((k[1], k[2], comparison))
                                       if paper is not None else None),
            "pct_seeds_both_halves_same_direction":
                np.nan if full_sign[i] == 0 else 100 * agree[:, i].mean(),
            "pct_seeds_both_halves_significant_same_direction":
                np.nan if full_sign[i] == 0 else 100 * both_sig[:, i].mean(),
            "pct_seeds_any_half_significant": 100 * any_sig[:, i].mean(),
            "half_diff_p2.5": np.percentile(hd, 2.5),
            "half_diff_p97.5": np.percentile(hd, 97.5),
        })
    pw = pd.DataFrame(rows)
    pw.to_csv(os.path.join(args.out_dir, "seed_stability_pairwise.csv"),
              index=False, encoding="utf-8-sig")

    # ---- ranking table -------------------------------------------------
    rank_rows = []
    for k in keys:
        full_m = np.nanmean(values[k], axis=0)
        ranked = np.argsort(-full_m, kind="stable")
        order = models[ranked[0]]
        for prev, j in zip(ranked[:-1], ranked[1:]):
            order += (" = " if full_m[prev] == full_m[j] else " > ") + models[j]
        fs = np.array([np.sign(full_m[a] - full_m[b]) for a, b in pairs])
        hm = half_means[k]                                          # (S, 2, M)
        hs = np.stack([np.sign(hm[..., a] - hm[..., b]) for a, b in pairs], axis=-1)
        # pairs tied in the full sample have no direction to reproduce
        untied = fs != 0
        same = (hs[..., untied] == fs[untied]).all(axis=-1).all(axis=-1)  # (S,)
        rank_rows.append({"family": k[0], "subset": k[1], "metric": k[2],
                          "full_sample_order": order,
                          "pct_seeds_both_halves_same_order": 100 * same.mean()})
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(os.path.join(args.out_dir, "seed_stability_ranking.csv"),
              index=False, encoding="utf-8-sig")

    # ---- per-model spread ----------------------------------------------
    pm_rows = []
    for k in keys:
        sc = 100.0 if k[0] == "task" else 1.0
        full_m = np.nanmean(values[k], axis=0) * sc
        for j, m in enumerate(models):
            est = half_means[k][:, :, j].ravel() * sc
            pm_rows.append({"family": k[0], "subset": k[1], "metric": k[2], "model": m,
                            "full_sample": full_m[j],
                            "half_mean": est.mean(), "half_sd": est.std(ddof=1),
                            "half_p2.5": np.percentile(est, 2.5),
                            "half_p97.5": np.percentile(est, 97.5),
                            "half_min": est.min(), "half_max": est.max()})
    pm = pd.DataFrame(pm_rows)
    pm.to_csv(os.path.join(args.out_dir, "seed_stability_per_model.csv"),
              index=False, encoding="utf-8-sig")

    # ---- headline summary ----------------------------------------------
    summary = {"n_seeds": S, "seeds": f"1..{S}", "half_size": int(len(stratified_halves(
               is_unans, np.random.default_rng(1))[0])),
               "n_perm_full": args.n_perm_full, "n_perm_half": args.n_perm_half,
               "families": {}}
    for fam in ("task", "ragas"):
        f = pw[pw["family"] == fam]
        sig, non = f[f["full_significant_holm"]], f[~f["full_significant_holm"]]
        summary["families"][fam] = {
            "n_comparisons": int(len(f)),
            "n_significant_full_sample_holm": int(len(sig)),
            "significant_min_pct_same_direction":
                float(sig["pct_seeds_both_halves_same_direction"].min()) if len(sig) else None,
            "significant_median_pct_same_direction":
                float(sig["pct_seeds_both_halves_same_direction"].median()) if len(sig) else None,
            "significant_min_pct_both_halves_significant":
                float(sig["pct_seeds_both_halves_significant_same_direction"].min()) if len(sig) else None,
            "significant_median_pct_both_halves_significant":
                float(sig["pct_seeds_both_halves_significant_same_direction"].median()) if len(sig) else None,
            "nonsignificant_median_pct_any_half_significant":
                float(non["pct_seeds_any_half_significant"].median()) if len(non) else None,
            "least_stable_significant_comparison":
                (sig.sort_values("pct_seeds_both_halves_same_direction")
                 .iloc[0][["subset", "metric", "comparison",
                           "pct_seeds_both_halves_same_direction"]].to_dict()
                 if len(sig) else None),
        }
        if fam == "ragas" and paper is not None:
            known = f[f["paper_significant_holm"].notna()]
            agree_n = int((known["paper_significant_holm"].astype(bool)
                           == known["full_significant_holm"]).sum())
            summary["families"][fam]["holm_decisions_matching_paper"] = f"{agree_n}/{len(known)}"
            mism = known[known["paper_significant_holm"].astype(bool)
                         != known["full_significant_holm"]]
            summary["families"][fam]["holm_mismatches"] = (
                mism[["subset", "metric", "comparison", "full_p_holm"]]
                .to_dict(orient="records"))
    with open(os.path.join(args.out_dir, "seed_stability_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- console report ------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("\n=== task metrics: pairwise replication across seeds (pp) ===")
    cols = ["subset", "metric", "comparison", "full_diff", "full_significant_holm",
            "pct_seeds_both_halves_same_direction",
            "pct_seeds_both_halves_significant_same_direction",
            "pct_seeds_any_half_significant"]
    print(pw[pw["family"] == "task"][cols].round(2).to_string(index=False))
    print("\n=== RAGAS: comparisons significant in the full sample ===")
    rs = pw[(pw["family"] == "ragas") & pw["full_significant_holm"]]
    print(rs[cols].round(3).to_string(index=False))
    print("\n=== model ordering reproduced in both halves ===")
    print(rk.round(1).to_string(index=False))
    print("\n=== PASTE THIS BACK ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nwrote 4 files to {args.out_dir}")


if __name__ == "__main__":
    main()
