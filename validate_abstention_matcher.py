# -*- coding: utf-8 -*-
"""
validate_abstention_matcher.py
==============================

Human validation of the flexible abstention matcher (Reviewer 1, Concern 5).

Two subcommands:

  build   Draw a blinded, stratified annotation sample from the predictions
          file and write one CSV per annotator. Matcher flags and model
          identity are withheld from the sheet.

  score   Read the completed annotation sheets, adjudicate, and report:
            - inter-annotator agreement (Cohen's kappa, raw agreement)
            - matcher precision / recall / F1 against the human labels,
              per model and pooled, with 95% bootstrap CIs
            - population-weighted re-estimates of flexible abstention
              accuracy (unanswerable) and false-abstention rate (answerable),
              for comparison with Tables 3 and 4 of the manuscript
            - an error taxonomy: which patterns produce the false positives

Usage
-----
  python validate_abstention_matcher.py build \
      --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
      --out-dir     arabicaqa_rag_results/annotation \
      --per-stratum 100

  # annotators fill the `human_label` column with 1 (abstention) or 0 (answer)

  python validate_abstention_matcher.py score \
      --sheets arabicaqa_rag_results/annotation/annotator_A_filled.csv \
               arabicaqa_rag_results/annotation/annotator_B_filled.csv \
      --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
      --out-dir arabicaqa_rag_results/annotation

Sampling design
---------------
The matcher's two error modes are asymmetric and occur at very different
base rates, so a simple random sample would spend almost all annotation
effort on the majority stratum. We therefore stratify on the matcher's own
output and re-weight when estimating population quantities:

  stratum FLAG_ANS    answerable   & flagged flexible   -> false-positive risk
  stratum NOFLAG_ANS  answerable   & not flagged
  stratum FLAG_UNANS  unanswerable & flagged flexible
  stratum NOFLAG_UNANS unanswerable & not flagged        -> false-negative risk

Precision is estimated within the flagged strata, recall from both, and each
population estimate is a stratum-size-weighted combination with a bootstrap
CI that resamples within strata.
"""

import argparse
import ast
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from abstention_matcher import (
    NO_ANSWER,
    NO_ANSWER_PATTERNS,
    is_flexible_no_answer,
    is_strict_no_answer,
    matched_patterns,
    normalize_arabic_text,
)

SEED = 42
N_BOOT = 10000

MODEL_DISPLAY_NAMES = {
    "command": "command-r7b-12-2024",
    "llama": "llama-3-8b-instruct",
    "mistral": "mistral-7b-instruct-v0.1",
}

ANNOTATION_GUIDELINE = """\
ANNOTATION GUIDELINE — abstention labelling
===========================================

You will see a question and a system output. Assign ONE binary label to the
OUTPUT only.

  1 = ABSTENTION. The output declines to supply an answer: it states that the
      answer is absent from the context, that the information is unavailable,
      or that the system cannot answer or does not know. It asserts no
      candidate answer to the question.

  0 = ANSWER ATTEMPT. The output asserts a candidate answer to the question,
      whether or not that answer is correct, and whether or not it is verbose,
      hedged, or partially expressed.

Decision rules for the hard cases:

  * Judge the OUTPUT, not the question. Do NOT consider whether the question
    is answerable, and do NOT consider whether the answer given is correct.
    An output that confidently asserts a wrong answer is 0, not 1.

  * Negation inside a substantive answer is 0. If the output answers the
    question and the answer itself happens to be a negative statement about
    the world (e.g. "there is no land border between the two countries" as
    the answer to a question about borders), label 0. The system has answered.

  * Mixed outputs — an answer followed or preceded by a hedge — are 0 if a
    candidate answer is asserted anywhere in the output, 1 only if no
    candidate answer is asserted at all.

  * Empty outputs, or outputs consisting only of punctuation, are 0.

  * Language is irrelevant: an English refusal is 1, an Arabic refusal is 1.

  * The exact prescribed phrase is not required. A paraphrased refusal is
    still 1.

Fill only the `human_label` column. Do not reorder or delete rows. If a row is
genuinely undecidable, write `x` and add a note in `human_note`.
"""


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def to_bool_safe(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"Unexpected is_impossible value: {x!r}")


def load_long(predictions_path):
    """Load the wide predictions file and melt to one row per (question, model)."""
    df = pd.read_csv(predictions_path)
    df["is_impossible"] = df["is_impossible"].apply(to_bool_safe)

    prefix = "predicted_answer_"
    model_keys = [c[len(prefix):] for c in df.columns if c.startswith(prefix)]
    if not model_keys:
        raise KeyError("No predicted_answer_* columns found.")

    qcol = "question" if "question" in df.columns else df.columns[0]

    rows = []
    for mk in model_keys:
        pcol = prefix + mk
        for idx, r in df.iterrows():
            pred = r[pcol]
            pred = "" if (isinstance(pred, float) and math.isnan(pred)) else str(pred)
            rows.append({
                "item_id": f"{idx}_{mk}",
                "row_index": idx,
                "model_key": mk,
                "model": MODEL_DISPLAY_NAMES.get(mk, mk),
                "question": r[qcol],
                "prediction": pred,
                "is_impossible": bool(r["is_impossible"]),
            })

    out = pd.DataFrame(rows)
    out["strict_flag"] = out["prediction"].apply(is_strict_no_answer).astype(int)
    out["flexible_flag"] = out["prediction"].apply(is_flexible_no_answer).astype(int)
    out["patterns_fired"] = out["prediction"].apply(
        lambda t: ";".join(f"P{i}" for i in matched_patterns(t))
    )
    out["stratum"] = np.where(
        out["is_impossible"],
        np.where(out["flexible_flag"] == 1, "FLAG_UNANS", "NOFLAG_UNANS"),
        np.where(out["flexible_flag"] == 1, "FLAG_ANS", "NOFLAG_ANS"),
    )
    return out


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, rng=None):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(SEED)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = v[rng.integers(0, v.size, v.size)].mean()
    return v.mean(), np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def cohen_kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    po = (a == b).mean()
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    pe = sum((a == L).mean() * (b == L).mean() for L in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------

def cmd_build(args):
    long_df = load_long(args.predictions)
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # stratum census over the full population, needed later for re-weighting
    census = (long_df.groupby(["model", "stratum"])
                     .size()
                     .rename("N_population")
                     .reset_index())
    census.to_csv(os.path.join(args.out_dir, "stratum_census.csv"),
                  index=False, encoding="utf-8-sig")

    picks = []
    for (model, stratum), grp in long_df.groupby(["model", "stratum"]):
        n = min(args.per_stratum, len(grp))
        if n == 0:
            continue
        idx = rng.choice(grp.index.values, size=n, replace=False)
        picks.append(long_df.loc[idx])

    sample = pd.concat(picks).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # key file retains the blinding information; annotators never see it
    key_cols = ["item_id", "row_index", "model_key", "model", "is_impossible",
                "stratum", "strict_flag", "flexible_flag", "patterns_fired"]
    sample[key_cols].to_csv(os.path.join(args.out_dir, "annotation_key.csv"),
                            index=False, encoding="utf-8-sig")

    sheet = sample[["item_id", "question", "prediction"]].copy()
    sheet["human_label"] = ""
    sheet["human_note"] = ""

    for ann in args.annotators:
        p = os.path.join(args.out_dir, f"annotator_{ann}.csv")
        sheet.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"wrote {p}  ({len(sheet)} items)")

    with open(os.path.join(args.out_dir, "ANNOTATION_GUIDELINE.txt"), "w",
              encoding="utf-8") as f:
        f.write(ANNOTATION_GUIDELINE)

    print("\nstratum census (full population, 3 models x 1000 questions):")
    print(census.to_string(index=False))
    print(f"\nsampled {len(sheet)} items "
          f"({args.per_stratum} per stratum per model, capped at stratum size)")
    print("Annotators see item_id, question and prediction only; model identity,")
    print("answerability and matcher flags are withheld in annotation_key.csv.")


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------

def cmd_score(args):
    long_df = load_long(args.predictions)
    key = pd.read_csv(os.path.join(args.out_dir, "annotation_key.csv"))
    census = pd.read_csv(os.path.join(args.out_dir, "stratum_census.csv"))

    sheets = []
    for p in args.sheets:
        s = pd.read_csv(p)
        s = s[["item_id", "human_label"]].copy()
        s["human_label"] = s["human_label"].astype(str).str.strip()
        s = s.rename(columns={"human_label": os.path.basename(p)})
        sheets.append(s)

    merged = sheets[0]
    for s in sheets[1:]:
        merged = merged.merge(s, on="item_id")

    label_cols = [c for c in merged.columns if c != "item_id"]
    if len(label_cols) < 2:
        print("WARNING: fewer than two annotator sheets supplied; "
              "no agreement statistic will be computed.")

    report = {}

    # --- inter-annotator agreement, pre-adjudication --------------------
    if len(label_cols) >= 2:
        sub = merged[merged[label_cols].apply(
            lambda r: all(v in ("0", "1") for v in r), axis=1)]
        k, po = cohen_kappa(sub[label_cols[0]].astype(int),
                            sub[label_cols[1]].astype(int))
        report["agreement"] = {"n": int(len(sub)), "cohen_kappa": round(k, 4),
                               "raw_agreement": round(po, 4)}
        print(f"\nInter-annotator agreement (n={len(sub)}): "
              f"kappa={k:.3f}, raw={po:.3f}")

        disagree = merged[merged[label_cols[0]] != merged[label_cols[1]]]
        dp = os.path.join(args.out_dir, "disagreements_for_adjudication.csv")
        disagree.merge(long_df[["item_id", "question", "prediction"]],
                       on="item_id").to_csv(dp, index=False, encoding="utf-8-sig")
        print(f"{len(disagree)} disagreements written to {dp}")

    # --- adjudicated labels ---------------------------------------------
    adj_path = os.path.join(args.out_dir, "adjudicated.csv")
    if os.path.exists(adj_path):
        adj = pd.read_csv(adj_path)[["item_id", "human_label"]]
        print(f"using adjudicated labels from {adj_path}")
    else:
        agreed = merged[merged[label_cols].nunique(axis=1) == 1].copy()
        agreed["human_label"] = agreed[label_cols[0]]
        adj = agreed[["item_id", "human_label"]]
        print(f"\nNOTE: {adj_path} not found. Scoring the {len(adj)} items on "
              f"which annotators already agree. Adjudicate the disagreements "
              f"and rerun for the final figures.")

    adj = adj[adj["human_label"].astype(str).isin(["0", "1"])]
    adj["human_label"] = adj["human_label"].astype(int)

    d = (key.drop(columns=[c for c in ("patterns_fired",) if c in key.columns])
            .merge(adj, on="item_id")
            .merge(long_df[["item_id", "prediction", "patterns_fired"]], on="item_id"))

    # --- matcher quality --------------------------------------------------
    rng = np.random.default_rng(SEED)
    rows = []
    for model in sorted(d["model"].unique()) + ["POOLED"]:
        dm = d if model == "POOLED" else d[d["model"] == model]
        tp = int(((dm.flexible_flag == 1) & (dm.human_label == 1)).sum())
        fp = int(((dm.flexible_flag == 1) & (dm.human_label == 0)).sum())
        fn = int(((dm.flexible_flag == 0) & (dm.human_label == 1)).sum())
        tn = int(((dm.flexible_flag == 0) & (dm.human_label == 0)).sum())

        prec_v = dm[dm.flexible_flag == 1]["human_label"].values
        rec_v = dm[dm.human_label == 1]["flexible_flag"].values
        pm, pl, ph = bootstrap_ci(prec_v, rng=rng) if len(prec_v) else (np.nan,) * 3
        rm, rl, rh = bootstrap_ci(rec_v, rng=rng) if len(rec_v) else (np.nan,) * 3
        f1 = 2 * pm * rm / (pm + rm) if (pm + rm) else float("nan")

        rows.append({"model": model, "n": len(dm),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": round(pm, 4), "prec_ci_low": round(pl, 4),
                     "prec_ci_high": round(ph, 4),
                     "recall": round(rm, 4), "rec_ci_low": round(rl, 4),
                     "rec_ci_high": round(rh, 4), "f1": round(f1, 4)})

    qual = pd.DataFrame(rows)
    qual.to_csv(os.path.join(args.out_dir, "matcher_quality.csv"),
                index=False, encoding="utf-8-sig")
    print("\nMatcher quality against adjudicated human labels "
          "(sample strata, unweighted):")
    print(qual.to_string(index=False))

    # --- population-weighted re-estimates --------------------------------
    # For each model and each question type, the human abstention rate is
    #   sum_s (N_s / N) * mean(human_label | stratum s)
    est_rows = []
    for model in sorted(d["model"].unique()):
        for impossible, name in [(True, "unanswerable_abstention_accuracy"),
                                 (False, "answerable_false_abstention_rate")]:
            strata = [s for s in census["stratum"].unique()
                      if s.endswith("UNANS" if impossible else "ANS")
                      and not (impossible is False and s.endswith("UNANS"))]
            num_m, num_b = 0.0, np.zeros(N_BOOT)
            denom = 0
            ok = True
            for s in strata:
                Ns = int(census[(census.model == model) &
                                (census.stratum == s)]["N_population"].sum())
                denom += Ns
                vals = d[(d.model == model) & (d.stratum == s)]["human_label"].values
                if Ns == 0:
                    continue
                if len(vals) == 0:
                    ok = False
                    continue
                num_m += Ns * vals.mean()
                boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                                  for _ in range(N_BOOT)])
                num_b += Ns * boots
            if not ok or denom == 0:
                continue
            est = num_m / denom
            bo = num_b / denom
            matcher_val = long_df[(long_df.model == model) &
                                  (long_df.is_impossible == impossible)]["flexible_flag"].mean()
            est_rows.append({
                "model": model, "quantity": name,
                "matcher_estimate_pct": round(100 * matcher_val, 2),
                "human_estimate_pct": round(100 * est, 2),
                "human_ci_low_pct": round(100 * np.percentile(bo, 2.5), 2),
                "human_ci_high_pct": round(100 * np.percentile(bo, 97.5), 2),
                "difference_pp": round(100 * (matcher_val - est), 2),
            })

    est = pd.DataFrame(est_rows)
    est.to_csv(os.path.join(args.out_dir, "population_reestimates.csv"),
               index=False, encoding="utf-8-sig")
    print("\nPopulation-weighted human re-estimates vs. reported matcher values:")
    print(est.to_string(index=False))

    # --- error taxonomy ---------------------------------------------------
    fps = d[(d.flexible_flag == 1) & (d.human_label == 0)]
    tax = Counter()
    for _, r in fps.iterrows():
        for p in str(r["patterns_fired"]).split(";"):
            if p:
                tax[p] += 1
    tax_df = (pd.DataFrame(sorted(tax.items(), key=lambda kv: -kv[1]),
                           columns=["pattern", "false_positive_count"])
              if tax else pd.DataFrame(columns=["pattern", "false_positive_count"]))
    tax_df.to_csv(os.path.join(args.out_dir, "false_positive_taxonomy.csv"),
                  index=False, encoding="utf-8-sig")
    fps.to_csv(os.path.join(args.out_dir, "false_positive_items.csv"),
               index=False, encoding="utf-8-sig")
    print("\nFalse positives by pattern:")
    print(tax_df.to_string(index=False) if len(tax_df) else "  none")

    fns = d[(d.flexible_flag == 0) & (d.human_label == 1)]
    fns.to_csv(os.path.join(args.out_dir, "false_negative_items.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n{len(fns)} false negatives written to false_negative_items.csv "
          f"(inspect these for abstention wordings absent from the pattern list)")

    report["matcher_quality"] = qual.to_dict(orient="records")
    report["population_reestimates"] = est.to_dict(orient="records")
    with open(os.path.join(args.out_dir, "validation_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--predictions", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--per-stratum", type=int, default=100)
    b.add_argument("--annotators", nargs="+", default=["A", "B"])
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score")
    s.add_argument("--sheets", nargs="+", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
