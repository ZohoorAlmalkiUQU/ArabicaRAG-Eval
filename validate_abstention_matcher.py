# -*- coding: utf-8 -*-
"""
validate_abstention_matcher.py
==============================

Human validation of the flexible abstention matcher (Reviewer 1, Comment 3).

Three subcommands:

  build   Draw a blinded, stratified annotation sample from the predictions
          file and write one CSV per annotator. Matcher flags and model
          identity are withheld from the sheet.

  check   Pipeline self-test, to run once after build and before annotating.
          Substitutes the matcher's own flags for the human labels and
          verifies that the weighted estimator then reproduces the reported
          matcher figures exactly (precision = recall = 1, zero difference).

  score   Read the completed annotation sheets, adjudicate, and report:
            - inter-annotator agreement (Cohen's kappa, raw agreement)
            - matcher precision / recall / F1 against the human labels,
              per model and pooled, population-weighted, with 95% intervals
            - population-weighted re-estimates of flexible abstention
              accuracy (unanswerable) and false-abstention rate (answerable),
              for comparison with Tables 3 and 4 of the manuscript
            - an error taxonomy: which patterns produce the false positives

Usage
-----
  python validate_abstention_matcher.py build \
      --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
      --out-dir     arabicaqa_rag_results/annotation \
      --per-stratum 30

  python validate_abstention_matcher.py check \
      --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
      --out-dir     arabicaqa_rag_results/annotation

  # annotators fill the `human_label` column with 1 (abstention) or 0 (answer)

  python validate_abstention_matcher.py score \
      --sheets arabicaqa_rag_results/annotation/annotator_A_filled.csv \
               arabicaqa_rag_results/annotation/annotator_B_filled.csv \
      --predictions arabicaqa_rag_results/predictions/comparison_llama_mistral_command_1000.csv \
      --out-dir arabicaqa_rag_results/annotation

  # score writes adjudication_template.csv; fill the blank human_label cells
  # (the disagreements), save it as adjudicated.csv in --out-dir, rerun score

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

Precision, recall and each population estimate are combinations of the
per-stratum human abstention rates weighted by stratum population size N_s,
not by sample size. Intervals are the 2.5th and 97.5th percentiles of 10,000
Monte Carlo draws from independent per-stratum Jeffreys posteriors,
Beta(k + 1/2, n - k + 1/2). A within-stratum bootstrap is not used because at
30 items per stratum many strata are expected to be unanimous, and the
bootstrap of a unanimous sample has zero width.
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
N_DRAWS = 10000

FLAG_STRATA = ("FLAG_ANS", "FLAG_UNANS")
NOFLAG_STRATA = ("NOFLAG_ANS", "NOFLAG_UNANS")

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


def clean_label(x):
    """Normalise an annotator cell to '1', '0', 'x', or '' (blank).

    Spreadsheet round-trips turn 1 into '1.0' or ' 1', so cells are not
    compared raw.
    """
    s = str(x).strip().lower()
    if s in ("1", "1.0"):
        return "1"
    if s in ("0", "0.0"):
        return "0"
    if s == "x":
        return "x"
    return ""


def read_raw_sheet(path):
    # dtype=str: a label column with any blank cell would otherwise be parsed
    # as float, and 1 would come back as '1.0'
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", encoding_errors="replace")


def read_sheet(path):
    """Read one annotator sheet, keeping only item_id and the cleaned label.

    Only these two columns are used, so a sheet whose Arabic text was damaged
    by a non-UTF-8 save still scores correctly.
    """
    s = read_raw_sheet(path)
    missing = {"item_id", "human_label"} - set(s.columns)
    if missing:
        raise KeyError(f"{path}: missing column(s) {sorted(missing)}. "
                       f"Save the sheet as 'CSV UTF-8 (Comma delimited)'.")
    s = s[["item_id", "human_label"]].copy()
    s["human_label"] = s["human_label"].apply(clean_label)
    n_blank = int((s["human_label"] == "").sum())
    if n_blank:
        print(f"WARNING: {path}: {n_blank} rows have no label")
    return s


def has_labels(path):
    if not os.path.exists(path):
        return False
    s = read_raw_sheet(path)
    return ("human_label" in s.columns
            and (s["human_label"].apply(clean_label) != "").any())


def stratum_posteriors(d, rng):
    """Per-stratum human abstention rate: plug-in value and posterior draws.

    Returns {(model, stratum): (k / n, draws from Beta(k + 1/2, n - k + 1/2))}.
    """
    post = {}
    for (model, stratum), grp in d.groupby(["model", "stratum"]):
        n = len(grp)
        k = int(grp["human_label"].sum())
        post[(model, stratum)] = (k / n, rng.beta(k + 0.5, n - k + 0.5, N_DRAWS))
    return post


def abstention_mass(cells, post, census_n):
    """Estimated population count of human-judged abstentions over `cells`.

    Each stratum contributes N_s * rate_s. Returns (plug-in value, draws).
    """
    point, draws = np.float64(0.0), np.zeros(N_DRAWS)
    for cell in cells:
        Ns = census_n.get(cell, 0)
        if Ns == 0:
            continue
        if cell not in post:
            raise ValueError(f"stratum {cell} has N={Ns} in the census but no "
                             f"labelled items, so it cannot be re-weighted.")
        p, dr = post[cell]
        point += Ns * p
        draws += Ns * dr
    return point, draws


def ci95(draws):
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


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

    started = [p for p in (os.path.join(args.out_dir, f"annotator_{a}.csv")
                           for a in args.annotators) if has_labels(p)]
    if started and not args.force:
        sys.exit("Refusing to overwrite sheets that already contain labels:\n  "
                 + "\n  ".join(started)
                 + "\nBack them up, or pass --force to rebuild anyway.")

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
# scoring core, shared by check and score
# ----------------------------------------------------------------------

def evaluate(key, labels, long_df, census):
    """Score final labels against the matcher.

    `labels` holds item_id and an integer human_label (1 = abstention).
    Returns (matcher-quality table, population re-estimates, merged items).
    """
    d = (key.drop(columns=[c for c in ("patterns_fired",) if c in key.columns])
            .merge(labels, on="item_id")
            .merge(long_df[["item_id", "prediction", "patterns_fired"]], on="item_id"))

    census_n = {(r.model, r.stratum): int(r.N_population)
                for r in census.itertuples(index=False)}
    rng = np.random.default_rng(SEED)
    post = stratum_posteriors(d, rng)
    models = sorted(d["model"].unique())

    # --- matcher quality --------------------------------------------------
    # Precision = P(human abstention | flagged) and recall = P(flagged | human
    # abstention) over the population, so each stratum enters with weight N_s.
    # The sample is equal-allocation, so the raw TP/FP/FN/TN counts below are
    # reported for transparency only and do not estimate population rates.
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for model in models + ["POOLED"]:
            ms = models if model == "POOLED" else [model]
            dm = d[d["model"].isin(ms)]
            flag_cells = [(m, s) for m in ms for s in FLAG_STRATA]
            noflag_cells = [(m, s) for m in ms for s in NOFLAG_STRATA]

            n_flagged = sum(census_n.get(c, 0) for c in flag_cells)
            a_flag, a_flag_d = abstention_mass(flag_cells, post, census_n)
            a_noflag, a_noflag_d = abstention_mass(noflag_cells, post, census_n)

            prec, prec_d = a_flag / n_flagged, a_flag_d / n_flagged
            rec = a_flag / (a_flag + a_noflag)
            rec_d = a_flag_d / (a_flag_d + a_noflag_d)
            f1 = 2 * prec * rec / (prec + rec)
            f1_d = 2 * prec_d * rec_d / (prec_d + rec_d)
            (pl, ph), (rl, rh), (fl, fh) = ci95(prec_d), ci95(rec_d), ci95(f1_d)

            rows.append({
                "model": model, "n_labelled": len(dm),
                "TP": int(((dm.flexible_flag == 1) & (dm.human_label == 1)).sum()),
                "FP": int(((dm.flexible_flag == 1) & (dm.human_label == 0)).sum()),
                "FN": int(((dm.flexible_flag == 0) & (dm.human_label == 1)).sum()),
                "TN": int(((dm.flexible_flag == 0) & (dm.human_label == 0)).sum()),
                "precision": round(prec, 4), "prec_ci_low": round(pl, 4),
                "prec_ci_high": round(ph, 4),
                "recall": round(rec, 4), "rec_ci_low": round(rl, 4),
                "rec_ci_high": round(rh, 4),
                "f1": round(f1, 4), "f1_ci_low": round(fl, 4),
                "f1_ci_high": round(fh, 4),
            })

    # --- population-weighted re-estimates --------------------------------
    # For each model and each question type, the human abstention rate is
    #   sum_s N_s * mean(human_label | stratum s) / sum_s N_s
    est_rows = []
    for model in models:
        for impossible, name, strata in [
                (True, "unanswerable_abstention_accuracy", ("FLAG_UNANS", "NOFLAG_UNANS")),
                (False, "answerable_false_abstention_rate", ("FLAG_ANS", "NOFLAG_ANS"))]:
            cells = [(model, s) for s in strata]
            denom = sum(census_n.get(c, 0) for c in cells)
            a, a_d = abstention_mass(cells, post, census_n)
            est, (lo, hi) = a / denom, ci95(a_d / denom)
            matcher_val = long_df[(long_df.model == model) &
                                  (long_df.is_impossible == impossible)]["flexible_flag"].mean()
            est_rows.append({
                "model": model, "quantity": name,
                "matcher_estimate_pct": round(100 * matcher_val, 2),
                "human_estimate_pct": round(100 * est, 2),
                "human_ci_low_pct": round(100 * lo, 2),
                "human_ci_high_pct": round(100 * hi, 2),
                "difference_pp": round(100 * (matcher_val - est), 2),
            })

    return pd.DataFrame(rows), pd.DataFrame(est_rows), d


def read_key_and_census(out_dir):
    key = pd.read_csv(os.path.join(out_dir, "annotation_key.csv"), encoding="utf-8-sig")
    census = pd.read_csv(os.path.join(out_dir, "stratum_census.csv"), encoding="utf-8-sig")
    return key, census


# ----------------------------------------------------------------------
# check
# ----------------------------------------------------------------------

def cmd_check(args):
    """Run the scoring path with the matcher's own flags as the labels.

    If the key, census and predictions are mutually consistent, the matcher
    then agrees with itself perfectly and every population re-estimate equals
    the figure reported in the manuscript.
    """
    long_df = load_long(args.predictions)
    key, census = read_key_and_census(args.out_dir)

    now = key.merge(long_df[["item_id", "flexible_flag"]], on="item_id",
                    suffixes=("", "_now"))
    if len(now) != len(key) or (now.flexible_flag != now.flexible_flag_now).any():
        sys.exit("check: FAIL. annotation_key.csv does not match the predictions "
                 "file; rerun build.")

    labels = key[["item_id", "flexible_flag"]].rename(
        columns={"flexible_flag": "human_label"})
    qual, est, _ = evaluate(key, labels, long_df, census)
    print(qual[["model", "n_labelled", "precision", "recall"]].to_string(index=False))
    print()
    print(est[["model", "quantity", "matcher_estimate_pct",
               "human_estimate_pct"]].to_string(index=False))

    ok = (np.allclose(qual["precision"], 1) and np.allclose(qual["recall"], 1)
          and np.allclose(est["difference_pp"], 0))
    print("\ncheck:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------

def cmd_score(args):
    long_df = load_long(args.predictions)
    key, census = read_key_and_census(args.out_dir)

    sheets = [read_sheet(p).rename(columns={"human_label": os.path.basename(p)})
              for p in args.sheets]
    merged = sheets[0]
    for s in sheets[1:]:
        merged = merged.merge(s, on="item_id")

    label_cols = [c for c in merged.columns if c != "item_id"]
    if len(label_cols) < 2:
        print("WARNING: fewer than two annotator sheets supplied; "
              "no agreement statistic will be computed.")

    # an item is settled when every annotator gave it the same 0/1 label
    merged["settled"] = ((merged[label_cols].nunique(axis=1) == 1)
                         & merged[label_cols[0]].isin(["0", "1"]))

    report = {}

    # --- inter-annotator agreement, pre-adjudication --------------------
    if len(label_cols) >= 2:
        sub = merged[merged[label_cols].isin(["0", "1"]).all(axis=1)]
        k, po = cohen_kappa(sub[label_cols[0]].astype(int),
                            sub[label_cols[1]].astype(int))
        report["agreement"] = {"n": int(len(sub)), "cohen_kappa": round(k, 4),
                               "raw_agreement": round(po, 4),
                               "n_unsettled": int((~merged.settled).sum())}
        print(f"\nInter-annotator agreement (n={len(sub)}): "
              f"kappa={k:.3f}, raw={po:.3f}")

        disagree = merged[~merged.settled].drop(columns="settled")
        dp = os.path.join(args.out_dir, "disagreements_for_adjudication.csv")
        disagree.merge(long_df[["item_id", "question", "prediction"]],
                       on="item_id").to_csv(dp, index=False, encoding="utf-8-sig")
        print(f"{len(disagree)} disagreements (incl. x / blank) written to {dp}")

    # adjudication template: every item, settled labels pre-filled, the rest blank
    tmpl = merged.merge(long_df[["item_id", "question", "prediction"]], on="item_id")
    tmpl["human_label"] = np.where(tmpl["settled"], tmpl[label_cols[0]], "")
    tp = os.path.join(args.out_dir, "adjudication_template.csv")
    tmpl.drop(columns="settled").to_csv(tp, index=False, encoding="utf-8-sig")

    # --- adjudicated labels ---------------------------------------------
    adj_path = os.path.join(args.out_dir, "adjudicated.csv")
    if os.path.exists(adj_path):
        adj = read_sheet(adj_path)
        print(f"using adjudicated labels from {adj_path}")
    else:
        adj = (merged[merged.settled]
               .rename(columns={label_cols[0]: "human_label"})[["item_id", "human_label"]])
        print(f"\nNOTE: {adj_path} not found. Scoring the {len(adj)} items on "
              f"which annotators already agree. Fill the blank human_label cells "
              f"in {tp}, save it as {adj_path}, and rerun for the final figures.")

    adj = adj[adj["human_label"].isin(["0", "1"])].copy()
    adj["human_label"] = adj["human_label"].astype(int)
    print(f"scoring {len(adj)} of {len(key)} sampled items")

    qual, est, d = evaluate(key, adj, long_df, census)

    qual.to_csv(os.path.join(args.out_dir, "matcher_quality.csv"),
                index=False, encoding="utf-8-sig")
    print("\nMatcher quality against adjudicated human labels "
          "(precision / recall / F1 population-weighted; counts are raw sample):")
    print(qual.to_string(index=False))

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
    report["false_positive_taxonomy"] = tax_df.to_dict(orient="records")
    with open(os.path.join(args.out_dir, "validation_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--predictions", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--per-stratum", type=int, default=30)
    b.add_argument("--annotators", nargs="+", default=["A", "B"])
    b.add_argument("--force", action="store_true",
                   help="overwrite annotator sheets even if they contain labels")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check")
    c.add_argument("--predictions", required=True)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("score")
    s.add_argument("--sheets", nargs="+", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
