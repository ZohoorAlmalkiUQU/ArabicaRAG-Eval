# -*- coding: utf-8 -*-
"""
second_judge_faithfulness_check.py
==================================

Optional, descriptive, no API calls. The two judges disagree on Faithfulness
(the second judge rates Mistral-7B-Instruct-v0.1 lower). This script shows
where that disagreement sits:

1. missing Faithfulness scores per model under each judge (pairwise exclusion
   could select different items per model);
2. Faithfulness under each judge by model, subset, and whether the answer is
   an abstention (flexible matcher of abstention_matcher.py);
3. the second-judge minus original-judge difference per item, by the same groups.

Usage (from the repository root, after second_judge_compare.py)
---------------------------------------------------------------
python second_judge_faithfulness_check.py
"""

import argparse
import os

import pandas as pd

from abstention_matcher import is_flexible_no_answer, is_strict_no_answer

PER_QUESTION_CSV = os.path.join(
    "arabicaqa_rag_results", "ragas_full_llama_mistral_command",
    "ragas_per_question_all_models.csv",
)
SECOND_DIR = os.path.join("arabicaqa_rag_results", "second_judge",
                          "qwen__qwen3-235b-a22b-2507")
MODEL_KEYS = ["command", "llama", "mistral"]


def main():
    ap = argparse.ArgumentParser(description="Where the Faithfulness disagreement sits.")
    ap.add_argument("--per-question-csv", default=PER_QUESTION_CSV)
    ap.add_argument("--second-dir", default=SECOND_DIR)
    args = ap.parse_args()

    s = pd.read_csv(os.path.join(args.second_dir, "second_judge_per_question.csv"),
                    encoding="utf-8-sig")
    a = pd.read_csv(args.per_question_csv, encoding="utf-8-sig")[
        ["question_id", "model_key", "answer"]]
    m = s.merge(a, on=["question_id", "model_key"], how="left")
    m["is_answerable"] = m["is_answerable"].astype(bool)
    m["answer"] = m["answer"].fillna("").astype(str)
    m["abstains"] = m["answer"].apply(is_flexible_no_answer)
    m["strict_phrase"] = m["answer"].apply(is_strict_no_answer)
    m["subset"] = m["is_answerable"].map({True: "ANSWERABLE", False: "UNANSWERABLE"})
    m["J1"] = m["faithfulness_original"]
    m["J2"] = m["faithfulness"]

    out_dir = os.path.join(args.second_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 26 + " FAITHFULNESS CHECK (paste this back) " + "=" * 26)
    print(f"items: {len(m)}  questions: {m['question_id'].nunique()}")

    print("\n1) Missing Faithfulness per model (original / second):")
    for mk in MODEL_KEYS:
        g = m[m["model_key"] == mk]
        print(f"   {mk:8s} {int(g['J1'].isna().sum()):3d} / {int(g['J2'].isna().sum()):3d}"
              f"   (of {len(g)}; second-judge NaN among abstentions: "
              f"{int(g.loc[g['abstains'], 'J2'].isna().sum())}/{int(g['abstains'].sum())})")

    rows = []
    for (sub, mk, ab), g in m.groupby(["subset", "model_key", "abstains"]):
        both = g.dropna(subset=["J1", "J2"])
        rows.append({"subset": sub, "model": mk, "abstains": ab, "n": len(g),
                     "n_complete": len(both), "nan_J1": int(g["J1"].isna().sum()),
                     "nan_J2": int(g["J2"].isna().sum()),
                     "mean_J1": both["J1"].mean(), "mean_J2": both["J2"].mean(),
                     "mean_J2_minus_J1": (both["J2"] - both["J1"]).mean(),
                     "strict_phrase_share": g["strict_phrase"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "faithfulness_by_abstention.csv"), index=False,
             encoding="utf-8-sig")
    print("\n2) Faithfulness by subset, model and abstention (complete pairs):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n3) All items per model (complete pairs): mean original / second / difference")
    for sub in ["ANSWERABLE", "UNANSWERABLE"]:
        for mk in MODEL_KEYS:
            g = m[(m["subset"] == sub) & (m["model_key"] == mk)].dropna(subset=["J1", "J2"])
            print(f"   {sub:12s} {mk:8s} n={len(g):3d}  {g['J1'].mean():.3f} / {g['J2'].mean():.3f}"
                  f" / {(g['J2'] - g['J1']).mean():+.3f}   abstaining share {g['abstains'].mean():.2f}")
    print("=" * 90)
    print(f"Saved {os.path.join(out_dir, 'faithfulness_by_abstention.csv')}")


if __name__ == "__main__":
    main()
