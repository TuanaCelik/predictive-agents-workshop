#!/usr/bin/env python
"""Score a predictions file against held-out labels.

Usage:
  python scripts/evaluate.py reports/telco_churn/scores.csv data/telco_churn_test.csv Churn
  python scripts/evaluate.py <scores.csv> <labeled.csv> <target> --regression

The scores file is what scripts/tabpfn_predict.py wrote (same row order as the
file that was scored). Prints ROC AUC for classification, R² for regression.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("labeled")
    ap.add_argument("target")
    ap.add_argument("--regression", action="store_true")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    truth = pd.read_csv(args.labeled)[args.target]
    assert len(scores) == len(truth), "row counts differ — were the same rows scored?"

    if args.regression:
        print(json.dumps({"metric": "r2", "value": round(float(r2_score(truth, scores["prediction"])), 4), "n": len(truth)}))
        return
    classes = sorted(truth.astype(str).unique())
    if len(classes) == 2:
        pos = classes[-1]  # 'Yes' > 'No', 1 > 0
        auc = roc_auc_score((truth.astype(str) == pos).astype(int), scores[f"p_{pos}"])
    else:
        auc = roc_auc_score(truth.astype(str), scores[[f"p_{c}" for c in classes]], multi_class="ovr")
    acc = float((scores["prediction"].astype(str) == truth.astype(str)).mean())
    print(json.dumps({"metric": "roc_auc", "value": round(float(auc), 4), "accuracy": round(acc, 4), "n": len(truth)}))


if __name__ == "__main__":
    main()
