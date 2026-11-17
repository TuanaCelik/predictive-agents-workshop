#!/usr/bin/env python
"""Fit TabPFN on a labeled CSV and score another CSV, through the Prior Labs API.

This is the handoff the Claude Code skill calls. It does NO modelling decisions
of its own: the agent decides which file is history, which columns to drop and
which text columns are really numbers; this script just runs the fit + predict.

Usage:
  python scripts/tabpfn_predict.py \
      --history data/telco_churn_history.csv --target Churn \
      --score data/telco_churn_current.csv \
      --numeric TotalCharges \
      --out reports/telco_churn/scores.csv

  Add --regression for a numeric target. --drop COL to leave columns out
  (IDs, leaky columns). --numeric COL to coerce a text column to numbers.

Prints a JSON summary to stdout; writes the scored rows (features + prediction)
to --out. Needs TABPFN_TOKEN in the environment (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import os

try:  # pick up TABPFN_TOKEN from the repo's .env, if there is one
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("TABPFN_TOKEN") and sys.stdin.isatty():  # a human at a terminal: ask
    from getpass import getpass

    os.environ["TABPFN_TOKEN"] = getpass("Prior Labs API key (platform.priorlabs.ai → API Keys): ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True, help="labeled CSV to fit on")
    ap.add_argument("--target", required=True, help="target column name")
    ap.add_argument("--score", required=True, help="CSV of rows to score (target column ignored if present)")
    ap.add_argument("--out", required=True, help="where to write the scored rows")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--drop", action="append", default=[], help="column to drop (repeatable)")
    ap.add_argument("--numeric", action="append", default=[], help="text column to coerce to numeric (repeatable)")
    args = ap.parse_args()

    hist = pd.read_csv(args.history)
    score = pd.read_csv(args.score)
    for col in args.numeric:
        for df in (hist, score):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    drop = [c for c in args.drop if c in hist.columns]
    X = hist.drop(columns=[args.target, *drop])
    y = hist[args.target]
    X_score = score.drop(columns=[c for c in [args.target, *drop] if c in score.columns])
    missing = [c for c in X.columns if c not in X_score.columns]
    for c in missing:
        X_score[c] = pd.NA
    X_score = X_score[X.columns]

    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    t0 = time.perf_counter()
    model = TabPFNRegressor() if args.regression else TabPFNClassifier()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = score.copy()
    summary: dict = {
        "task": "regression" if args.regression else "classification",
        "target": args.target,
        "fit_rows": int(len(X)),
        "scored_rows": int(len(X_score)),
        "features": ", ".join(X.columns),
        "dropped": drop,
        "coerced_numeric": args.numeric,
        "missing_in_score_filled_with_NA": missing,
    }
    if args.regression:
        out["prediction"] = model.predict(X_score)
        summary["prediction_mean"] = float(out["prediction"].mean())
    else:
        proba = model.predict_proba(X_score)
        classes = [str(c) for c in model.classes_]
        for i, c in enumerate(classes):
            out[f"p_{c}"] = proba[:, i].round(4)
        out["prediction"] = [classes[i] for i in proba.argmax(axis=1)]
        summary["classes"] = classes
        summary["predicted_share"] = {c: round(float((out["prediction"] == c).mean()), 3) for c in classes}
    summary["fit_seconds"] = round(fit_s, 1)
    summary["predict_seconds"] = round(time.perf_counter() - t1, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary["out"] = args.out
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
