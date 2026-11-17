"""Load the telco churn dataset splits.

The dataset ships pre-split into three files in ``data/``:

- ``telco_churn_history.csv`` — 5,634 customers with a known ``Churn`` label.
  This is the ONLY data a model may learn from.
- ``telco_churn_current.csv`` — 1,409 customers WITHOUT the label. These are the
  "current" customers we want to score.
- ``telco_churn_test.csv`` — the same 1,409 customers WITH the label, so we can
  check predictions afterwards. Never fit on it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Churn"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """The one data-quality trap in this dataset.

    ``TotalCharges`` is stored as text because brand-new customers (tenure 0)
    have a blank instead of a number. Coerce it to numeric; blanks become NaN,
    which TabPFN handles natively.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return strip_quotes(df)


def strip_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """The CSVs wrap multi-word values in single quotes ('Fiber optic').

    Strip them so pasted rows and file rows look the same to the model.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip("'")
    return df


def load_history(clean: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "telco_churn_history.csv")
    return _clean(df) if clean else df


def load_current(clean: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "telco_churn_current.csv")
    return _clean(df) if clean else df


def load_test(clean: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "telco_churn_test.csv")
    return _clean(df) if clean else df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled frame into features and the ``Churn`` target."""
    return df.drop(columns=[TARGET]), df[TARGET]
