"""The two things the agent can do with the data.

- ``ChurnModel``: the handoff to TabPFN. Fits once on the history split via the
  Prior Labs API and scores whatever rows it's given. The agent never trains
  a model itself.
- ``run_lookup``: runs a short pandas snippet written by the LLM against the
  history data, for questions that are facts about the data as it is.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from .data import TARGET, load_current, load_history, split_xy, strip_quotes


class ChurnModel:
    """A TabPFN classifier fit on the telco history split, lazily and once."""

    def __init__(self) -> None:
        self._model = None
        self.feature_columns: list[str] | None = None
        self.classes_: list[str] | None = None
        self.fit_seconds: float | None = None

    def fit(self) -> "ChurnModel":
        from tabpfn_client import TabPFNClassifier

        X, y = split_xy(load_history())
        self.feature_columns = list(X.columns)
        t0 = time.perf_counter()
        self._model = TabPFNClassifier()
        self._model.fit(X, y)
        self.fit_seconds = time.perf_counter() - t0
        self.classes_ = [str(c) for c in self._model.classes_]
        return self

    @property
    def is_fit(self) -> bool:
        return self._model is not None

    def _align(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Make pasted rows look like the training features."""
        assert self.feature_columns is not None
        rows = strip_quotes(rows.copy())
        if TARGET in rows.columns:
            rows = rows.drop(columns=[TARGET])
        if "TotalCharges" in rows.columns:
            rows["TotalCharges"] = pd.to_numeric(rows["TotalCharges"], errors="coerce")
        missing = [c for c in self.feature_columns if c not in rows.columns]
        for c in missing:
            rows[c] = np.nan  # TabPFN handles missing values natively
        return rows[self.feature_columns]

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        """Probability of churn (class 'Yes') for each row."""
        if not self.is_fit:
            self.fit()
        assert self._model is not None and self.classes_ is not None
        X = self._align(rows)
        proba = self._model.predict_proba(X)
        yes = self.classes_.index("Yes")
        return pd.Series(proba[:, yes], index=rows.index, name="p_churn")

    # --- the two tool-shaped entry points the LLM gets to call -------------

    def score_current_customers(self, top_n: int = 10) -> dict:
        """Score every current (unlabeled) customer; return the riskiest."""
        current = load_current()
        t0 = time.perf_counter()
        p = self.predict_proba(current)
        secs = time.perf_counter() - t0
        out = current.assign(p_churn=p.round(3)).sort_values("p_churn", ascending=False)
        cols = ["tenure", "Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "p_churn"]
        return {
            "n_scored": int(len(out)),
            "share_above_50pct": round(float((p > 0.5).mean()), 3),
            "top_customers": out[cols].head(top_n).reset_index().rename(columns={"index": "row"}).to_dict("records"),
            "seconds": round(secs, 1),
            "note": "p_churn is TabPFN's calibrated probability that the customer churns. Base rate in history is 26.5%.",
        }

    def score_rows(self, rows_csv: str) -> dict:
        """Score rows pasted as CSV (with or without a header)."""
        from .data import DATA_DIR

        header = pd.read_csv(DATA_DIR / "telco_churn_current.csv", nrows=0).columns
        first = rows_csv.strip().splitlines()[0]
        has_header = first.split(",")[0].strip().strip("'\"") in header
        rows = pd.read_csv(
            io.StringIO(rows_csv.strip()),
            header=0 if has_header else None,
            names=None if has_header else list(header),
            quotechar="'",  # the dataset quotes multi-word values with single quotes
        )
        p = self.predict_proba(rows)
        return {
            "predictions": [
                {"row": int(i), "p_churn": round(float(v), 3), "label": "Yes" if v >= 0.5 else "No"}
                for i, v in p.items()
            ],
            "note": "p_churn is TabPFN's probability of churn; label uses a 0.5 cut-off.",
        }


def run_lookup(code: str) -> str:
    """Run a pandas snippet against the history data and return what it prints.

    The snippet sees a DataFrame called ``df`` (the labeled history split) and
    ``pd``. This is a workshop convenience — in a real app you'd sandbox it.
    """
    df = load_history()
    buf = io.StringIO()
    local: dict = {"df": df, "pd": pd, "np": np}
    try:
        with redirect_stdout(buf):
            exec(code, local)  # noqa: S102 - workshop demo, trusted LLM output
    except Exception as e:  # surface errors to the LLM so it can retry
        return f"ERROR: {type(e).__name__}: {e}"
    out = buf.getvalue().strip()
    return out[:4000] if out else "(the code printed nothing — use print())"
