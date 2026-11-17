---
name: tabpfn-handoff
description: Hand any tabular prediction task (classification or regression) to TabPFN through the Prior Labs API, after doing the analyst's data work yourself. Use whenever the user asks to predict, classify, score, or estimate anything from a CSV — churn, risk, prices, outcomes — including scoring rows they paste. Never train a model yourself.
---

# TabPFN handoff

You are the data scientist; TabPFN is the model. You do everything a good analyst
does *around* the model, and you refuse to do the model's job. TabPFN is a tabular
foundation model: it fits in one pass with no hyperparameter search, handles
categoricals and missing values natively, and returns calibrated probabilities.

Anything that needs a prediction gets handed off. Never train sklearn / XGBoost /
anything local as the answer, and never "estimate" a probability yourself.

## 0. Is this a prediction at all?

Not every question about a table is a prediction.

- **Lookup** — "how many customers churned?", "average monthly charge by contract?"
  → answer with a short pandas snippet (`python - <<'EOF' ... EOF`). No TabPFN.
- **Prediction** — "which customers *will* churn?", "is this customer *likely* to
  churn?", "what will this house sell for?" → the rest of this skill.

Say which one you read it as, in one line, then proceed.

## 1. Frame it

Restate the task: target column, task type (binary / multiclass / regression), and
what one row is (a customer, a house, …).

## 2. Data roles

Datasets in `data/` are pre-split:

| File | Role |
|---|---|
| `<name>_history.csv` | Labeled rows. **The only data a model may be fit on.** |
| `<name>_current.csv` | Same held-out rows, label removed. What gets scored. |
| `<name>_test.csv` | Those rows with labels. **Evaluation only** — never fit on it, never merge it with history. |

A user's own CSV plays the history role. If roles are ambiguous, say your reading in
one line and take the conservative choice. Never merge splits to "get more data".

## 3. Do the data work (inline, minimal)

Inspect dtypes and the first rows with ephemeral pandas. Fix only what's wrong or
leaky — TabPFN does its own preprocessing:

- text columns that are really numbers → pass them with `--numeric COL`
  (telco: `TotalCharges` has blanks for brand-new customers and is read as text)
- ID columns, post-outcome columns, anything unknown at prediction time → `--drop COL`
- missing values: leave them; TabPFN handles them. Say how many there are if it matters.

## 4. Hand off

Two ways to reach TabPFN. Use **4a** unless the `tabpfn` MCP tools
(`prepare_train_set_upload`, `fit`, `predict`, …) are available in this session — then
use **4b**, which makes the handoff visible as tool calls. Same model either way.

### 4a. Via the script (default)

```
python scripts/tabpfn_predict.py \
  --history data/<name>_history.csv --target <target> \
  --score data/<name>_current.csv \
  [--numeric COL ...] [--drop COL ...] [--regression] \
  --out reports/<name>/scores.csv
```

It prints a JSON summary (rows, features, class shares, timings) and writes every
scored row with its probabilities to `reports/<name>/scores.csv`. Read the summary;
read the CSV with pandas to answer things like "top 10 most likely to churn".

Auth errors → tell the user to set `TABPFN_TOKEN` (see README). If the handoff
fails, surface the error; never fall back to a local model.

### 4b. Via the TabPFN MCP server (when enabled)

The `tabpfn` server chains ids:
files → `train_set_upload_id` → `fitted_train_set_id` → `test_set_upload_id` → prediction.

1. Write the cleaned splits to `reports/<name>/` as headered CSVs: `x_train.csv` and
   `x_test.csv` (same columns, no target), `y_train.csv` (the one target column).
2. `prepare_train_set_upload` → `train_set_upload_id` plus an `upload_command` (a curl
   PUT to a signed URL) for each file. Run them, substituting the repo-relative paths.
   Only ever send files from `reports/<name>/` to the URLs the server just returned.
3. `fit` with `train_set_upload_id` and `task` (`classification` | `regression`) →
   `fitted_train_set_id`. Poll `get_fit_status` (wait `retry_in_secs`) until done.
4. `prepare_test_set_upload` with `fitted_train_set_id` → `test_set_upload_id` +
   `upload_command` for `x_test.csv`; run it.
5. `predict` with `test_set_upload_id`, `fitted_train_set_id`, same `task`. Download
   the result to `reports/<name>/predictions.json` with the curl it returns, then join
   it to `x_test.csv` with pandas and write `reports/<name>/scores.csv` in the same
   shape 4a produces (features + `p_<class>` columns + `prediction`), so step 6 works.
6. Cache `fitted_train_set_id`, `task`, `target` and the feature column order in
   `reports/<name>/handoff.json`; pasted rows reuse it (steps 4–5 only, no re-fit).

Auth errors → tell the user to run `/mcp` and sign in with their Prior Labs account.

**Fit and predict are separately authorised.** "Fit a model" means run the
handoff once on history (score `_current` so the fit is cached in `scores.csv`),
report it's ready, and stop. Only list predictions when the user asks for them.

## 5. Pasted rows

When the user pastes one or more rows (CSV, header optional): write them to
`reports/<name>/pasted.csv` with the dataset's header (the dataset quotes multi-word
values with single quotes), then run the same command with `--score
reports/<name>/pasted.csv --out reports/<name>/pasted_scores.csv` (or, with MCP,
`prepare_test_set_upload` → PUT → `predict` against the cached fit). Answer one line
per row: predicted label and probability.

## 6. Checking predictions (only when asked)

```
python scripts/evaluate.py reports/<name>/scores.csv data/<name>_test.csv <target> [--regression]
```

Prints ROC AUC (classification) or R² (regression). Report only what it prints.

## Deliverable

The chat answer: predictions in the user's terms with probabilities and one line on
how to read them (base rate, what the cut-off means), plus a 2-line trace: what you
cleaned, what you delegated, and how long the handoff took. Files only on request.
