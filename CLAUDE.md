# predictive-agents-workshop

Workshop repo: an agent that does the analyst's data work itself but **never trains a
model** — every prediction on tabular data is handed to TabPFN through the Prior Labs
API (`scripts/tabpfn_predict.py`, wrapped by the `tabpfn-handoff` skill).

## Rules

- For ANY prediction / scoring / classification request on tabular data, use the
  `tabpfn-handoff` skill. Never train a local model as the answer, never guess a
  probability yourself.
- Questions that are facts about the data as it is (counts, averages, filters) are
  **lookups**: answer with a short inline pandas snippet, no TabPFN call. Say which
  kind of question you read it as.
- Run Python as `python ...` from the repo root (the venv from the README is active).
  Ephemeral snippets: `python - <<'EOF' ... EOF`. Don't write per-task `.py` files.
- `data/<name>_history.csv` is the ONLY data a model may be fit on. `_test.csv` is
  for evaluation only; `_current.csv` is what gets scored. Never merge splits.
- Working files go in `reports/<name>/` (gitignored). `<name>` is the dataset stem
  exactly as in `data/` (`telco_churn`).
- Headliner dataset: telco churn, target `Churn`, ~26.5% positive. Known trap:
  `TotalCharges` is read as text because new customers have a blank — pass
  `--numeric TotalCharges`.
- Fit and predict are separately authorised: "fit a model" ends after the handoff
  runs and you report it's ready. Don't list predictions nobody asked for.
- Default deliverable is the chat answer plus a 2-line trace (what you cleaned, what
  you delegated, wall-clock). Write report files only when asked.
- Two handoff paths, same model: `scripts/tabpfn_predict.py` (default) or the
  `tabpfn` MCP server from `.mcp.json` when its tools are available in the session.
  The skill covers both. Never mix them for one task.
- Call it "TabPFN" — the API picks the model version.
- Never download local TabPFN weights; the hosted API is the point of this workshop.
