# Predictive agents: handing prediction tasks to tabular foundation models

Workshop repo for **AI Dev NYC** (DeepLearning.AI, Nov 30 – Dec 1, 2026).

Agents built on LLMs are good at reading questions, writing code and deciding what to
do. They are bad at *predicting things from tables* — and they'll happily try anyway.
This workshop builds an agent that knows the difference: it recognises when a question
is a **prediction task**, and hands that part to **TabPFN**, a tabular foundation
model, through an API. Everything else it does itself.

```
you ──ask──▶ agent (LLM: frame the question, clean the data, decide)
                │
                ├─ "how many churned?"        ──▶ pandas          (lookup)
                └─ "who *will* churn?"        ──▶ TabPFN API      (prediction)
                                                  ◀── calibrated probabilities
```

The running example is **telco customer churn**: 7,043 customers of a telecom company,
what plan they're on, and whether they left.

## What you need

Set these up before the workshop — the first one has a sign-up step.

1. **A Prior Labs account** for the TabPFN API. Create one at
   [platform.priorlabs.ai](https://platform.priorlabs.ai), redeem the promo code
   **`TABPFN4AIDEV`**, and create an API key under *API Keys*.
2. **Claude.** The notebooks call the Claude API, so you need an API key from
   [platform.claude.com](https://platform.claude.com). Step 3 uses
   [Claude Code](https://code.claude.com/docs/en/setup) as the coding agent — install it
   if you want to follow that step as written. (You can use another coding agent for that
   step; see [Using a different coding agent](#using-a-different-coding-agent).)
3. **Optional: a Jev account.** In step 4 the routing decision is made by
   [Jev](https://typesafe.ai), TypeSafe's decision model. Get a key at
   [console.typesafe.ai](https://console.typesafe.ai). Without it the agent falls back
   to routing with Claude, so you can skip this and still run everything.
4. **Python 3.10+**, with [uv](https://docs.astral.sh/uv/) (recommended) or pip.

## Setup

```bash
git clone https://github.com/TuanaCelik/predictive-agents-workshop
cd predictive-agents-workshop
```

Install dependencies with one of:

```bash
uv sync && source .venv/bin/activate
```

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**API keys.** Every notebook starts with a cell that looks for `TABPFN_TOKEN`,
`ANTHROPIC_API_KEY` (and, in notebook 04, optionally `TYPESAFE_API_KEY`). A key that's
already in your environment or in a `.env` file is used as-is; anything missing is
asked for with a hidden prompt and kept for that session only. So you can either:

```bash
cp .env.example .env     # fill in the keys once, never get prompted
```

or just paste them when asked. (`python -m agent` and `scripts/tabpfn_predict.py` do
the same when run from a terminal. Claude Code in step 3 runs the script
non-interactively, so for that step use `.env` or export the key in your shell.)

Start Jupyter with `jupyter lab` and open the `notebooks/` folder. (Step 5 is a web
page instead: `python -m app`.)

## The workshop

### 1. Getting started — LLMs vs tabular foundation models

`notebooks/01_getting_started.ipynb`

Load the churn data, ask Claude to predict who churns, then ask TabPFN the same thing
through the API. Compare accuracy, speed and what the two models can actually learn
from. This sets up the one idea everything else rests on: *use each model for what
it's for*.

### 2. Prediction or lookup?

`notebooks/02_prediction_decision.ipynb`

Before an agent can hand off a prediction, it has to notice it's being asked for one.
*"How many customers churned?"* is a lookup — the answer is in the data. *"Who will
churn?"* is a prediction — it isn't. We build the classifier with Claude (a JSON
schema with an enum), test it on a batch of questions, and look at what it can't
give us: a confidence.

### 3. Hand prediction tasks to TabPFN from a coding agent

No notebook for this one — it happens in Claude Code. The repo contains a **skill**
(`.claude/skills/tabpfn-handoff/SKILL.md`) that tells the agent how to recognise a
prediction task, do the analyst's data work, and hand the fit + predict to TabPFN via
`scripts/tabpfn_predict.py`. `CLAUDE.md` holds the standing rules (never train a model
yourself, never fit on the test split, …).

With your venv active, run `claude` in the repo and try these in order:

1. *"How many customers do we have, and what share of them churned?"*
   — a lookup. The agent answers from the data; no TabPFN call.
2. *"Fit a churn model on our customer history."*
   — the agent inspects the data, catches `TotalCharges` being stored as text, hands
   the fit to TabPFN, and stops.
3. *"Which of our current customers are most likely to churn? Give me the top 10."*
4. *"Is this customer likely to churn?"* followed by a pasted row — e.g. the last line
   of `data/telco_churn_current.csv` (the true answer is `Yes`):
   `Female,0,Yes,Yes,1,Yes,Yes,'Fiber optic',Yes,No,Yes,No,Yes,Yes,Month-to-month,Yes,'Electronic check',102.45,102.45`
5. *"How well did those predictions do against the real outcomes?"*
   — runs `scripts/evaluate.py` against the held-out labels.

Open the skill file while you do this. The point of the step is seeing how little the
skill has to say: it's almost entirely about the *data work* and *when* to hand off, not
about modelling — because there's no model selection or tuning to describe.

#### Optional: the TabPFN MCP server

By default the handoff runs through `scripts/tabpfn_predict.py` — one command, no
prompts. The repo also ships `.mcp.json` pointing at Prior Labs' hosted MCP server, so
the agent can call TabPFN as **tool calls** (`prepare_train_set_upload` → `fit` →
`predict`) instead. Same model, more visible handoff, a few more permission prompts
(the agent uploads your CSVs with `curl`).

**Turn it on:** the first time you run `claude` in the repo it asks whether to use the
project's MCP server — say yes, then run `/mcp` once and sign in with your Prior Labs
account. The skill detects the `tabpfn` tools and uses them.

**Turn it off:** say no at that prompt, or disable it later from `/mcp`. To get the
prompt again (either direction), run `claude mcp reset-project-choices` in the repo.

With the server off, the skill falls back to the script. Nothing else changes.

This step is adapted from [egargari/tabpfn-agent](https://github.com/egargari/tabpfn-agent),
which goes further (XGBoost comparisons, reports, more datasets).

### 4. The predictive agent

`notebooks/04_predictive_agent.ipynb` · code in `agent/`

Everything in one loop. **Jev** classifies the question (prediction / lookup / other)
and returns a probability for each — not just a label. **TabPFN** does the predicting.
**Claude** does the analyst's job: writes the pandas for lookups, calls the TabPFN
tools for predictions, explains the result. When the router isn't confident, the agent
gets both tool sets and can ask instead of committing.

Also runnable from the terminal:

```bash
python -m agent "Which of our current customers are most likely to churn?"
python -m agent            # interactive
```

### 5. The same agent, as a page you can watch

The agent from step 4 behind a small local web page. Type a question and watch it move
through the workflow: the router's probabilities fill in, the branch it picks lights up,
TabPFN reports its fit and scoring times, and the answer and a full trace of tool calls
appear below. Tabby walks the path.

**Run it** (from the repo root, with your venv active):

```bash
python -m app
```

Then open [http://localhost:8000](http://localhost:8000) in a browser. The first thing
the command does is check for your keys: anything already in your environment or `.env`
is used, anything missing is asked for (hidden input). The Jev key is optional; without
it the router dropdown only offers Claude.

A few things to try, in order:

1. *"How many customers do we have, and what share of them churned?"*: routes to lookup,
   no TabPFN call.
2. *"Which of our current customers are most likely to churn? Top 10."*: routes to
   prediction. The first prediction fits TabPFN on 5,634 rows over the API (a few
   seconds); after that the fit is cached, so later predictions only score.
3. The ambiguous one from the chips (*"…I mean the ones who already did, how many?"*):
   with Jev, watch the probability split and both branches light up.
4. Switch the **Router** dropdown between Jev and Claude and ask the same question
   twice: Jev returns a distribution, Claude returns a label.

Stop it with `Ctrl-C`. To use another port: `PORT=8080 python -m app`.

It's one FastAPI endpoint streaming the agent's events to one HTML page
(`app/server.py`, `app/static/index.html`); the agent code is untouched. The page uses
Prior Labs' colours; type is Inter and JetBrains Mono from Google Fonts, so it looks the
same on every laptop. The `run_pandas` tool runs code Claude writes on your machine,
same as in notebook 04: fine for a workshop, not a pattern to put on the internet.

## Using a different coding agent

Step 3 is written for Claude Code, but nothing in it is Claude-specific. The skill is a
markdown file with instructions; the handoff is a Python script. To use another coding
agent (Codex, Gemini CLI, Cursor, …), point it at `CLAUDE.md` and
`.claude/skills/tabpfn-handoff/SKILL.md` as its project instructions — most agents
have an equivalent file (`AGENTS.md`, `GEMINI.md`, `.cursorrules`) you can copy them into.

The notebooks and `agent/` use the Claude API directly; swapping in another LLM means
changing the tool-use loop in `agent/agent.py` to your provider's SDK.

## Repo layout

| Path | What it is |
|---|---|
| `notebooks/01_getting_started.ipynb` | LLMs vs TFMs, TabPFN via the API |
| `notebooks/02_prediction_decision.ipynb` | LLM classifies prediction vs lookup |
| `notebooks/04_predictive_agent.ipynb` | Jev routes, TabPFN predicts, Claude talks |
| `app/` | Step 5: the local web page (`python -m app`) |
| `agent/` | The agent: `data.py`, `router.py` (Jev), `tools.py` (TabPFN handoff), `agent.py`, `keys.py` (key prompts) |
| `.claude/skills/tabpfn-handoff/SKILL.md` | The Claude Code skill for step 3 |
| `CLAUDE.md` | Standing rules for the coding agent |
| `.claude/settings.json` | Pre-approved permissions for Claude Code (python, read-only shell, the MCP tools) |
| `.mcp.json` | Optional: Prior Labs' hosted TabPFN MCP server (off until you enable it) |
| `scripts/tabpfn_predict.py` | Fit on one CSV, score another, via the Prior Labs API |
| `scripts/evaluate.py` | ROC AUC / R² of a scores file against held-out labels |
| `data/` | Telco churn, split into `history` (labeled), `current` (unlabeled), `test` (labels for `current`) |
| `reports/` | Created by step 3 as you go; gitignored |

## The data split

| File | Rows | Role |
|---|---|---|
| `telco_churn_history.csv` | 5,634 | Labeled. **The only data a model may learn from.** |
| `telco_churn_current.csv` | 1,409 | Same held-out customers, `Churn` column removed. What gets scored. |
| `telco_churn_test.csv` | 1,409 | Those customers *with* `Churn`, same order. Evaluation only. |

Known trap: `TotalCharges` is read as text because brand-new customers have a blank.
Watching the agent catch that on its own is part of the fun.

## Notes

- Predictions run on Prior Labs' hosted API, so the data is uploaded there. The
  workshop promo code covers what the notebooks use.
- Running the full `02` and `04` notebooks makes a few dozen Claude API calls — small,
  but not free.
- The `run_pandas` tool in `agent/tools.py` executes code the LLM writes, unsandboxed.
  Fine for a workshop on your own laptop; not a pattern to ship.
