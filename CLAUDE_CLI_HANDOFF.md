# Raiders Analysis 2026: Project Setup Handoff

## Purpose of this document

This is a scoped handoff for Claude CLI to set up the initial project structure and first runnable data-ingestion milestone for `raiders-analysis-2026`. This document covers ONLY the setup phase. Do not build the archetype model, scoring logic, or Streamlit UI in this phase. Those are separate handoffs after the data layer is verified.

## Project context

The end goal is a Streamlit app that grades the 2026 Las Vegas Raiders offensive roster against the offensive archetype that Klint Kubiak ran as OC of the 2024 New Orleans Saints and 2025 Seattle Seahawks. The approach is data-driven similarity scoring: build feature vectors for Kubiak's prior starters, compute distance from the current Raiders roster, output position grades and an overall offensive grade.

Kubiak is now the Raiders head coach (hired February 2026). Andrew Janocko is the new OC. We are modeling Kubiak's tendencies, not Janocko's.

## Design principles to follow throughout

1. Simple beats complex. One correct path, no fallbacks, no alternatives.
2. Each function has a single responsibility.
3. Fail fast. Throw clear errors when preconditions aren't met. Do not silently fall back to default values.
4. No backup mechanisms. Trust the primary one.
5. Clarity over backward compatibility.
6. Surgical changes. When editing, change only what needs to change.
7. Evidence-based debugging. Add targeted logging, not blanket print statements.
8. Fix root causes, not symptoms.
9. Do not over-engineer. If the user did not ask for it, do not build it.
10. Do not write emojis in any file, including README.md.

## Environment requirements

- macOS
- Python 3.13 (newest stable)
- `uv` as the package manager (faster than pip, handles lockfiles natively)
- Git installed and configured with the user's GitHub account
- VSCode as the editor

If `uv` is not installed, install it first with:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Repository setup steps

### Step 1: Create the repository

Create a new public GitHub repository named `raiders-analysis-2026`. Initialize it locally first, then push.

```
mkdir raiders-analysis-2026
cd raiders-analysis-2026
git init
```

### Step 2: Initialize the Python project with uv

```
uv init --python 3.13
```

This creates `pyproject.toml` and a `.python-version` file pinned to 3.13.

### Step 3: Verification checkpoint 1

Stop and confirm with the user that:

- The directory `raiders-analysis-2026` exists in the chosen location
- `uv init` ran without errors
- `pyproject.toml` and `.python-version` are present
- `python3 --version` inside the venv reports 3.13.x

Do not proceed until the user confirms.

## Project structure

### Step 4: Create the directory structure

```
raiders-analysis-2026/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── archetype.py
│   ├── scoring.py
│   └── roster.py
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── notebooks/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

In this handoff, only `data_ingestion.py` will have working code. The other module files should be created as empty placeholders with a single-line docstring describing their future responsibility. Do not write logic in them yet.

Each module's single responsibility:

- `data_ingestion.py`: pull data from `nfl_data_py`, cache to DuckDB, expose query functions
- `archetype.py`: build Kubiak reference profiles from 2024 Saints and 2025 Seahawks starters
- `scoring.py`: compute similarity between current Raiders players and the archetype
- `roster.py`: manage the current 2026 Raiders roster state and player swaps
- `streamlit_app.py`: the UI

### Step 5: Add dependencies

Add these to the project with uv. Do not install extras or alternative packages.

```
uv add nfl_data_py pandas duckdb scikit-learn streamlit plotly
uv add --dev pytest ruff
```

### Step 6: Verification checkpoint 2

Stop and confirm with the user that:

- The directory tree matches the structure above
- `uv sync` completes without errors
- `uv run python3 -c "import nfl_data_py, pandas, duckdb, sklearn, streamlit, plotly; print('ok')"` prints `ok`

Do not proceed until the user confirms.

## Configuration files

### Step 7: Create .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/

# uv
.python-version

# Data caches and local DuckDB
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
*.duckdb
*.duckdb.wal

# IDE
.vscode/
.idea/

# OS
.DS_Store

# Notebooks
.ipynb_checkpoints/
```

Add `.gitkeep` files to `data/raw/` and `data/processed/` so the directories are tracked but their contents are not.

### Step 8: Create README.md

Write a professional README with no emojis. Include these sections:

- Project title: Raiders Analysis 2026
- One-paragraph overview describing the goal: data-driven scheme fit analysis for the 2026 Las Vegas Raiders offensive roster against Klint Kubiak's OC tendencies from 2024 (Saints) and 2025 (Seahawks)
- Tech stack list
- Setup instructions (clone, `uv sync`, how to activate the environment)
- Project structure section showing the directory tree
- Current status: "In development. Data ingestion phase."
- A "Notes on data sources" section explaining that all NFL data is sourced from the open-source `nfl_data_py` package, and that OL grades will use proxy metrics rather than paid PFF data in v1

### Step 9: Verification checkpoint 3

Stop and confirm with the user that:

- `.gitignore` and `README.md` are present and look right
- `git status` shows the expected files staged for the initial commit

Do not proceed until the user confirms.

## First runnable milestone: data ingestion

### Step 10: Implement data_ingestion.py

Write `src/data_ingestion.py` to do the following, and nothing more:

1. Connect to a DuckDB file at `data/raw/nfl.duckdb` (create if missing)
2. Define one function per dataset, each pulling from `nfl_data_py` and writing to a DuckDB table:
   - `ingest_snap_counts(years: list[int])` -> table `snap_counts`
   - `ingest_weekly_stats(years: list[int])` -> table `weekly_stats`
   - `ingest_combine(years: list[int])` -> table `combine`
   - `ingest_ngs_passing(years: list[int])` -> table `ngs_passing`
   - `ingest_ngs_rushing(years: list[int])` -> table `ngs_rushing`
   - `ingest_ngs_receiving(years: list[int])` -> table `ngs_receiving`
   - `ingest_rosters(years: list[int])` -> table `rosters`
3. A `main()` function that runs all seven ingest functions for years `[2024, 2025, 2026]` (rosters only for 2026 will likely be partial; that is expected)
4. If any pull returns an empty DataFrame, raise a `RuntimeError` with a clear message naming the dataset and year. Do not silently continue. Do not retry. Do not fall back.

Use this structure for each ingest function:

```python
def ingest_snap_counts(years: list[int]) -> None:
    """Pull NFL snap counts for given years and write to DuckDB."""
    df = nfl.import_snap_counts(years)
    if df.empty:
        raise RuntimeError(f"snap_counts pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS snap_counts")
    con.register("df", df)
    con.execute("CREATE TABLE snap_counts AS SELECT * FROM df")
    con.close()
```

Constants like `DB_PATH` go at the top of the module.

### Step 11: Run the ingestion

```
uv run python3 src/data_ingestion.py
```

Expected behavior: the script prints which dataset is being pulled, completes in roughly 1 to 3 minutes, and exits cleanly with no errors. A DuckDB file appears at `data/raw/nfl.duckdb`.

If any function raises `RuntimeError`, stop. Do not try to work around it. Surface the exact error to the user and ask how to proceed.

### Step 12: Verification checkpoint 4 (final for this handoff)

Run a verification query and show the user the output:

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')
tables = ['snap_counts','weekly_stats','combine','ngs_passing','ngs_rushing','ngs_receiving','rosters']
for t in tables:
    n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {n} rows')
"
```

Confirm with the user that:

- All seven tables exist
- Row counts are non-zero for 2024 and 2025 datasets
- The 2026 rosters table may have fewer rows but should have at least the Raiders' currently-signed players

Once this checkpoint passes, the setup handoff is complete. Stop here. Do not proceed to archetype building or any other phase. Wait for the next handoff document from the user.

## Initial commit and push

After the final verification passes:

```
git add .
git commit -m "Initial project setup: structure, dependencies, and data ingestion"
git branch -M main
gh repo create raiders-analysis-2026 --public --source=. --remote=origin --push
```

If `gh` CLI is not installed, prompt the user to create the repo manually on github.com and then run:

```
git remote add origin https://github.com/<username>/raiders-analysis-2026.git
git push -u origin main
```

## What NOT to do in this handoff

- Do not write the archetype building logic in `archetype.py`
- Do not write any scoring logic in `scoring.py`
- Do not write the Streamlit app beyond the placeholder file
- Do not add fallback paths if `nfl_data_py` returns unexpected data; raise and stop
- Do not add caching layers beyond DuckDB
- Do not install packages not listed in Step 5
- Do not add emojis to any file
- Do not use em dashes in any documentation
- Do not skip the verification checkpoints

## Summary of checkpoints

There are four points in this handoff where you stop and wait for user confirmation before proceeding:

1. After uv init (Step 3)
2. After directory structure and dependencies (Step 6)
3. After config files and README (Step 9)
4. After data ingestion runs successfully (Step 12)

Follow them.
