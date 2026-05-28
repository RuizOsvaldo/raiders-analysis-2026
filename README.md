# Raiders Analysis 2026

Data-driven scheme fit analysis for the 2026 Las Vegas Raiders offensive roster against Klint Kubiak's offensive coordinator tendencies from the 2024 New Orleans Saints and 2025 Seattle Seahawks. Kubiak is now the Raiders head coach. The project builds feature vectors for his prior starters, computes distance scores against the current Raiders roster, and outputs position grades and an overall offensive grade via a Streamlit app.

## Tech stack

- Python 3.13
- uv (package manager)
- nfl_data_py (data source)
- pandas
- DuckDB (local data cache)
- scikit-learn (similarity scoring)
- Streamlit (UI)
- plotly (charts)
- pytest, ruff (dev)

## Setup

```
git clone https://github.com/osvaldoruiz/raiders-analysis-2026.git
cd raiders-analysis-2026
uv sync
```

To activate the virtual environment manually:

```
source .venv/bin/activate
```

Or prefix any command with `uv run` to execute inside the project environment without activating.

## Project structure

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

## Current status

In development. Archetype phase complete. Scoring and UI pending.

## Notes on data sources

All NFL data is sourced from the open-source `nfl_data_py` package. Offensive line grades use proxy metrics (snap counts, position, roster data) rather than paid PFF data in v1.

Data is sourced from the open-source nfl_data_py package, which reads from the nflverse data repositories. Note that nfl_data_py is officially deprecated in favor of nflreadpy, with no further maintenance planned. It is retained here because it works correctly against the project's pinned dependencies and because the historical 2024 and 2025 season data being pulled is static. If in-season 2026 data pulls break in the future, the migration path is nflreadpy, which returns Polars DataFrames and would require a conversion layer to pandas.

## Archetype methodology

The Kubiak scheme profile and position archetypes are built from two reference
seasons: the 2024 New Orleans Saints (Kubiak's first OC year, weighted 40%) and
the 2025 Seattle Seahawks (Kubiak's most recent OC year, weighted 60%). The
Seahawks season is weighted higher because it is more recent and reflects the
offense Kubiak built with greater autonomy.

Players contribute to the archetype if they recorded at least 20% offensive
snap share in at least 4 games of the reference season. Their contribution is
weighted by their total offensive snap count, so a 70% snap-share player has
a larger influence on the archetype than a 25% snap-share player.

Red zone is defined as yardline_100 <= 20. Several features are split between
red-zone and non-red-zone contexts because a coach's tendencies often change
when the field shrinks.

Offensive line features use team-level proxies (adjusted line yards by gap and
team pressure rate allowed) rather than individual snap-by-snap grades. The
individual OL data exists in paid services like PFF but is not available
free. This is a known v1 limitation; OL grades will be noisier than skill
position grades.
