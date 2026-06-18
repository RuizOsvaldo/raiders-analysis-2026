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

## Running the app

After running the data ingestion and scoring pipelines:

```
uv run python3 src/data_ingestion.py
uv run python3 src/archetype.py
uv run python3 src/scoring.py
uv run python3 src/roster.py
```

Launch the Streamlit app:

```
cd app
uv run streamlit run streamlit_app.py
```

The app opens at http://localhost:8501 with four pages:
- **Roster Overview** - filterable table of all 45 offensive players with both grades
- **Player Detail** - per-player grade breakdown and archetype comparison
- **Position Comparison** - scatter plot of physical vs statistical fit per position group
- **Methodology** - grade computation explanation and Kubiak scheme profile data

## Notes on data sources

Most NFL data is sourced from the open-source `nfl_data_py` package. Offensive
line blocking grades are the exception: they come from **PFF Premium** (pass-block
grade, run-block grade, pass-block efficiency), manually exported per season to
`data/raw/pff_blocking_<season>.csv` (gitignored, proprietary) and ingested by
`ingest_pff_blocking`. Joined to players by name + team + season.

Data is sourced from the open-source nfl_data_py package, which reads from the nflverse data repositories. Note that nfl_data_py is officially deprecated in favor of nflreadpy, with no further maintenance planned. It is retained here because it works correctly against the project's pinned dependencies and because the historical 2024 and 2025 season data being pulled is static. If in-season 2026 data pulls break in the future, the migration path is nflreadpy, which returns Polars DataFrames and would require a conversion layer to pandas.

## Archetype methodology

The Kubiak position archetypes are built from three reference seasons of Klint
Kubiak as offensive coordinator, recency-weighted: the 2021 Minnesota Vikings
(his first OC year, weighted 15%), the 2024 New Orleans Saints (35%), and the
2025 Seattle Seahawks (50%). More recent seasons are weighted higher because
they reflect the offense Kubiak built with greater autonomy. The reference
team-seasons live in a single `REFERENCE` config in `src/archetype.py`; adding
another season is a one-line change.

FTN play-level charting (used for play-action, motion, RPO and out-of-pocket
features) only exists in nflverse from 2022 on, so the 2021 Vikings players
contribute to the physical and non-FTN performance features but abstain from the
FTN-derived ones, and the scheme-tendency profile is built from the 2024/2025
seasons only.

Methodology note: role-splitting (separate slot/perimeter WR archetypes, or
nearest-reference scoring instead of a single centroid) was tested against the
validation harness and degraded separation, because too few reference players
have complete records to define stable sub-archetypes. The single snap-weighted
centroid with a league-wide scaler is retained as the best-supported method.

Players contribute to the archetype if they recorded at least 20% offensive
snap share in at least 4 games of the reference season. Their contribution is
weighted by their total offensive snap count, so a 70% snap-share player has
a larger influence on the archetype than a 25% snap-share player.

Red zone is defined as yardline_100 <= 20. Several features are split between
red-zone and non-red-zone contexts because a coach's tendencies often change
when the field shrinks.

Offensive line features are individual PFF blocking grades (pass-block,
run-block, pass-block efficiency). Because these are pure quality scores rather
than scheme-specific traits, the OL grade is computed one-sided: a lineman who
blocks at or above the level of Kubiak's reference linemen scores at the top,
and only shortfalls below that level lower the grade. Rookies have no NFL
blocking grade and fall back to athletic measurables (wide confidence band).

## Scoring methodology

Physical Fit (primary grade)

Each player is scored on physical and athletic traits against the snap-weighted
mean of Kubiak's reference roster. Features per position are drawn from NFL
Combine measurables (forty, shuttle, cone, vertical, broad jump) and roster
data (height, weight). Feature lists vary by position to match the coverage
actually available in free data: skill positions drop shuttle and cone where
combine attendance was thin, tight ends drop forty for the same reason, and
offensive line keeps all features given their relatively complete records.

The physical grade isolates traits from performance and system experience.
A player with a high physical grade has the measurables that fit Kubiak's
scheme, regardless of what offense they've previously run. This answers the
project's core question: do the Raiders have the right players for this
offense?

Statistical Similarity (secondary grade)

The performance-based grade is preserved as secondary context. It measures
how a player's recent statistical profile compares to Kubiak's reference
players' profiles. Low scores often reflect that a player hasn't run a
compatible offense yet, not that they're a poor player. The grade is useful
for identifying players who have already demonstrated Kubiak-system
production, but it should not be used in isolation.

Combine coverage limitation: roughly 24% of reference players have no combine
record at all (largely undrafted free agents who skipped the combine). These
players contribute only height and weight to the physical archetype.
Coverage is weakest for offensive linemen, where shuttle and three-cone data
is incomplete.

Each current Raiders player is also scored against the Kubiak position archetype
for their position group. Performance features (for veterans) and combine
measurables (for rookies) are standardized against the spread of values
within Kubiak's reference player set, then a Euclidean distance to the
archetype is converted to a 0-100 grade.

Rookies are scored on combine measurables only. Sophomores are scored on a
blend of combine measurables and their rookie-season performance, with the
blend weighted by how much they played as a rookie. Veterans are scored on
performance only.

A confidence band accompanies each grade, widening when features are missing
or when the reference set has few players at that position. The overall
offensive grade is a weighted average across position groups (QB 30%, OL 25%,
WR 15%, RB 20%, TE 10%); these weights reflect general analytical consensus
on offensive importance in a wide-zone play-action system and are a designed
choice rather than a learned one. The 20% RB weight absorbs both rushing
contribution and pass-protection contribution, since individual pass-pro
grades are not available in free data.

Roughly 9% of pass and run plays from the reference seasons did not match
against FTN charting data and were excluded from the scheme profile.
