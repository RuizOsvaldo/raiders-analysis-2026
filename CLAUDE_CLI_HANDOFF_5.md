# Raiders Analysis 2026: Handoff #5 — Streamlit UI

## Purpose

Build the user-facing Streamlit application that displays the Physical Fit grades, Statistical Similarity grades, and Scheme Experience tags for the 2026 Las Vegas Raiders offensive roster. This is the final deliverable for v1.

Also build the manually-curated Scheme Experience lookup table (a CSV) that feeds the third indicator in the UI.

This is the last handoff for v1. After this, the project ships.

## Prerequisites

Handoffs #1 through #4 must be complete. Specifically, `data/raw/nfl.duckdb` must contain:

- `raiders_player_grades` and `raiders_offense_summary` (statistical)
- `raiders_physical_player_grades` and `raiders_physical_offense_summary` (physical)
- Both grade tables must include the `coverage` and `raw_grade` columns added in the recent fix
- `rosters` with 2026 LV players (73 rows)

## Design principles

Same as prior handoffs. Specifically critical for the UI:

- Each Streamlit page has one responsibility (roster view, player detail, methodology). Do not build a single mega-page.
- Display coverage prominently next to every grade. Never show a grade without its coverage.
- Honest framing: Physical Fit is primary, Statistical Similarity is secondary with caveat, Scheme Experience is informational.
- Streamlit auto-runs on save during development; use `st.cache_data` for DuckDB reads so the app is responsive after first load.
- No emojis in any code or text. No em dashes.

## Scope of work

Three deliverables:

1. A manually-built CSV at `data/raw/scheme_experience.csv` mapping each 2026 Raider to yes/partial/no
2. A small loader function that joins the CSV with the player grades
3. The Streamlit app at `app/streamlit_app.py` with four pages

## Phase A: Scheme Experience lookup

### Step 1: Generate the player list to research

Run this to produce the list of current Raiders offensive players the lookup needs to cover:

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')
df = con.execute('''
    SELECT
        gsis_id AS player_id,
        full_name,
        position,
        years_exp,
        college
    FROM rosters
    WHERE season = 2026 AND team = 'LV'
      AND position IN ('QB','RB','FB','WR','TE','T','G','C','OT','OG','OL')
    ORDER BY position, full_name
''').fetchdf()
df.to_csv('data/raw/scheme_experience_to_fill.csv', index=False)
print(f'Wrote {len(df)} players to data/raw/scheme_experience_to_fill.csv')
print(df.head(10).to_string())
"
```

This creates a template CSV the user fills in.

### Step 2: User adds the scheme_experience column

The user (Osvaldo) edits the CSV to add a column `scheme_experience` with one of three values per player:

- **yes**: Player has run a Kubiak-tree wide-zone offense (49ers/Vikings/Rams/Saints/Seahawks under Shanahan, McVay, Kubiak, McDaniel, LaFleur, or direct Shanahan branches)
- **partial**: Player has run something with significant overlap (zone-blocking emphasis, heavy play-action, similar concepts)
- **no**: Different scheme entirely (pure pass-pro Air Raid, smash-mouth power running, RPO-heavy spread, etc.)

A starting reference list of clearly-yes coordinator lineages, for the user to apply:

- **Mike Shanahan tree**: Kyle Shanahan (SF), Sean McVay (LAR), Matt LaFleur (GB), Mike McDaniel (MIA), Klint Kubiak (NO 2024, SEA 2025), Bobby Slowik, Zac Robinson
- **Kubiak family**: Gary Kubiak (longtime HC/OC), Klint Kubiak
- **Notable partials**: any OC who came out of a wide-zone system but adapted to other concepts (e.g., some Andy Reid disciples, some recent Vikings under O'Connell who blended Shanahan with other ideas)

For each Raider, identify their primary offensive coordinator(s) from 2022-2025 and apply the rule above. If a player is a rookie, base it on their college offense (most college teams run zone concepts so most rookies will be "partial" by default, except those from pro-style pass-heavy programs).

This is judgment-based research, not data extraction. Plan for roughly 1 to 2 hours to do it carefully. Save the result back as `data/raw/scheme_experience.csv`.

### Step 3: Validate the CSV

```
uv run python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/scheme_experience.csv')
print(f'Rows: {len(df)}')
print(df['scheme_experience'].value_counts())
print()
missing = df[df['scheme_experience'].isna() | ~df['scheme_experience'].isin(['yes','partial','no'])]
if not missing.empty:
    print('Rows with invalid scheme_experience values:')
    print(missing.to_string())
    raise RuntimeError('Fix the CSV before proceeding')
"
```

If any rows have missing or invalid values, fix and re-validate before continuing.

### Step 4: Load the CSV into DuckDB

Add a small function to `src/roster.py` (currently an empty placeholder):

```python
"""Manage the current 2026 Raiders roster, including the Scheme Experience lookup."""

import duckdb
import pandas as pd

DB_PATH = "data/raw/nfl.duckdb"
SCHEME_CSV = "data/raw/scheme_experience.csv"

def load_scheme_experience() -> None:
    """Read the manually-curated CSV and persist as a DuckDB table."""
    df = pd.read_csv(SCHEME_CSV)

    valid = {"yes", "partial", "no"}
    if not df["scheme_experience"].isin(valid).all():
        raise RuntimeError(f"scheme_experience must be one of {valid}")

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS scheme_experience")
    con.register("df", df)
    con.execute("CREATE TABLE scheme_experience AS SELECT * FROM df")
    con.close()

if __name__ == "__main__":
    load_scheme_experience()
    print("Scheme experience loaded.")
```

Run it:

```
uv run python3 src/roster.py
```

## Phase B: Streamlit app structure

### Step 5: App entry point and shared utilities

Replace the placeholder `app/streamlit_app.py` with a multi-page setup. Streamlit handles multi-page apps via a `pages/` subdirectory. Structure:

```
app/
├── streamlit_app.py          # Main entry: roster overview
├── pages/
│   ├── 1_Player_Detail.py
│   ├── 2_Position_Comparison.py
│   └── 3_Methodology.py
└── utils.py                  # Shared DuckDB loaders
```

### Step 6: Shared utilities (app/utils.py)

```python
"""Shared helpers for the Streamlit app: DuckDB readers with caching."""

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = "data/raw/nfl.duckdb"

@st.cache_data(ttl=600)
def load_roster_grades() -> pd.DataFrame:
    """Return a single DataFrame joining physical, statistical, scheme experience, and roster."""
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT
            r.gsis_id AS player_id,
            r.full_name,
            r.position,
            r.jersey_number,
            r.years_exp,
            r.college,
            phys.position_group,
            phys.grade AS physical_grade,
            phys.raw_grade AS physical_raw,
            phys.coverage AS physical_coverage,
            phys.features_used AS physical_features_used,
            phys.features_missing AS physical_features_missing,
            stat.grade AS statistical_grade,
            stat.coverage AS statistical_coverage,
            stat.features_used AS statistical_features_used,
            stat.features_missing AS statistical_features_missing,
            stat.experience_bucket,
            COALESCE(sx.scheme_experience, 'unknown') AS scheme_experience
        FROM rosters r
        LEFT JOIN raiders_physical_player_grades phys ON phys.player_id = r.gsis_id
        LEFT JOIN raiders_player_grades stat ON stat.player_id = r.gsis_id
        LEFT JOIN scheme_experience sx ON sx.player_id = r.gsis_id
        WHERE r.season = 2026 AND r.team = 'LV'
          AND r.position IN ('QB','RB','FB','WR','TE','T','G','C','OT','OG','OL')
        ORDER BY phys.position_group, phys.grade DESC NULLS LAST
    """).fetchdf()
    con.close()
    return df

@st.cache_data(ttl=600)
def load_offense_summary() -> dict:
    """Return both offense summaries (physical and statistical) and group-level grades."""
    con = duckdb.connect(DB_PATH)
    physical = con.execute("SELECT * FROM raiders_physical_offense_summary").fetchdf()
    statistical = con.execute("SELECT * FROM raiders_offense_summary").fetchdf()
    con.close()
    return {"physical": physical, "statistical": statistical}

@st.cache_data(ttl=600)
def load_scheme_profile() -> pd.DataFrame:
    """Return Kubiak's scheme profile (play-call rates by RZ vs non-RZ)."""
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM kubiak_scheme_profile").fetchdf()
    con.close()
    return df

@st.cache_data(ttl=600)
def load_archetypes() -> dict:
    """Return both archetype tables (physical and performance)."""
    con = duckdb.connect(DB_PATH)
    return {
        "physical": con.execute("SELECT * FROM kubiak_physical_archetypes").fetchdf(),
        "performance": con.execute("SELECT * FROM kubiak_position_archetypes").fetchdf(),
    }

def grade_color(grade: float, coverage: float) -> str:
    """Return a color hex for a grade, dimmed when coverage is low."""
    if pd.isna(grade):
        return "#888888"
    # Solid colors at full coverage, faded at low coverage
    base_alpha = max(0.4, coverage if not pd.isna(coverage) else 1.0)
    if grade >= 65:
        return f"rgba(46, 139, 87, {base_alpha})"  # green
    elif grade >= 45:
        return f"rgba(218, 165, 32, {base_alpha})"  # gold
    else:
        return f"rgba(178, 34, 34, {base_alpha})"  # red

def coverage_label(coverage: float) -> str:
    """Human-readable coverage label."""
    if pd.isna(coverage):
        return "unknown"
    if coverage >= 0.99:
        return "full data"
    if coverage >= 0.6:
        return f"{int(coverage*100)}% data"
    return f"sparse data ({int(coverage*100)}%)"
```

### Step 7: Main page (app/streamlit_app.py)

```python
"""Raiders Analysis 2026: main roster overview page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_roster_grades, load_offense_summary, grade_color, coverage_label

st.set_page_config(
    page_title="Raiders Analysis 2026",
    page_icon=None,
    layout="wide",
)

st.title("Raiders 2026 Scheme Fit Analysis")
st.caption(
    "Grading the 2026 Las Vegas Raiders offensive roster against Klint Kubiak's "
    "scheme tendencies from 2024 New Orleans and 2025 Seattle."
)

# Top-level summary
summary = load_offense_summary()
phys_summary = summary["physical"]
stat_summary = summary["statistical"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Physical Fit (overall)", f"{phys_summary['overall_grade'].iloc[0]:.1f}")
    st.caption("Primary grade. Trait-based, scheme-agnostic.")
with col2:
    st.metric("Statistical Similarity (overall)", f"{stat_summary['overall_grade'].iloc[0]:.1f}")
    st.caption("Secondary. Performance similarity to Kubiak's prior players.")
with col3:
    st.metric("Reference seasons", "2 (Saints 2024, Seahawks 2025)")
    st.caption("Weighted 40 / 60.")

st.markdown("---")

# Group-level grades
st.subheader("By position group")
groups = ["QB", "RB", "WR", "TE", "OL"]
group_cols = st.columns(len(groups))
for i, g in enumerate(groups):
    with group_cols[i]:
        phys_val = phys_summary[g].iloc[0] if g in phys_summary.columns else None
        stat_val = stat_summary[g].iloc[0] if g in stat_summary.columns else None
        st.markdown(f"**{g}**")
        if phys_val is not None:
            st.markdown(f"Physical: **{phys_val:.1f}**")
        if stat_val is not None:
            st.markdown(f"Stat: {stat_val:.1f}")

st.markdown("---")

# Roster table
st.subheader("Full roster")
df = load_roster_grades()

# Filter widgets
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    position_filter = st.multiselect(
        "Position group",
        options=sorted(df["position_group"].dropna().unique()),
        default=sorted(df["position_group"].dropna().unique()),
    )
with fcol2:
    scheme_filter = st.multiselect(
        "Scheme experience",
        options=["yes", "partial", "no", "unknown"],
        default=["yes", "partial", "no", "unknown"],
    )
with fcol3:
    min_coverage = st.slider(
        "Minimum coverage",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="Hide grades built on sparse data. 1.0 = only show players with full feature coverage."
    )

# Apply filters
filtered = df[
    df["position_group"].isin(position_filter)
    & df["scheme_experience"].isin(scheme_filter)
    & ((df["physical_coverage"].fillna(0) >= min_coverage)
       | (df["statistical_coverage"].fillna(0) >= min_coverage))
].copy()

# Display
display_df = filtered[[
    "full_name", "position", "position_group", "physical_grade", "physical_coverage",
    "statistical_grade", "statistical_coverage", "scheme_experience", "experience_bucket"
]].copy()
display_df.columns = [
    "Player", "Pos", "Group", "Physical", "Phys Coverage",
    "Statistical", "Stat Coverage", "Scheme Exp", "NFL Exp"
]
display_df["Physical"] = display_df["Physical"].round(1)
display_df["Statistical"] = display_df["Statistical"].round(1)
display_df["Phys Coverage"] = display_df["Phys Coverage"].apply(lambda c: coverage_label(c))
display_df["Stat Coverage"] = display_df["Stat Coverage"].apply(lambda c: coverage_label(c))

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    "Click into individual players in the Player Detail page for feature-level breakdowns. "
    "See the Methodology page for how grades are computed."
)
```

### Step 8: Player Detail page (app/pages/1_Player_Detail.py)

```python
"""Per-player detail: feature contributions, comparison to archetype, both grades."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_roster_grades, load_archetypes, coverage_label

st.title("Player Detail")

df = load_roster_grades()

# Player selector
player_name = st.selectbox("Select player", options=sorted(df["full_name"].dropna().unique()))
player = df[df["full_name"] == player_name].iloc[0]

# Header card
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.subheader(f"{player['full_name']}")
    st.caption(f"#{player['jersey_number']} | {player['position']} | {player['college']} | {player['years_exp']} yrs exp")
with c2:
    st.metric("Physical Fit", f"{player['physical_grade']:.1f}" if pd.notna(player['physical_grade']) else "N/A")
    st.caption(coverage_label(player['physical_coverage']))
with c3:
    st.metric("Statistical", f"{player['statistical_grade']:.1f}" if pd.notna(player['statistical_grade']) else "N/A")
    st.caption(coverage_label(player['statistical_coverage']))

# Scheme experience badge
exp_color = {"yes": "green", "partial": "orange", "no": "red", "unknown": "gray"}.get(
    player['scheme_experience'], "gray"
)
st.markdown(
    f"**Scheme experience**: :{exp_color}[{player['scheme_experience'].upper()}]"
)

st.markdown("---")

# Feature contributions chart
st.subheader("How each feature contributed to the grade")
st.caption(
    "Positive bars: the player exceeds Kubiak's archetype on this feature. "
    "Negative: below archetype. Bigger absolute value = bigger impact on grade."
)

# IMPLEMENTATION: read feature_contributions from raiders_physical_player_grades for this player
# Render as a horizontal bar chart with Plotly
# Default to Physical; add a toggle for Statistical if desired

# Comparison to archetype
st.subheader("vs Kubiak archetype")
archetypes = load_archetypes()
phys_archetype = archetypes["physical"][archetypes["physical"]["position_group"] == player["position_group"]]
if not phys_archetype.empty:
    # IMPLEMENTATION: side-by-side table of player feature values vs archetype values
    st.dataframe(phys_archetype.iloc[0])
```

The feature_contributions and detailed comparisons require pulling per-player data that may need to be added to the scoring tables. If feature contributions are stored as JSON strings in the grades table, parse them. If they're stored as separate columns, read directly. Verify the actual table shape and implement accordingly.

### Step 9: Position Comparison page (app/pages/2_Position_Comparison.py)

```python
"""Compare players within a position group. Optional: swap players to see roster impact."""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_roster_grades, load_archetypes

st.title("Position Group Comparison")

df = load_roster_grades()

group = st.selectbox("Position group", options=sorted(df["position_group"].dropna().unique()))
group_df = df[df["position_group"] == group].sort_values("physical_grade", ascending=False)

st.subheader(f"{group} depth chart")

# Two-axis scatter: physical grade vs statistical grade
fig = px.scatter(
    group_df,
    x="statistical_grade",
    y="physical_grade",
    text="full_name",
    color="scheme_experience",
    size="physical_coverage",
    hover_data=["position", "experience_bucket"],
    color_discrete_map={"yes": "#2E8B57", "partial": "#DAA520", "no": "#B22222", "unknown": "#888888"},
    range_x=[0, 100],
    range_y=[0, 100],
)
fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis_title="Statistical Similarity",
    yaxis_title="Physical Fit",
    height=600,
)
# Quadrant annotations
fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Top-right: high physical AND statistical fit. The most credible scheme fits. "
    "Top-left: physically fits but no system production yet. Upside bets. "
    "Bottom-right: produced in similar systems but doesn't physically match. "
    "Bottom-left: neither traits nor production match. Largest concerns."
)

# Table view
st.subheader("Players in group")
st.dataframe(
    group_df[["full_name", "position", "physical_grade", "physical_coverage",
              "statistical_grade", "statistical_coverage", "scheme_experience"]],
    use_container_width=True,
    hide_index=True,
)
```

### Step 10: Methodology page (app/pages/3_Methodology.py)

```python
"""Explain how the grades are computed and what they mean."""

import streamlit as st
from utils import load_scheme_profile

st.title("Methodology")

st.markdown("""
## What this project measures

This app grades the 2026 Las Vegas Raiders offensive roster against Klint
Kubiak's scheme tendencies from his two most recent OC seasons: the 2024
New Orleans Saints (weighted 40%) and the 2025 Seattle Seahawks (weighted 60%).

The newer season is weighted higher because it reflects the offense Kubiak
built with the most recent autonomy.

## Two grades, one Scheme Experience tag

**Physical Fit (primary)**

Trait-based. Compares a player's height, weight, and combine measurables
(forty, shuttle, three-cone, vertical, broad jump) against the snap-weighted
mean of those same metrics across Kubiak's reference players at the position.
Distance to the archetype is converted to a 0 to 100 grade, then multiplied
by the fraction of features the player has data for. A player with only
height and weight on file gets penalized to roughly 40% of their raw grade.

This grade is scheme-agnostic. It does not care what offense the player has
run before. It only asks: do they have the physical and athletic traits
Kubiak's scheme has historically rewarded.

**Statistical Similarity (secondary)**

Performance-based. Compares a player's recent NFL statistics (last two
seasons) against Kubiak's reference players' statistics. Same distance
calculation, same grade formula.

This grade conflates three things: physical traits, system experience, and
recent performance. Low Statistical Similarity often means a player has not
run a compatible scheme yet, not that they cannot perform in one. Read it
together with the Physical Fit grade, not in isolation.

**Scheme Experience (informational)**

A manually-curated tag indicating whether the player has prior NFL experience
in a Kubiak-tree offense:
- yes: has run a wide-zone Shanahan-tree system under one of the recognized lineages
- partial: has run something with significant overlap
- no: different scheme entirely
- unknown: not yet researched

## Data sources

All data is from the open-source nflverse project, accessed via the
nfl_data_py and nflreadpy Python packages. Specifically:

- Snap counts, NextGen Stats, rosters, play-by-play, FTN charting, combine: nfl_data_py
- Weekly player stats: nflreadpy (after nfl_data_py's 2025 file paths returned 404)

## Known limitations

- Personnel groupings (11 / 12 / 21 personnel) are paywalled (PFF) and not used here
- Individual OL grades require paid charting; this project uses team-level proxies
- Roughly 9% of plays did not match between PBP and FTN charting and were excluded
- 24% of reference players have no combine record; these contribute only height and weight
- Combine coverage is weakest for offensive linemen
- Reference set is only 2 seasons; archetypes are noisier than they would be with 4 to 5 years of data

## Kubiak's actual play-calling tendencies
""")

scheme = load_scheme_profile()
st.dataframe(scheme, use_container_width=True, hide_index=True)
st.caption(
    "Play-call rates from the weighted Saints 2024 / Seahawks 2025 reference set, "
    "split between red zone (yardline_100 <= 20) and the rest of the field."
)
```

## Phase C: Run the app

### Step 11: Test locally

```
cd app
uv run streamlit run streamlit_app.py
```

A browser tab opens automatically at `http://localhost:8501`. Verify:

- Main page loads, shows the offense summary and the filterable roster table
- Player Detail page loads, lets you select a player, shows both grades
- Position Comparison page loads, scatter plot renders, quadrants are interpretable
- Methodology page loads, scheme profile table renders

If any page errors out, the error message will tell you what's wrong; common issues are missing columns in the joined data or NaN handling.

### Step 12: Deployment (optional, only if user wants public access)

Streamlit Community Cloud deploys for free from a GitHub repo:

1. Push the project to GitHub if not already there
2. Visit https://share.streamlit.io and connect the GitHub repo
3. Point it at `app/streamlit_app.py` as the entry file
4. Add a `requirements.txt` at the project root (or use the uv-generated `pyproject.toml`)

The DuckDB file is around 100-200 MB after all the data is loaded; Streamlit Cloud has limits. If the file exceeds limits, options:

- Pre-compute the final grade tables and ship only those (small)
- Strip the raw `pbp` and `ftn` tables from the DuckDB file before deploying (they're only needed for archetype building, not for runtime queries)

Skip deployment if the user wants to keep the project local only.

## Phase D: README final update

Add a usage section to the README:

```
## Running the app

After running the data ingestion and scoring pipelines:

    uv run python3 src/data_ingestion.py
    uv run python3 src/archetype.py
    uv run python3 src/scoring.py
    uv run python3 src/roster.py

Launch the Streamlit app:

    cd app
    uv run streamlit run streamlit_app.py

The app opens at http://localhost:8501.
```

## What NOT to do in this handoff

- Do not modify the scoring math or feature lists.
- Do not add new tables to DuckDB beyond the scheme_experience load.
- Do not skip the coverage display in the UI. It is the most important user-facing safety rail.
- Do not display the statistical grade more prominently than the physical grade. Physical is primary.
- Do not deploy to Streamlit Cloud without the user explicitly asking.

## Final commit

After Step 11 looks correct end to end:

```
git add .
git commit -m "Add Streamlit UI: roster overview, player detail, position comparison, methodology"
git push
```

The project is shippable at this point.

## After this handoff

Come back to me. We have one piece of work left that's not in any handoff: drafting the public post about building this project with free data resources, naming the specific features that required paid services, and asking the community for free alternatives. We agreed back during handoff design to write that together at the end.
