# Raiders Analysis 2026: Handoff #2 — Archetype Phase

## Purpose

Build Klint Kubiak's scheme profile and position-group archetypes from 2024 New Orleans Saints and 2025 Seattle Seahawks data. These artifacts become the reference points that current Raiders players are scored against in later handoffs.

This handoff does NOT build scoring logic or the Streamlit UI. Stop at the end of this document and wait for handoff #3.

## Prerequisites

The data layer from handoff #1 must be complete and verified. Specifically, `data/raw/nfl.duckdb` must contain populated tables for: snap_counts, weekly_stats, combine, ngs_passing, ngs_rushing, ngs_receiving, rosters. If any of these is missing or empty for 2024 or 2025, stop and resolve before continuing.

## Design principles to follow

Same principles as handoff #1. Specifically relevant for this phase:

- Each function has a single responsibility.
- Fail fast. If a data prerequisite is missing, raise a clear error naming the table and season. Do not silently continue with partial data.
- One correct path. No fallback logic if a join fails or a stat is missing; fix the root cause.
- Surgical changes. Do not modify the seven existing ingestion functions. Only ADD new ones.
- No emojis. No em dashes.
- Document limitations directly in the code as comments where relevant, and in the README data sources section.

## Scope of work

Three new responsibilities are added to the project:

1. Two new data ingestion functions (PBP and FTN), added to `src/data_ingestion.py`
2. The archetype building logic, implemented in `src/archetype.py` (currently an empty placeholder)
3. Two final verification queries to confirm the archetypes built correctly

## Phase A: Add PBP and FTN ingestion

### Step 1: Add the two ingestion functions to src/data_ingestion.py

Add these to the existing module. Keep the existing functions untouched. Both new functions follow the same pattern as the existing ones (write to DuckDB, raise on empty, raise on missing season).

For PBP, pull only the columns we actually need. The full PBP table has ~370 columns; we are deliberately keeping the DuckDB file small by filtering at ingestion.

```python
PBP_COLUMNS = [
    "play_id", "game_id", "season", "week", "posteam", "defteam",
    "yardline_100", "down", "ydstogo", "qtr", "score_differential",
    "play_type", "shotgun", "no_huddle", "qb_dropback", "pass", "rush",
    "run_location", "run_gap", "epa", "success",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
]

def ingest_pbp(years: list[int]) -> None:
    """Pull play-by-play data with filtered columns and write to DuckDB."""
    df = nfl.import_pbp_data(years, columns=PBP_COLUMNS)
    if df.empty:
        raise RuntimeError(f"pbp pull returned empty for years {years}")
    present = sorted(df["season"].unique().tolist())
    for y in years:
        if y not in present:
            raise RuntimeError(f"pbp missing season {y}; got {present}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS pbp")
    con.register("df", df)
    con.execute("CREATE TABLE pbp AS SELECT * FROM df")
    con.close()

def ingest_ftn(years: list[int]) -> None:
    """Pull FTN charting data and write to DuckDB."""
    df = nfl.import_ftn_data(years)
    if df.empty:
        raise RuntimeError(f"ftn pull returned empty for years {years}")
    present = sorted(df["season"].unique().tolist())
    for y in years:
        if y not in present:
            raise RuntimeError(f"ftn missing season {y}; got {present}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS ftn")
    con.register("df", df)
    con.execute("CREATE TABLE ftn AS SELECT * FROM df")
    con.close()
```

Also update the `main()` function in `data_ingestion.py` to call these two new functions for years `[2024, 2025]`. PBP and FTN for 2026 are not needed (the season has not happened yet).

### Step 2: Run the new ingestions

```
uv run python3 -c "from src.data_ingestion import ingest_pbp, ingest_ftn; ingest_pbp([2024, 2025]); ingest_ftn([2024, 2025])"
```

Expected: completes without errors. The PBP pull is the slowest yet (likely 2 to 5 minutes per season). FTN is faster.

### Step 3: Verify both new tables landed

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')
print('pbp:', con.execute('SELECT season, COUNT(*) FROM pbp GROUP BY season ORDER BY season').fetchall())
print('ftn:', con.execute('SELECT season, COUNT(*) FROM ftn GROUP BY season ORDER BY season').fetchall())
"
```

Expected: both tables have rows for both 2024 and 2025. PBP should be roughly 45,000 to 50,000 rows per season. FTN should be roughly 47,000 to 48,000 rows per season.

If row counts are zero for either season, stop and surface the error.

## Phase B: Build the archetype module

This is implemented in `src/archetype.py`. The placeholder file currently has only a docstring. Replace it with the full module described below.

The module has one job: read raw data from DuckDB, compute the scheme profile and position archetypes, write the results back to DuckDB as derived tables. It does not score players. It does not handle the current Raiders. Those are later handoffs.

### Step 4: Constants and structure for archetype.py

The module structure should be:

```python
"""Build Kubiak scheme profile and position archetypes from 2024 Saints and 2025 Seahawks."""

import duckdb
import pandas as pd

DB_PATH = "data/raw/nfl.duckdb"

# Reference teams, weighted per handoff design
REFERENCE = {
    "NO": {"season": 2024, "weight": 0.40},
    "SEA": {"season": 2025, "weight": 0.60},
}

# Inclusion floor: a player must have at least 20% offensive snap share
# in at least 4 games of the season to feed the archetype
MIN_SNAP_SHARE = 0.20
MIN_GAMES = 4

# Red zone definition
RZ_YARDLINE = 20  # yardline_100 <= 20 is red zone
```

### Step 5: Identify reference players

The first function reads snap_counts and identifies which players hit the inclusion floor for each reference team.

```python
def get_reference_players() -> pd.DataFrame:
    """Return players who met the snap floor on the reference Kubiak teams.

    Output columns: player_id, player, position, team, season, games_qualified,
                    avg_snap_share, total_offensive_snaps, weight
    """
    con = duckdb.connect(DB_PATH)
    query = """
        WITH per_game AS (
            SELECT
                pfr_player_id AS player_id,
                player,
                position,
                team,
                season,
                game_id,
                offense_pct AS snap_share,
                offense_snaps AS snaps
            FROM snap_counts
            WHERE (team = 'NO' AND season = 2024)
               OR (team = 'SEA' AND season = 2025)
        ),
        qualifying AS (
            SELECT
                player_id,
                player,
                position,
                team,
                season,
                COUNT(*) FILTER (WHERE snap_share >= ?) AS games_qualified,
                AVG(snap_share) AS avg_snap_share,
                SUM(snaps) AS total_offensive_snaps
            FROM per_game
            GROUP BY player_id, player, position, team, season
        )
        SELECT * FROM qualifying
        WHERE games_qualified >= ?
    """
    df = con.execute(query, [MIN_SNAP_SHARE, MIN_GAMES]).fetchdf()
    con.close()

    if df.empty:
        raise RuntimeError("No reference players passed the snap floor. Check snap_counts data.")

    df["weight"] = df["team"].map(lambda t: REFERENCE[t]["weight"])
    return df
```

Note: the column name `offense_pct` and `offense_snaps` are what `nfl_data_py.import_snap_counts` returns. If those exact column names differ in the table, the function will raise on the DuckDB query, and that error message will tell you what to fix. Do not write any fallback for missing columns; correct the column name in the SQL.

### Step 6: Build the scheme profile

This computes Kubiak's play-call tendencies, joining PBP and FTN at the play level for each reference team-season, with red zone splits.

```python
def build_scheme_profile() -> pd.DataFrame:
    """Build Kubiak's scheme profile from PBP + FTN, weighted 40/60.

    Output: one row, with columns for each metric, split by red_zone (1/0).
    Columns include: play_action_rate, motion_rate, shotgun_rate, no_huddle_rate,
                     screen_rate, rpo_rate, pass_rate, avg_epa, play_count
    """
    con = duckdb.connect(DB_PATH)
    query = """
        WITH plays AS (
            SELECT
                p.season,
                p.posteam AS team,
                p.play_id,
                p.epa,
                p.shotgun,
                p.no_huddle,
                p.pass,
                p.rush,
                CASE WHEN p.yardline_100 <= ? THEN 1 ELSE 0 END AS red_zone,
                f.is_play_action,
                f.is_motion,
                f.is_screen_pass,
                f.is_rpo
            FROM pbp p
            LEFT JOIN ftn f
                ON p.play_id = f.nflverse_play_id
                AND p.game_id = f.nflverse_game_id
            WHERE
                ((p.posteam = 'NO' AND p.season = 2024)
                 OR (p.posteam = 'SEA' AND p.season = 2025))
                AND p.play_type IN ('pass', 'run')
        )
        SELECT
            team,
            season,
            red_zone,
            COUNT(*) AS play_count,
            AVG(epa) AS avg_epa,
            AVG(CAST(pass AS DOUBLE)) AS pass_rate,
            AVG(CAST(shotgun AS DOUBLE)) AS shotgun_rate,
            AVG(CAST(no_huddle AS DOUBLE)) AS no_huddle_rate,
            AVG(CAST(is_play_action AS DOUBLE)) AS play_action_rate,
            AVG(CAST(is_motion AS DOUBLE)) AS motion_rate,
            AVG(CAST(is_screen_pass AS DOUBLE)) AS screen_rate,
            AVG(CAST(is_rpo AS DOUBLE)) AS rpo_rate
        FROM plays
        GROUP BY team, season, red_zone
    """
    raw = con.execute(query, [RZ_YARDLINE]).fetchdf()
    con.close()

    if raw.empty:
        raise RuntimeError("scheme profile query returned empty rows")

    # Apply 40/60 weighting per team
    raw["weight"] = raw["team"].map(lambda t: REFERENCE[t]["weight"])

    metric_cols = [
        "avg_epa", "pass_rate", "shotgun_rate", "no_huddle_rate",
        "play_action_rate", "motion_rate", "screen_rate", "rpo_rate"
    ]

    # Group by red_zone and weighted-average the metrics across the two teams
    rows = []
    for rz_value in [0, 1]:
        subset = raw[raw["red_zone"] == rz_value]
        if subset.empty:
            raise RuntimeError(f"scheme profile missing red_zone={rz_value}")
        row = {"red_zone": rz_value, "total_plays": int(subset["play_count"].sum())}
        for col in metric_cols:
            row[col] = (subset[col] * subset["weight"]).sum() / subset["weight"].sum()
        rows.append(row)

    return pd.DataFrame(rows)
```

### Step 7: Build position-group archetypes

This is the largest single function. It computes the position-group feature vectors from weekly_stats, NGS data, and snap_counts, weighted by snap volume and by the 40/60 team weighting.

The function operates per position group. For v1, build five groups: QB, RB, WR, TE, OL. The OL features are weaker due to no PFF; this is documented in comments and surfaced in the README.

Write a single function per position group rather than one mega-function. Each returns a one-row DataFrame with that position's archetype features. A coordinator function calls all five and stacks them.

```python
def _qb_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build QB archetype feature vector from reference players."""
    qb_refs = ref_players[ref_players["position"] == "QB"]
    if qb_refs.empty:
        raise RuntimeError("No QB reference players found")

    con = duckdb.connect(DB_PATH)
    player_ids = qb_refs["player_id"].tolist()

    # Pull QB performance stats from weekly_stats and ngs_passing
    # join on player_id/season; aggregate weighted by total snaps and team weight
    # Features: completion_pct, epa_per_dropback, time_to_throw, completed_air_yards,
    #           sack_rate, plus play_action_completion_pct, out_of_pocket_completion_pct
    # The play_action and out_of_pocket splits come from joining pbp+ftn filtered to this player

    # IMPLEMENTATION DETAIL: write the SQL to pull weekly_stats rows for these player_ids
    # in their reference season, then merge with ngs_passing on (player_id, season),
    # then join with the pbp+ftn play-level data filtered by passer_player_id.
    # Aggregate using weighted average: weight = total_offensive_snaps * team_weight.
    # Return a pd.Series with one entry per feature.

    con.close()
    # Return a pd.Series of features
    raise NotImplementedError("Implement using the pattern described above")
```

Important: I am giving you the pattern, not the full SQL for each position, because each position's stat columns differ. Implement each `_qb_archetype`, `_rb_archetype`, `_wr_archetype`, `_te_archetype`, `_ol_archetype` following the same shape:

1. Filter `ref_players` to the relevant position(s)
2. Pull performance stats from the right tables (weekly_stats for skill positions; ngs_passing for QB; ngs_rushing for RB; ngs_receiving for WR/TE; for OL, use PBP-derived team-level stats since we have no individual OL stats)
3. Compute red zone splits where the feature list calls for them (QB completion %, WR/TE/RB target share)
4. Aggregate using a weighted average: weight is `total_offensive_snaps * team_weight`
5. Return a `pd.Series` of features

The features list per position is exactly what was specified in the design conversation:

- **QB:** completion_pct, epa_per_dropback, sack_rate, time_to_throw, completed_air_yards, play_action_completion_pct, out_of_pocket_completion_pct, rz_completion_pct, rz_epa_per_dropback
- **RB:** rush_yards_over_expected, success_rate_outside_zone, shotgun_efficiency, rb_target_share, snap_share, rz_success_rate, rz_target_share
- **WR:** avg_separation, yac_over_expected, target_share, air_yards_share, motion_catch_rate, rz_target_share
- **TE:** target_share, air_yards_per_target, yac_over_expected, snap_share, rz_target_share
- **OL (team-level proxies, documented as such):** team_adjusted_line_yards_by_gap, team_pressure_rate_allowed, snap_share. OL stays a single archetype but stores `position_slot` (LT/LG/C/RG/RT) so we know which slot a Raiders OL is being scored against.

Where a feature requires a play-level join (red zone splits, play-action splits), do it with a SQL join inside the per-position function. Where it's a season-total stat, pull directly from `weekly_stats` or the NGS tables.

### Step 8: Coordinator function and writing results

```python
def build_position_archetypes() -> pd.DataFrame:
    """Build all five position-group archetypes and return as a DataFrame."""
    ref_players = get_reference_players()
    rows = [
        _qb_archetype(ref_players).rename("QB"),
        _rb_archetype(ref_players).rename("RB"),
        _wr_archetype(ref_players).rename("WR"),
        _te_archetype(ref_players).rename("TE"),
        _ol_archetype(ref_players).rename("OL"),
    ]
    df = pd.concat(rows, axis=1).T
    df.index.name = "position_group"
    return df.reset_index()

def write_archetypes() -> None:
    """Run all archetype builds and persist to DuckDB."""
    scheme = build_scheme_profile()
    archetypes = build_position_archetypes()
    ref_players = get_reference_players()

    con = duckdb.connect(DB_PATH)
    for table, df in [
        ("kubiak_scheme_profile", scheme),
        ("kubiak_position_archetypes", archetypes),
        ("kubiak_reference_players", ref_players),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.register("df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.close()

if __name__ == "__main__":
    write_archetypes()
    print("Archetypes built and persisted.")
```

### Step 9: Run the archetype build

```
uv run python3 src/archetype.py
```

Expected: prints "Archetypes built and persisted." and exits cleanly. If any function raises, stop and surface the error.

## Phase C: Final verification

### Step 10: Sanity-check the outputs

Run all three queries and paste the output:

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')

print('=== SCHEME PROFILE ===')
print(con.execute('SELECT * FROM kubiak_scheme_profile').fetchdf().to_string())

print()
print('=== POSITION ARCHETYPES ===')
print(con.execute('SELECT * FROM kubiak_position_archetypes').fetchdf().to_string())

print()
print('=== REFERENCE PLAYERS (count by team and position) ===')
print(con.execute('''
    SELECT team, season, position, COUNT(*) AS n_players
    FROM kubiak_reference_players
    GROUP BY team, season, position
    ORDER BY team, position
''').fetchdf().to_string())
"
```

What to look for in the output before declaring victory:

1. **Scheme profile**: two rows (red_zone = 0, red_zone = 1), all metric columns populated with reasonable values (rates between 0 and 1, EPA between roughly -0.5 and 0.5, play counts in the hundreds for red zone and thousands for non-red-zone).

2. **Position archetypes**: five rows (QB, RB, WR, TE, OL), all feature columns populated. Sanity-check: completion_pct should be between 0.55 and 0.75 for the QB row; pass_rate roughly 0.55 to 0.65 in the scheme profile non-RZ row.

3. **Reference players counts**: roughly 1 to 2 QBs per team, 2 to 4 RBs, 3 to 6 WRs, 2 to 3 TEs, 5 to 7 OL. If any position has 0 players for either team, the snap floor or position labeling needs investigation.

If anything looks off (zero counts, missing red zone row, NaN in features), stop and surface the specific issue. Do not paper over it.

### Step 11: Update the README

Add a new section under the existing "Notes on data sources" titled "Archetype methodology" with this text exactly:

```
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
```

## What NOT to do in this handoff

- Do not build the scoring function that compares current Raiders players to the archetype. That is handoff #3.
- Do not build any Streamlit UI. That is handoff #4.
- Do not add any features beyond the list in Step 7.
- Do not modify the seven existing ingestion functions.
- Do not include personnel grouping (11 personnel, 12 personnel, etc.). That data is not in free sources.
- Do not include block-and-release tracking. Same reason.
- Do not add fallback logic if a join produces nulls; raise.
- Do not commit until Step 10 output is verified clean.

## Final commit

After the verification in Step 10 looks correct:

```
git add .
git commit -m "Build Kubiak scheme profile and position archetypes from 2024 NO and 2025 SEA"
git push
```

Stop here. Wait for handoff #3.
