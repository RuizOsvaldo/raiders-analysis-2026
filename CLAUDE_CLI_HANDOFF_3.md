# Raiders Analysis 2026: Handoff #3 — Scoring Engine

## Purpose

Build the scoring engine that takes any current Raiders player, computes their feature vector in the same shape as the Kubiak archetypes, calculates similarity to the relevant position archetype, and converts the result to a 0-100 grade with a confidence band. Roll grades up to position-group grades and an overall offensive grade.

This handoff does NOT build the Streamlit UI. Stop at the end of this document and wait for handoff #4.

## Prerequisites

Handoff #2 must be complete. Specifically, `data/raw/nfl.duckdb` must contain populated `kubiak_scheme_profile`, `kubiak_position_archetypes`, and `kubiak_reference_players` tables. If any of these is missing or empty, stop and resolve before continuing.

## Design principles

Same as prior handoffs. Specifically critical for this phase:

- Each function has one responsibility.
- Fail fast. If a player has insufficient data to score, raise a clear error naming the player and what's missing. Do not silently substitute league average or NaN.
- One correct path. There is one way to compute similarity (Mahalanobis-style distance on standardized features), one way to convert to a grade (the same formula applied uniformly), one way to weight rookie vs veteran inputs.
- No fallbacks. If NGS data is missing for a player who should have it, raise.
- Surgical changes. This handoff modifies only `src/scoring.py` (currently an empty placeholder). Do not touch `data_ingestion.py` or `archetype.py`.
- No emojis. No em dashes.

## Scope of work

Three responsibilities, implemented in `src/scoring.py`:

1. Build a feature vector for any current Raiders player, applying the rookie/sophomore/veteran logic
2. Compute distance to the relevant position archetype, convert to a 0-100 grade
3. Aggregate to position-group grades and an overall offensive grade, with confidence bands

## Phase A: Player feature vector builder

### Step 1: Constants and module structure

```python
"""Score current Raiders players against the Kubiak archetypes."""

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/raw/nfl.duckdb"

# Years of NFL experience determine combine vs performance weighting
# (years counted as completed seasons before 2026)
WEIGHTING_RULES = {
    "rookie": {"combine": 1.00, "performance": 0.00},      # 0 prior seasons
    "sophomore_high_snap": {"combine": 0.30, "performance": 0.70},  # 1 prior season, >=40% snap share
    "sophomore_low_snap": {"combine": 0.70, "performance": 0.30},   # 1 prior season, <40% snap share
    "veteran": {"combine": 0.00, "performance": 1.00},     # 2+ prior seasons
}

# Snap-share threshold for sophomore classification
SOPHOMORE_SNAP_THRESHOLD = 0.40

# Feature lists per position. Must match the columns in kubiak_position_archetypes.
POSITION_FEATURES = {
    "QB": [
        "completion_pct", "epa_per_dropback", "sack_rate", "time_to_throw",
        "completed_air_yards", "play_action_completion_pct",
        "out_of_pocket_completion_pct", "rz_completion_pct", "rz_epa_per_dropback"
    ],
    "RB": [
        "rush_yards_over_expected", "success_rate_outside_zone",
        "shotgun_efficiency", "rb_target_share", "snap_share",
        "rz_success_rate", "rz_target_share"
    ],
    "WR": [
        "avg_separation", "yac_over_expected", "target_share",
        "air_yards_share", "motion_catch_rate", "rz_target_share"
    ],
    "TE": [
        "target_share", "air_yards_per_target", "yac_over_expected",
        "snap_share", "rz_target_share"
    ],
    "OL": [
        "rush_epa_guard", "rush_epa_tackle", "rush_epa_end",
        "team_pressure_rate_allowed", "avg_snap_share"
    ],
}

# Combine features (used for rookies and weighted for sophomores)
COMBINE_FEATURES_BY_POSITION = {
    "QB": ["ht", "wt", "hand_size", "arm_length"],
    "RB": ["ht", "wt", "forty", "shuttle", "three_cone", "broad_jump", "vertical"],
    "WR": ["ht", "wt", "forty", "shuttle", "three_cone", "broad_jump", "vertical"],
    "TE": ["ht", "wt", "forty", "shuttle", "three_cone", "broad_jump", "vertical"],
    "OL": ["ht", "wt", "shuttle", "three_cone", "broad_jump"],
}

# Map roster position labels to position groups
POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
}
```

The exact column names in `combine` from `nfl_data_py.import_combine_data` may differ slightly (e.g., `ht` might be `height`, `forty` might be `forty_yard`). When implementing, check the actual columns once and adjust `COMBINE_FEATURES_BY_POSITION` to match what the table has. Do not write fallback logic; just use the right column names.

### Step 2: Classify a player's experience bucket

```python
def classify_player(player_id: str) -> str:
    """Return 'rookie', 'sophomore_high_snap', 'sophomore_low_snap', or 'veteran'."""
    con = duckdb.connect(DB_PATH)
    # Count completed seasons before 2026 where the player appeared in snap_counts
    prior_seasons = con.execute("""
        SELECT DISTINCT season FROM snap_counts
        WHERE pfr_player_id = ? AND season < 2026
        ORDER BY season
    """, [player_id]).fetchdf()["season"].tolist()

    if len(prior_seasons) == 0:
        con.close()
        return "rookie"

    if len(prior_seasons) == 1:
        # Sophomore: check rookie-year snap share
        rookie_year = prior_seasons[0]
        snap_share = con.execute("""
            SELECT AVG(offense_pct) FROM snap_counts
            WHERE pfr_player_id = ? AND season = ?
        """, [player_id, rookie_year]).fetchone()[0]
        con.close()
        if snap_share is None:
            raise RuntimeError(f"Player {player_id} has 1 prior season but no snap_share. Investigate.")
        return "sophomore_high_snap" if snap_share >= SOPHOMORE_SNAP_THRESHOLD else "sophomore_low_snap"

    con.close()
    return "veteran"
```

### Step 3: Build a player's performance feature vector

```python
def get_performance_features(player_id: str, position_group: str) -> pd.Series:
    """Return the player's performance features in the same order as POSITION_FEATURES[group].

    Pulls from weekly_stats, NGS tables, pbp, and ftn as needed depending on the
    position group. Uses the most recent two seasons of data available for the player.

    Raises if the player has no usable performance data for this position group.
    """
    features = POSITION_FEATURES[position_group]

    # IMPLEMENTATION PATTERN, applied per position group:
    # 1. Pull the player's rows from the relevant source tables, last 2 seasons
    # 2. Aggregate to season-level or career-level metrics matching the archetype feature names
    # 3. Return as a pd.Series indexed by the feature names from POSITION_FEATURES[group]
    #
    # For QB: weekly_stats + ngs_passing + pbp+ftn join filtered by passer_player_id
    # For RB: weekly_stats + ngs_rushing + pbp filtered by rusher_player_id
    # For WR/TE: weekly_stats + ngs_receiving + pbp+ftn filtered by receiver_player_id
    # For OL: aggregate team-level stats during the games this player was on the field
    #         (lookup via snap_counts, then join with pbp run_gap and pressure metrics)

    raise NotImplementedError("Implement per position group following the pattern above")
```

This is intentionally a per-position-group implementation. Each position pulls from different tables. The function should call private helpers like `_qb_performance`, `_rb_performance`, etc., dispatching on `position_group`.

Where a feature can't be computed because the player has zero plays in that situation (e.g., a backup QB with no play-action attempts), raise a clear error: `RuntimeError(f"Player {player_id} ({position_group}) has no data for feature {feat}")`. Do not impute a default. The caller (the scoring function) will handle whether to skip the feature or grade only on available features.

### Step 4: Build a player's combine feature vector

```python
def get_combine_features(player_id: str, position_group: str) -> pd.Series:
    """Return the player's combine measurables in the order specified by
    COMBINE_FEATURES_BY_POSITION[group]. Raises if no combine record exists."""
    features = COMBINE_FEATURES_BY_POSITION[position_group]
    con = duckdb.connect(DB_PATH)
    row = con.execute(f"""
        SELECT {', '.join(features)}
        FROM combine
        WHERE pfr_id = ?
    """, [player_id]).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Player {player_id} has no combine record. Cannot score as rookie.")

    return pd.Series(dict(zip(features, row)))
```

Note: the join key on `combine` may not be `pfr_id`; check the actual column. The `nfl_data_py.import_combine_data` table uses `pfr_id` in many years but `cfb_player_id` or similar in others. Use whichever the actual table has and document it as a comment.

## Phase B: Distance to archetype and grade conversion

### Step 5: Build the archetype reference vectors as standardized scales

The position archetype is one row of feature values, but to convert distance to a meaningful grade we need a sense of the *spread* of those features across NFL players (so a distance of 0.5 standard deviations is interpretable). We use `kubiak_reference_players` to build that spread.

```python
def build_position_scaler(position_group: str) -> tuple[pd.Series, StandardScaler]:
    """Return the archetype mean vector and a fitted StandardScaler.

    The scaler is fit on the same reference player population that defined
    the archetype, so a player's standardized feature distance is in units
    of standard deviations within Kubiak's actual player set.
    """
    # Pull the reference players for this position group from kubiak_reference_players
    # For each reference player, get their performance features (the same way
    #   get_performance_features works)
    # Fit a StandardScaler on the resulting matrix
    # Return the archetype mean (from kubiak_position_archetypes) and the scaler
    raise NotImplementedError("Implement per the description above")
```

Important detail: the scaler is **fit on the reference player population, not on all NFL players**. This is intentional. The grade should reflect "how Kubiak-like" the player is, where the unit of measure is the natural variation within Kubiak's actual roster. If we used league-wide variance, the scale would be dominated by outliers irrelevant to scheme fit.

### Step 6: Compute the player's grade

```python
def score_player(player_id: str) -> dict:
    """Score a player against the Kubiak archetype for their position group.

    Returns:
        {
            "player_id": str,
            "position_group": str,
            "experience_bucket": str,
            "grade": float,                   # 0-100
            "confidence_interval": tuple[float, float],
            "feature_contributions": dict,    # feature -> signed contribution to distance
            "features_used": list[str],
            "features_missing": list[str],
        }
    """
    # 1. Determine position group from the player's current 2026 roster row
    # 2. Classify experience bucket (rookie / sophomore_* / veteran)
    # 3. Get the relevant feature vectors based on weights:
    #    - veteran: only performance features
    #    - rookie: only combine features
    #    - sophomore: both, blended by weight
    # 4. Standardize against the position scaler
    # 5. Compute distance to the archetype (Euclidean on standardized features)
    # 6. Convert distance to grade with the formula below
    # 7. Compute per-feature contributions for explainability
    # 8. Compute confidence interval based on number of features used and the
    #    population variance of grades within the reference set
    raise NotImplementedError("Implement per the description above")
```

#### Grade conversion formula

Distance is in standard-deviation units. Convert with:

```
grade = 100 * exp(-distance / scale_factor)
```

Where `scale_factor = 2.0` (chosen so a player at 2 sd from archetype gets ~37, and at 4 sd gets ~14). This is one tunable constant; document it and don't change it without good reason.

Bounds: grade is naturally in (0, 100]; clip the rare outlier above 100 (a player who's *exactly* the archetype) to 100.

#### Confidence interval

This is where the small sample size of the archetype gets surfaced honestly. Compute it as:

```
ci_width = base_uncertainty * sqrt(features_missing / features_total + 1/n_reference_players)
ci = (grade - 100*ci_width, grade + 100*ci_width)
```

Where `base_uncertainty = 0.15` (another tunable constant). Wider CI if features are missing or the reference set is small. This gives users a sense of which grades to trust.

#### Feature contributions

For each feature used, compute `(player_value - archetype_value) / archetype_std`. This is the signed standardized difference. Positive means "above archetype," negative means "below." The Streamlit UI will display these as a horizontal bar chart so the user can see exactly which features pulled the grade up or down.

## Phase C: Aggregate to unit and overall grades

### Step 7: Position-group grades

```python
def score_position_group(group: str, raiders_players: list[str]) -> dict:
    """Score all current Raiders players in a position group.

    Returns:
        {
            "position_group": str,
            "players": list[dict],            # each player's score_player() output
            "starter_grade": float,           # grade of the projected starter(s)
            "unit_grade": float,              # weighted across rotation
        }
    """
    # Score each player individually
    # Identify starter(s) by current snap-share rank or roster depth chart order
    # Compute unit grade as snap-share-weighted average across the rotation
    raise NotImplementedError("Implement per the description above")
```

For starter identification, use this rule: within a position group, the player with the highest projected snap share is the starter. For v1 with pre-season data, we don't have 2026 snap shares yet, so use roster depth chart ordering from the `rosters` table as the proxy (the `depth_chart_position` and `depth_team` columns if present). Document this as a v1 limitation in code comments.

### Step 8: Overall offensive grade

```python
def score_offense() -> dict:
    """Score the entire Raiders 2026 offense.

    Returns:
        {
            "overall_grade": float,
            "groups": dict,  # group -> score_position_group output
            "computed_at": str,
        }
    """
    # Load current LV Raiders offensive roster from rosters table (season=2026)
    # Group by position_group using POSITION_GROUP_MAP
    # Call score_position_group for each
    # Weight overall grade: QB 30%, OL 25%, WR 15%, RB 10%, TE 10%, RB pass-pro/blocking 10%
    # (these weights reflect Kubiak-system position importance; document in comments)
    raise NotImplementedError("Implement per the description above")
```

The overall weighting (QB 30%, OL 25%, WR 15%, RB 10%, TE 10%) is the one place we are not data-driven. These reflect general analytical consensus on offensive importance in a wide-zone / play-action system. Document this in code comments and in the README so users know it's a designed weight, not a learned one. This is the cleanest place for opinion in the model and surfacing it honestly is part of the value.

### Step 9: Persist results

```python
def write_grades() -> None:
    """Compute the full Raiders offense grade and persist to DuckDB."""
    result = score_offense()
    # Flatten to two tables:
    #   raiders_player_grades: one row per player with their grade and details
    #   raiders_offense_summary: one row with the overall grade and group grades
    # Write both to DuckDB.
    raise NotImplementedError("Implement per the description above")

if __name__ == "__main__":
    write_grades()
    print("Raiders offense scored and persisted.")
```

## Phase D: Verification

### Step 10: Run the scoring and inspect

```
uv run python3 src/scoring.py
```

Expected: prints "Raiders offense scored and persisted." and exits cleanly.

Then run the inspection:

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')

print('=== OFFENSE SUMMARY ===')
print(con.execute('SELECT * FROM raiders_offense_summary').fetchdf().to_string())

print()
print('=== TOP/BOTTOM PLAYERS BY GRADE ===')
print(con.execute('''
    SELECT player_id, position_group, experience_bucket, grade,
           features_used, features_missing
    FROM raiders_player_grades
    ORDER BY grade DESC
    LIMIT 5
''').fetchdf().to_string())

print()
print(con.execute('''
    SELECT player_id, position_group, experience_bucket, grade,
           features_used, features_missing
    FROM raiders_player_grades
    ORDER BY grade ASC
    LIMIT 5
''').fetchdf().to_string())

print()
print('=== GRADES BY POSITION GROUP ===')
print(con.execute('''
    SELECT position_group, COUNT(*) AS n_players,
           AVG(grade) AS avg_grade,
           MIN(grade) AS min_grade,
           MAX(grade) AS max_grade
    FROM raiders_player_grades
    GROUP BY position_group
''').fetchdf().to_string())
"
```

### What to look for in the output

1. **Overall grade**: a single number 0-100. Hard to predict exact value, but for a roster Kubiak personally helped assemble this offseason, expect 55-80 range. Anything outside 30-90 warrants a sanity check.

2. **Top/bottom players**: do the top-graded players make sense? Geno Smith, who Kubiak just brought over from Seattle, should grade very high (he IS the archetype). Brock Bowers should grade well. Newcomers who don't fit the scheme should grade lower. If a clearly elite player is at the bottom or a clearly mediocre player is at the top, something is wrong.

3. **Grades by position group**: averages should be in plausible ranges. If any group is averaging under 30 or above 90, there's likely an aggregation issue.

4. **Features used / missing**: most veterans should have most features. Many missing features for a veteran means a data join issue worth investigating.

Paste this output back. The Geno Smith sanity check in particular is the cleanest signal we have that the model is working, since he's both in the reference set (SEA 2025 starter) and on the current roster (LV 2026), so his grade should be near 100.

## Phase E: README update

Add a new section under "Archetype methodology" titled "Scoring methodology":

```
Each current Raiders player is scored against the Kubiak position archetype
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
WR 15%, RB 10%, TE 10%, RB blocking and unit cohesion 10%); these weights
reflect general analytical consensus on offensive importance in a wide-zone
play-action system and are a designed choice rather than a learned one.

Roughly 9% of pass and run plays from the reference seasons did not match
against FTN charting data and were excluded from the scheme profile.
```

## What NOT to do in this handoff

- Do not build any Streamlit UI. Handoff #4.
- Do not add new features beyond those in `POSITION_FEATURES`.
- Do not modify the archetype tables. If something feels wrong about an archetype value, raise it with the user, don't paper over it in scoring.
- Do not impute missing features. Raise.
- Do not add player swap logic. That's UI-layer work in handoff #4.
- Do not change the grade formula or weighting without explicit user agreement.

## Final commit

After the verification output in Step 10 looks correct (and especially after the Geno Smith sanity check passes):

```
git add .
git commit -m "Implement scoring engine: player grades, unit grades, overall offense grade"
git push
```

Stop here. Wait for handoff #4.
