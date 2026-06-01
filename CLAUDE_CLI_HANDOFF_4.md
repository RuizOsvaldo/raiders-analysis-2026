# Raiders Analysis 2026: Handoff #4 — Physical Fit Model

## Purpose

Build a second scoring model that grades current Raiders players on physical and athletic traits against Kubiak's reference roster. This becomes the project's primary grade, with the existing Statistical Similarity grade (built in handoffs #2 and #3) becoming secondary context.

The reason for the split: the existing performance-based model conflates three things (physical traits, system experience, recent performance). Physical Fit isolates traits, which directly answers "does this player have what Kubiak's scheme needs."

This handoff does NOT build the Streamlit UI or the Scheme Experience lookup. Those are handoff #5. Stop at the end of this document and wait.

## Prerequisites

Handoffs #1, #2, and #3 must be complete. Specifically:

- The `combine` table in `data/raw/nfl.duckdb` must contain rows for 2010-2025 (5,391 rows). If you have only 2024-2025, re-run `ingest_combine(list(range(2010, 2026)))` first.
- The `kubiak_reference_players` table must exist and have 46 rows.
- The `rosters` table must contain 2026 LV Raiders data (73 rows).

## Design principles

Same as prior handoffs. Specifically critical for this phase:

- Each function has one responsibility.
- Fail fast. If a player has neither combine data nor roster physical data, raise. Do not silently substitute league average.
- One correct path. The grade formula is the same as the Statistical Similarity model so the two are comparable on the same scale.
- Surgical changes. This handoff adds to `src/archetype.py` and `src/scoring.py`. It does NOT modify the existing performance-based functions in either file.
- No emojis. No em dashes.
- Reuse infrastructure where possible. The reference player identification, snap-weighted aggregation, and grade conversion formula already exist. Use them.

## Scope of work

Three responsibilities, split across two existing modules:

1. In `src/archetype.py`: add a function that builds the physical archetype from combine + roster data, persisted as a new table
2. In `src/scoring.py`: add a function that scores any player on physical traits against the physical archetype
3. A coordinator function that scores the full Raiders roster on physical fit and persists results

## Phase A: Physical archetype construction

### Step 1: Add constants to src/archetype.py

Add this block alongside the existing constants. Do not modify the existing performance feature constants.

```python
# Physical Fit feature lists per position, tuned to actual combine coverage.
# Features dropped where coverage is below ~50% for the reference set.
PHYSICAL_FEATURES = {
    "QB": ["ht", "wt", "forty", "shuttle", "cone", "vertical", "broad_jump"],
    "RB": ["ht", "wt", "forty", "vertical", "broad_jump"],
    "WR": ["ht", "wt", "forty", "vertical", "broad_jump"],
    "TE": ["ht", "wt", "vertical", "broad_jump"],
    "OL": ["ht", "wt", "forty", "shuttle", "cone", "vertical", "broad_jump"],
}

# Map roster/position labels to position groups (same map as in scoring)
PHYSICAL_POSITION_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
}
```

The exact column names in `combine` are: `ht`, `wt`, `forty`, `shuttle`, `cone`, `vertical`, `broad_jump`. These were verified against the actual table schema. Do not use `three_cone`, `height`, or `weight`. If a column name fails the SQL, the error will name it; fix the constant rather than adding fallback logic.

### Step 2: Build the physical archetype function

Add this function to `src/archetype.py`. It mirrors the existing position archetype builder in structure but pulls combine and roster physical data instead of performance stats.

```python
def build_physical_archetypes() -> pd.DataFrame:
    """Build physical-fit archetypes per position group from combine + roster data.

    Returns one row per position group with the snap-weighted mean of each
    physical feature, computed across the reference player set.

    Coverage limitations:
    - Players without combine records contribute only ht and wt from the rosters table
    - Players with partial combines (e.g., missing shuttle) contribute to the
      features they have. The weighted mean is computed per-feature.
    """
    ref_players = get_reference_players()
    if ref_players.empty:
        raise RuntimeError("No reference players found. Run get_reference_players first.")

    con = duckdb.connect(DB_PATH)
    rows = []

    for group, features in PHYSICAL_FEATURES.items():
        # Find reference players whose position maps to this group
        members = ref_players[
            ref_players["position"].map(PHYSICAL_POSITION_MAP).fillna("") == group
        ].copy()

        if members.empty:
            raise RuntimeError(f"No reference players found for group {group}")

        # For each member, pull their physical data
        # ht and wt come from rosters (most recent season we have for them)
        # forty/shuttle/cone/vertical/broad_jump come from combine
        member_ids = members["player_id"].tolist()
        placeholders = ",".join(["?"] * len(member_ids))

        physical_query = f"""
            SELECT
                r.gsis_id AS player_id,
                AVG(r.height) AS ht,
                AVG(r.weight) AS wt,
                MAX(c.forty) AS forty,
                MAX(c.shuttle) AS shuttle,
                MAX(c.cone) AS cone,
                MAX(c.vertical) AS vertical,
                MAX(c.broad_jump) AS broad_jump
            FROM rosters r
            LEFT JOIN combine c ON r.pfr_id = c.pfr_id
            WHERE r.gsis_id IN ({placeholders})
            GROUP BY r.gsis_id
        """
        # Note: roster join key is gsis_id but kubiak_reference_players uses
        # pfr_player_id from snap_counts. Verify that column mapping before running.
        # If the join keys differ, build a small id-crosswalk via the rosters table.

        physical = con.execute(physical_query, member_ids).fetchdf()

        if physical.empty:
            raise RuntimeError(f"No physical data found for {group} reference players")

        # Merge in the snap-weighting from members (snap count * team weight)
        merged = members.merge(physical, on="player_id", how="inner")
        merged["agg_weight"] = merged["total_offensive_snaps"] * merged["weight"]

        # Weighted mean per feature, ignoring NaNs
        row = {"position_group": group}
        for feat in features:
            valid = merged[merged[feat].notna()]
            if valid.empty:
                raise RuntimeError(
                    f"No {group} reference players have feature {feat}. "
                    f"Coverage diagnostic suggested {feat} should exist for this group."
                )
            row[feat] = (valid[feat] * valid["agg_weight"]).sum() / valid["agg_weight"].sum()
            row[f"{feat}_n_players"] = len(valid)
        rows.append(row)

    con.close()
    return pd.DataFrame(rows)
```

A few notes the implementation must handle:

1. **Join key crosswalk.** `kubiak_reference_players.player_id` comes from `snap_counts.pfr_player_id`. The `combine` table joins on `pfr_id`. The `rosters` table uses `gsis_id` for player identity, but also carries `pfr_id`. Build the join through `rosters` if direct id matching fails. Verify which id is in `kubiak_reference_players` before writing the query; the column is called `player_id` but the underlying value type matters.

2. **Multiple roster rows per player.** A player can appear in `rosters` across multiple seasons. The query above uses `AVG` for ht/wt to collapse, which is fine because those don't change much year to year.

3. **Combine row uniqueness.** Some players have multiple combine records (rare, but possible if they re-tested). `MAX` picks the best recorded number. This is a deliberate choice: we want the player's measured peak athletic value, not an average that includes off-day re-tests.

### Step 3: Add to the write_archetypes coordinator

Modify the existing `write_archetypes` function to also write the physical archetype table. This is one of the rare cases where we touch existing code; the change is purely additive.

```python
def write_archetypes() -> None:
    """Run all archetype builds and persist to DuckDB."""
    scheme = build_scheme_profile()
    archetypes = build_position_archetypes()
    physical = build_physical_archetypes()  # NEW
    ref_players = get_reference_players()

    con = duckdb.connect(DB_PATH)
    for table, df in [
        ("kubiak_scheme_profile", scheme),
        ("kubiak_position_archetypes", archetypes),
        ("kubiak_physical_archetypes", physical),  # NEW
        ("kubiak_reference_players", ref_players),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.register("df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.close()
```

### Step 4: Run the archetype build

```
uv run python3 src/archetype.py
```

Expected: prints "Archetypes built and persisted." and exits cleanly.

### Step 5: Inspect the physical archetypes

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')
print(con.execute('SELECT * FROM kubiak_physical_archetypes').fetchdf().to_string())
"
```

What to sanity-check in the output:

- **QB row:** ht should be roughly 74-76 (6'2"-6'4"), wt 215-235, forty around 4.7-4.9
- **RB row:** ht around 70-72 (5'10"-6'0"), wt 210-225, forty 4.45-4.55
- **WR row:** ht 71-74, wt 195-215, forty 4.45-4.55
- **TE row:** ht 76-78 (6'4"-6'6"), wt 245-260
- **OL row:** ht 76-78, wt 305-320, forty 5.1-5.4

If any number is wildly outside these ranges (e.g., RB averaging 6'4", forty under 4.3), there's a join issue or a unit problem. Stop and surface.

The `*_n_players` columns tell you how many reference players contributed to each feature. If any is 0 or 1, that feature is too thin to trust and should be removed from `PHYSICAL_FEATURES`. Report it and we'll adjust.

## Phase B: Physical scoring

### Step 6: Add physical scoring constants to src/scoring.py

```python
# Import the physical feature lists from archetype module to avoid duplication
from src.archetype import PHYSICAL_FEATURES, PHYSICAL_POSITION_MAP
```

### Step 7: Function to get a player's physical features

```python
def get_physical_features(player_id: str, position_group: str) -> pd.Series:
    """Return a player's physical features in PHYSICAL_FEATURES[group] order.

    Pulls ht/wt from rosters (most recent), combine measurables from combine.
    Returns NaN for features the player doesn't have; the scoring function
    decides whether to skip them or fail.
    """
    features = PHYSICAL_FEATURES[position_group]
    con = duckdb.connect(DB_PATH)

    row = con.execute("""
        SELECT
            (SELECT AVG(height) FROM rosters WHERE gsis_id = ?) AS ht,
            (SELECT AVG(weight) FROM rosters WHERE gsis_id = ?) AS wt,
            c.forty,
            c.shuttle,
            c.cone,
            c.vertical,
            c.broad_jump
        FROM (SELECT ? AS gsis_id) p
        LEFT JOIN rosters r ON r.gsis_id = p.gsis_id
        LEFT JOIN combine c ON c.pfr_id = r.pfr_id
        LIMIT 1
    """, [player_id, player_id, player_id]).fetchone()
    con.close()

    if row is None or all(v is None for v in row):
        raise RuntimeError(f"Player {player_id} has no physical data in rosters or combine")

    all_features = {
        "ht": row[0], "wt": row[1], "forty": row[2], "shuttle": row[3],
        "cone": row[4], "vertical": row[5], "broad_jump": row[6]
    }
    return pd.Series({f: all_features[f] for f in features})
```

The triple-`player_id` in the query parameters is intentional; the subqueries and the join all need it. Slightly awkward but explicit.

### Step 8: Build the physical scaler

The scaler is fit on the reference player population's physical features, so the standardized distance is in units of "standard deviations among Kubiak's actual roster."

```python
def build_physical_scaler(position_group: str) -> tuple[pd.Series, StandardScaler]:
    """Return the physical archetype mean vector and a fitted StandardScaler.

    Scaler is fit on reference players' physical features for the group.
    """
    con = duckdb.connect(DB_PATH)

    archetype = con.execute(f"""
        SELECT * FROM kubiak_physical_archetypes WHERE position_group = ?
    """, [position_group]).fetchdf().iloc[0]

    # Get the reference player population for this group, with their physical features
    ref_ids = con.execute("""
        SELECT player_id FROM kubiak_reference_players
        WHERE position IN (
            SELECT position FROM kubiak_reference_players
            WHERE position = ANY(?)
        )
    """, [list(PHYSICAL_POSITION_MAP.keys())]).fetchdf()["player_id"].tolist()

    # Filter to just this group
    ref_for_group = [
        pid for pid in ref_ids
        if PHYSICAL_POSITION_MAP.get(
            con.execute("SELECT position FROM kubiak_reference_players WHERE player_id = ?", [pid]).fetchone()[0]
        ) == position_group
    ]

    con.close()

    feature_matrix = []
    for pid in ref_for_group:
        try:
            vec = get_physical_features(pid, position_group)
            if vec.isna().any():
                continue  # skip incomplete records for the scaler
            feature_matrix.append(vec.values)
        except RuntimeError:
            continue

    if len(feature_matrix) < 3:
        raise RuntimeError(
            f"Only {len(feature_matrix)} complete physical records for {position_group}. "
            f"Need at least 3 to fit a scaler."
        )

    scaler = StandardScaler()
    scaler.fit(feature_matrix)

    features = PHYSICAL_FEATURES[position_group]
    archetype_vec = archetype[features]
    return archetype_vec, scaler
```

This builds the scaler from "complete" reference players only (no missing features). Incomplete players still contribute to the archetype mean via the weighted-mean approach in Step 2, but only complete records define the variance.

### Step 9: Score a player on physical fit

```python
def score_player_physical(player_id: str, position_group: str = None) -> dict:
    """Score a player's physical fit against the Kubiak physical archetype.

    Mirrors score_player() in shape and return structure so the UI can
    consume both grades the same way.

    Returns:
        {
            "player_id": str,
            "position_group": str,
            "grade": float,                   # 0-100
            "confidence_interval": tuple[float, float],
            "feature_contributions": dict,
            "features_used": list[str],
            "features_missing": list[str],
            "model": "physical",
        }
    """
    # 1. Resolve position group if not provided (look up from most recent roster row)
    # 2. Pull player's physical features via get_physical_features
    # 3. Pull archetype vector and scaler via build_physical_scaler
    # 4. Identify which features the player has vs is missing
    # 5. For features the player has, compute standardized distance
    # 6. Apply same grade formula: grade = 100 * exp(-distance / 2.0)
    # 7. Wider CI if features are missing
    # 8. Per-feature contributions = (player_value - archetype_value) / archetype_std
    raise NotImplementedError("Implement following the pattern of the existing score_player")
```

Use the exact same grade formula and CI calculation as the statistical model. They must be on the same 0-100 scale or the UI comparison is meaningless.

### Step 10: Score the full Raiders roster on physical fit

```python
def score_raiders_player_physical(player_id: str) -> dict:
    """Look up a player on the 2026 LV roster and score their physical fit."""
    con = duckdb.connect(DB_PATH)
    row = con.execute("""
        SELECT position FROM rosters
        WHERE gsis_id = ? AND season = 2026 AND team = 'LV'
        LIMIT 1
    """, [player_id]).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Player {player_id} not found in 2026 LV Raiders roster")

    position_group = PHYSICAL_POSITION_MAP.get(row[0])
    if position_group is None:
        raise RuntimeError(f"Position {row[0]} not in PHYSICAL_POSITION_MAP")

    return score_player_physical(player_id, position_group)

def score_offense_physical() -> dict:
    """Score the entire Raiders 2026 offense on physical fit.

    Returns a structure parallel to score_offense() so the UI can render both.
    """
    # Mirror score_offense() structure exactly, but use score_raiders_player_physical
    # Apply same position-group weights for the overall grade
    raise NotImplementedError("Implement following the pattern of score_offense")
```

### Step 11: Persist the physical grades

Update `write_grades()` to also write the physical grades, as parallel tables:

```python
def write_grades() -> None:
    """Compute both Statistical Similarity and Physical Fit grades, persist to DuckDB."""
    statistical_result = score_offense()
    physical_result = score_offense_physical()

    # Persist statistical (existing)
    # Persist physical to NEW tables: raiders_physical_player_grades, raiders_physical_offense_summary
    raise NotImplementedError("Mirror the existing persistence pattern")
```

## Phase C: Verification

### Step 12: Run and inspect

```
uv run python3 src/scoring.py
```

Expected: completes without errors.

```
uv run python3 -c "
import duckdb
con = duckdb.connect('data/raw/nfl.duckdb')

print('=== PHYSICAL OFFENSE SUMMARY ===')
print(con.execute('SELECT * FROM raiders_physical_offense_summary').fetchdf().to_string())

print()
print('=== TOP 5 PHYSICAL GRADES ===')
print(con.execute('''
    SELECT p.player_id, r.full_name, p.position_group, p.grade,
           p.features_used, p.features_missing
    FROM raiders_physical_player_grades p
    LEFT JOIN rosters r ON p.player_id = r.gsis_id AND r.season = 2026 AND r.team = 'LV'
    ORDER BY p.grade DESC
    LIMIT 5
''').fetchdf().to_string())

print()
print('=== BOTTOM 5 PHYSICAL GRADES ===')
print(con.execute('''
    SELECT p.player_id, r.full_name, p.position_group, p.grade,
           p.features_used, p.features_missing
    FROM raiders_physical_player_grades p
    LEFT JOIN rosters r ON p.player_id = r.gsis_id AND r.season = 2026 AND r.team = 'LV'
    ORDER BY p.grade ASC
    LIMIT 5
''').fetchdf().to_string())

print()
print('=== PHYSICAL VS STATISTICAL, SIDE BY SIDE ===')
print(con.execute('''
    SELECT
        s.player_id,
        r.full_name,
        s.position_group,
        s.grade AS statistical_grade,
        p.grade AS physical_grade
    FROM raiders_player_grades s
    JOIN raiders_physical_player_grades p ON s.player_id = p.player_id
    LEFT JOIN rosters r ON s.player_id = r.gsis_id AND r.season = 2026 AND r.team = 'LV'
    ORDER BY p.grade DESC
    LIMIT 15
''').fetchdf().to_string())
"
```

### What to look for

1. **Physical grades should be much less uniformly low than statistical grades.** The statistical grades were depressed because players hadn't run Kubiak's system. The physical grades shouldn't have that problem; physical traits don't depend on scheme experience. Expect a spread roughly between 30 and 90.

2. **Geno Smith physical grade.** He should be plausible. Not necessarily near 100 (he's not in the Kubiak reference set the way Darnold is), but mid-to-high. A QB who Kubiak personally chose for the Raiders should look physically Kubiak-compatible.

3. **Side-by-side comparison.** Look at the last query. Players with high physical fit and low statistical fit are the most interesting; they have the traits but haven't shown the production in a compatible system yet. That's exactly the kind of insight the project was supposed to surface.

Paste all four sections of output when complete.

## Phase D: README update

Add a new section under "Scoring methodology":

```
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

The performance-based grade described above is preserved as secondary context.
It measures how a player's recent statistical profile compares to Kubiak's
reference players' profiles. Low scores often reflect that a player hasn't
run a compatible offense yet, not that they're a poor player. The grade is
useful for identifying players who have already demonstrated Kubiak-system
production, but it should not be used in isolation.

Combine coverage limitation: roughly 24% of reference players have no combine
record at all (largely undrafted free agents who skipped the combine). These
players contribute only height and weight to the physical archetype.
Coverage is weakest for offensive linemen, where shuttle and three-cone data
is incomplete.
```

## What NOT to do in this handoff

- Do not modify the existing performance-based archetype or scoring functions.
- Do not add new features to the Statistical Similarity model.
- Do not build the Streamlit UI. Handoff #5.
- Do not build the Scheme Experience lookup table. Handoff #5.
- Do not impute missing physical features. If a player has no combine and no roster physical data, raise.
- Do not change the grade formula or scale_factor; both models share them so they're comparable.

## Final commit

After Step 12 output looks correct:

```
git add .
git commit -m "Add Physical Fit model: archetype, scoring, and persistence"
git push
```

Stop here. Wait for handoff #5.
