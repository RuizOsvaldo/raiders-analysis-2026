"""Shared helpers for the Streamlit app: DuckDB readers with caching."""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = str(Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")

# Physical feature lists per position (mirrors src/archetype.py)
PHYSICAL_FEATURES = {
    "QB": ["ht", "wt", "forty", "shuttle", "cone", "vertical", "broad_jump"],
    "RB": ["ht", "wt", "forty", "vertical", "broad_jump"],
    "WR": ["ht", "wt", "forty", "vertical", "broad_jump"],
    "TE": ["ht", "wt", "vertical", "broad_jump"],
    "OL": ["ht", "wt", "forty", "shuttle", "cone", "vertical", "broad_jump"],
}

FEATURE_LABELS = {
    "ht": "Height", "wt": "Weight", "forty": "40-Yard",
    "shuttle": "Shuttle", "cone": "3-Cone",
    "vertical": "Vertical", "broad_jump": "Broad Jump",
}

# (min, max, lower_is_better) — NFL combine realistic bounds
FEATURE_RANGES = {
    "ht":         (66.0, 80.0,  False),
    "wt":         (155.0, 365.0, False),
    "forty":      (4.2,  5.8,   True),
    "shuttle":    (3.8,  5.2,   True),
    "cone":       (6.4,  8.0,   True),
    "vertical":   (20.0, 48.0,  False),
    "broad_jump": (80.0, 138.0, False),
}

RAIDERS_LOGO = "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png"


def normalize_feature(feat: str, value) -> float | None:
    """Return a 0-100 score where 100 = best possible for that feature."""
    if feat not in FEATURE_RANGES or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    lo, hi, lower_is_better = FEATURE_RANGES[feat]
    normed = (v - lo) / (hi - lo)
    if lower_is_better:
        normed = 1.0 - normed
    return round(max(0.0, min(100.0, normed * 100)), 1)


@st.cache_data(ttl=600)
def load_roster_grades() -> pd.DataFrame:
    """Return a single DataFrame joining physical, statistical, scheme experience, and roster."""
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT
            r.player_id,
            r.player_name,
            r.position,
            r.jersey_number,
            r.years_exp,
            r.college,
            r.headshot_url,
            phys.position_group,
            phys.grade        AS physical_grade,
            phys.raw_grade    AS physical_raw,
            phys.coverage     AS physical_coverage,
            phys.features_used        AS physical_features_used,
            phys.features_missing     AS physical_features_missing,
            phys.missing_feature_names AS physical_missing_names,
            stat.grade        AS statistical_grade,
            stat.coverage     AS statistical_coverage,
            stat.features_used        AS statistical_features_used,
            stat.features_missing     AS statistical_features_missing,
            stat.experience_bucket,
            COALESCE(sx.scheme_experience, 'unknown') AS scheme_experience
        FROM rosters r
        LEFT JOIN raiders_physical_player_grades phys ON phys.player_id = r.player_id
        LEFT JOIN raiders_player_grades          stat ON stat.player_id  = r.player_id
        LEFT JOIN scheme_experience              sx   ON sx.player_id    = r.player_id
        WHERE r.season = 2026 AND r.team = 'LV'
          AND r.position IN ('QB','RB','FB','WR','TE','T','G','C','OT','OG','OL')
        ORDER BY phys.position_group, phys.grade DESC NULLS LAST
    """).fetchdf()
    con.close()
    return df


@st.cache_data(ttl=600)
def load_offense_summary() -> dict:
    """Return both offense summaries (physical and statistical)."""
    con = duckdb.connect(DB_PATH)
    physical    = con.execute("SELECT * FROM raiders_physical_offense_summary").fetchdf()
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
    result = {
        "physical":    con.execute("SELECT * FROM kubiak_physical_archetypes").fetchdf(),
        "performance": con.execute("SELECT * FROM kubiak_position_archetypes").fetchdf(),
    }
    con.close()
    return result


@st.cache_data(ttl=600)
def load_player_physical_features(player_id: str, position_group: str) -> dict:
    """Return a player's raw physical values alongside archetype values.

    Returns {feat: {"player": float|None, "archetype": float|None,
                    "player_norm": float|None, "arch_norm": float|None}}
    """
    features = PHYSICAL_FEATURES.get(position_group, [])
    if not features:
        return {}

    con = duckdb.connect(DB_PATH)
    row = con.execute("""
        SELECT
            AVG(r.height)     AS ht,
            AVG(r.weight)     AS wt,
            MAX(c.forty)      AS forty,
            MAX(c.shuttle)    AS shuttle,
            MAX(c.cone)       AS cone,
            MAX(c.vertical)   AS vertical,
            MAX(c.broad_jump) AS broad_jump
        FROM rosters r
        LEFT JOIN combine c ON c.pfr_id = COALESCE(
            r.pfr_id,
            (SELECT pfr_player_id FROM snap_counts WHERE player = r.player_name LIMIT 1)
        )
        WHERE r.player_id = ?
        GROUP BY r.player_id
    """, [player_id]).fetchone()

    arch_row = con.execute(
        "SELECT * FROM kubiak_physical_archetypes WHERE position_group = ?",
        [position_group],
    ).fetchdf()
    con.close()

    if row is None or arch_row.empty:
        return {}

    arch = arch_row.iloc[0]
    raw_map = {"ht": row[0], "wt": row[1], "forty": row[2],
               "shuttle": row[3], "cone": row[4],
               "vertical": row[5], "broad_jump": row[6]}

    result = {}
    for feat in features:
        pval = raw_map.get(feat)
        aval = float(arch[feat]) if feat in arch.index and not pd.isna(arch[feat]) else None
        result[feat] = {
            "player":    float(pval) if pval is not None else None,
            "archetype": aval,
            "player_norm": normalize_feature(feat, pval),
            "arch_norm":   normalize_feature(feat, aval),
        }
    return result


def coverage_label(coverage) -> str:
    """Human-readable coverage label."""
    if coverage is None or (isinstance(coverage, float) and pd.isna(coverage)):
        return "unknown"
    pct = int(coverage * 100)
    if pct >= 99:
        return "full data"
    if pct >= 60:
        return f"{pct}% data"
    return f"sparse ({pct}%)"


def grade_band(grade) -> str:
    """Return 'high', 'mid', or 'low' for a grade value."""
    if grade is None or (isinstance(grade, float) and pd.isna(grade)):
        return "low"
    if grade >= 65:
        return "high"
    if grade >= 45:
        return "mid"
    return "low"
