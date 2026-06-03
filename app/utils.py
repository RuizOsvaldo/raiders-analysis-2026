"""Shared helpers for the Streamlit app: DuckDB readers with caching."""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DB_PATH = str(Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")


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
        "physical":     con.execute("SELECT * FROM kubiak_physical_archetypes").fetchdf(),
        "performance":  con.execute("SELECT * FROM kubiak_position_archetypes").fetchdf(),
    }
    con.close()
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
    """Return 'high', 'mid', or 'low' band for a grade."""
    if grade is None or (isinstance(grade, float) and pd.isna(grade)):
        return "low"
    if grade >= 65:
        return "high"
    if grade >= 45:
        return "mid"
    return "low"
