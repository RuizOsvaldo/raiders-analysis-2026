"""Shared helpers for the Streamlit app: CSV readers with caching."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

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
    return pd.read_csv(DATA_DIR / "roster_grades.csv")


@st.cache_data(ttl=600)
def load_offense_summary() -> dict:
    """Return both offense summaries (physical and statistical)."""
    return {
        "physical":    pd.read_csv(DATA_DIR / "physical_offense_summary.csv"),
        "statistical": pd.read_csv(DATA_DIR / "offense_summary.csv"),
    }


@st.cache_data(ttl=600)
def load_scheme_profile() -> pd.DataFrame:
    """Return Kubiak's scheme profile (play-call rates by RZ vs non-RZ)."""
    return pd.read_csv(DATA_DIR / "scheme_profile.csv")


@st.cache_data(ttl=600)
def load_archetypes() -> dict:
    """Return both archetype tables (physical and performance)."""
    return {
        "physical":    pd.read_csv(DATA_DIR / "physical_archetypes.csv"),
        "performance": pd.read_csv(DATA_DIR / "position_archetypes.csv"),
    }


@st.cache_data(ttl=600)
def load_player_physical_features(player_id: str, position_group: str) -> dict:
    """Return a player's raw physical values alongside archetype values.

    Returns {feat: {"player": float|None, "archetype": float|None,
                    "player_norm": float|None, "arch_norm": float|None}}
    """
    features = PHYSICAL_FEATURES.get(position_group, [])
    if not features:
        return {}

    combine_df = pd.read_csv(DATA_DIR / "player_combine.csv")
    arch_df    = pd.read_csv(DATA_DIR / "physical_archetypes.csv")

    player_row = combine_df[combine_df["player_id"] == player_id]
    arch_row   = arch_df[arch_df["position_group"] == position_group]

    if player_row.empty or arch_row.empty:
        return {}

    player = player_row.iloc[0]
    arch   = arch_row.iloc[0]

    result = {}
    for feat in features:
        pval = player.get(feat)
        aval = arch.get(feat)
        pval = float(pval) if pval is not None and not pd.isna(pval) else None
        aval = float(aval) if aval is not None and not pd.isna(aval) else None
        result[feat] = {
            "player":    pval,
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
