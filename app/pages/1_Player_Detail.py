"""Per-player detail: grades, coverage, archetype comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import coverage_label, load_archetypes, load_roster_grades

st.set_page_config(page_title="Player Detail", layout="wide")
st.title("Player Detail")

df = load_roster_grades()
player_name = st.selectbox(
    "Select player",
    options=sorted(df["player_name"].dropna().unique()),
)
player = df[df["player_name"] == player_name].iloc[0]

# Header card
c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
with c1:
    jersey = int(player["jersey_number"]) if pd.notna(player["jersey_number"]) else "--"
    exp    = int(player["years_exp"]) if pd.notna(player["years_exp"]) else "R"
    college = player["college"] if pd.notna(player["college"]) else "unknown"
    st.subheader(player["player_name"])
    st.caption(f"#{jersey} | {player['position']} | {college} | {exp} yr NFL")

with c2:
    phys = player["physical_grade"]
    st.metric("Physical Fit", f"{phys:.1f}" if pd.notna(phys) else "N/A")
    st.caption(coverage_label(player["physical_coverage"]))

with c3:
    stat = player["statistical_grade"]
    st.metric("Statistical", f"{stat:.1f}" if pd.notna(stat) else "N/A")
    st.caption(coverage_label(player["statistical_coverage"]))

with c4:
    raw = player["physical_raw"]
    cov = player["physical_coverage"]
    if pd.notna(raw) and pd.notna(cov):
        st.metric("Raw (pre-penalty)", f"{raw:.1f}")
        st.caption(f"x {cov:.0%} coverage = {phys:.1f}")
    else:
        st.metric("Raw (pre-penalty)", "N/A")

# Scheme experience
exp_map = {"yes": ":green[YES]", "partial": ":orange[PARTIAL]", "no": ":red[NO]", "unknown": ":gray[UNKNOWN]"}
sx = player["scheme_experience"]
st.markdown(f"**Scheme experience (Kubiak-tree):** {exp_map.get(sx, sx.upper())}")
st.markdown(f"**NFL experience bucket:** {player['experience_bucket']}")

st.markdown("---")

# Missing features
if pd.notna(player["physical_missing_names"]) and player["physical_missing_names"]:
    st.warning(
        f"Physical grade missing features: **{player['physical_missing_names']}**. "
        f"Grade reflects {player['physical_features_used']} of "
        f"{player['physical_features_used'] + player['physical_features_missing']} features."
    )

# Archetype comparison
st.subheader("Position archetype targets (Physical Fit)")
archetypes = load_archetypes()
pos_group = player["position_group"]
phys_arch = archetypes["physical"]

if pos_group and not phys_arch.empty:
    arch_row = phys_arch[phys_arch["position_group"] == pos_group]
    if not arch_row.empty:
        arch = arch_row.iloc[0]
        feat_cols = [c for c in arch.index if not c.endswith("_n_players") and c != "position_group"]

        arch_display = pd.DataFrame({
            "Feature": feat_cols,
            "Kubiak Archetype": [round(arch[f], 2) if pd.notna(arch[f]) else "N/A" for f in feat_cols],
            "Players in avg": [arch.get(f"{f}_n_players", "N/A") for f in feat_cols],
        })
        st.dataframe(arch_display, use_container_width=True, hide_index=True)
        st.caption(
            "The archetype is the snap-weighted mean of the physical features across Kubiak's "
            "2024 Saints and 2025 Seahawks reference roster at this position. "
            "ht/wt in inches/lbs; forty/shuttle/cone in seconds; vertical/broad_jump in inches."
        )
    else:
        st.info(f"No physical archetype found for position group {pos_group}.")
else:
    st.info("Position group not available for this player.")

# Performance archetype for reference
st.subheader("Position archetype targets (Statistical Similarity)")
perf_arch = archetypes["performance"]
if pos_group and not perf_arch.empty:
    arch_row = perf_arch[perf_arch["position_group"] == pos_group]
    if not arch_row.empty:
        arch = arch_row.iloc[0]
        feat_cols = [c for c in arch.index if c != "position_group"]
        arch_display = pd.DataFrame({
            "Feature": feat_cols,
            "Kubiak Archetype": [round(float(arch[f]), 4) if pd.notna(arch[f]) else "N/A" for f in feat_cols],
        })
        st.dataframe(arch_display, use_container_width=True, hide_index=True)
        st.caption(
            "Performance archetype from the same reference set. "
            "Low Statistical Similarity often means the player has not run a compatible scheme, "
            "not that they cannot perform in one."
        )
