"""Raiders Analysis 2026: roster overview page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from utils import coverage_label, load_offense_summary, load_roster_grades

st.set_page_config(
    page_title="Raiders Analysis 2026",
    layout="wide",
)

st.title("Raiders 2026 Scheme Fit Analysis")
st.caption(
    "Grading the 2026 Las Vegas Raiders offensive roster against Klint Kubiak's "
    "scheme tendencies from 2024 New Orleans (40%) and 2025 Seattle (60%)."
)

summary = load_offense_summary()
phys_summary = summary["physical"]
stat_summary = summary["statistical"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Physical Fit (overall)", f"{phys_summary['overall_grade'].iloc[0]:.1f}")
    st.caption("Primary grade. Trait-based, coverage-penalized.")
with col2:
    st.metric("Statistical Similarity (overall)", f"{stat_summary['overall_grade'].iloc[0]:.1f}")
    st.caption("Secondary. Performance similarity to reference players.")
with col3:
    st.metric("Reference seasons", "Saints 2024 + Seahawks 2025")
    st.caption("Weighted 40 / 60.")

st.markdown("---")

st.subheader("By position group")
groups = ["QB", "RB", "WR", "TE", "OL"]
group_cols = st.columns(len(groups))
for i, g in enumerate(groups):
    with group_cols[i]:
        phys_col = f"grade_{g}"
        stat_col = f"grade_{g}"
        phys_val = phys_summary[phys_col].iloc[0] if phys_col in phys_summary.columns else None
        stat_val = stat_summary[stat_col].iloc[0] if stat_col in stat_summary.columns else None
        st.markdown(f"**{g}**")
        if phys_val is not None and not pd.isna(phys_val):
            st.markdown(f"Physical: **{phys_val:.1f}**")
        if stat_val is not None and not pd.isna(stat_val):
            st.markdown(f"Stat: {stat_val:.1f}")

st.markdown("---")

st.subheader("Full roster")
df = load_roster_grades()

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
        "Min physical coverage",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="Hide grades built on sparse data. 1.0 = full data only.",
    )

filtered = df[
    df["position_group"].isin(position_filter)
    & df["scheme_experience"].isin(scheme_filter)
    & (df["physical_coverage"].fillna(0) >= min_coverage)
].copy()

display_df = filtered[[
    "player_name", "position", "position_group",
    "physical_grade", "physical_coverage",
    "statistical_grade", "statistical_coverage",
    "scheme_experience", "experience_bucket",
]].copy()

display_df.columns = [
    "Player", "Pos", "Group",
    "Physical", "Phys Coverage",
    "Statistical", "Stat Coverage",
    "Scheme Exp", "NFL Exp",
]
display_df["Physical"]   = display_df["Physical"].round(1)
display_df["Statistical"] = display_df["Statistical"].round(1)
display_df["Phys Coverage"] = display_df["Phys Coverage"].apply(coverage_label)
display_df["Stat Coverage"] = display_df["Stat Coverage"].apply(coverage_label)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    "Use the Player Detail page for per-player breakdowns. "
    "Position Comparison for scatter-plot views by group. "
    "Methodology for how grades are computed."
)
