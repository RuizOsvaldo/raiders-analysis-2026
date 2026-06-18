"""Compare players within a position group on both grade dimensions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import coverage_label, load_roster_grades

st.set_page_config(page_title="Position Comparison", layout="wide")
st.title("Position Group Comparison")

df = load_roster_grades()

group = st.selectbox(
    "Position group",
    options=sorted(df["position_group"].dropna().unique()),
)
group_df = df[df["position_group"] == group].copy()
group_df = group_df.sort_values("physical_grade", ascending=False)

st.subheader(f"{group} scatter: Scheme Fit vs Athletic Profile")
st.caption(
    "Bubble size = data coverage (larger = more data). "
    "Color = scheme experience. "
    "Top-right is the ideal quadrant."
)

fig = px.scatter(
    group_df,
    x="statistical_grade",
    y="physical_grade",
    text="player_name",
    color="scheme_experience",
    size="physical_coverage",
    size_max=20,
    hover_data={
        "position": True,
        "experience_bucket": True,
        "physical_coverage": ":.0%",
        "statistical_coverage": ":.0%",
        "player_name": False,
    },
    color_discrete_map={
        "yes":     "#2E8B57",
        "partial": "#DAA520",
        "no":      "#B22222",
        "unknown": "#888888",
    },
    range_x=[0, 100],
    range_y=[0, 100],
    labels={
        "statistical_grade": "Athletic Profile",
        "physical_grade":    "Scheme Fit",
        "scheme_experience": "Scheme Exp",
    },
)
fig.update_traces(textposition="top center", textfont_size=11)
fig.update_layout(height=600)
fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "**Quadrant guide**  \n"
    "Top-right: high physical AND statistical fit -- most credible scheme fits.  \n"
    "Top-left: physically fits but no system production yet -- upside bets.  \n"
    "Bottom-right: produced in similar systems but traits do not match.  \n"
    "Bottom-left: neither traits nor production align -- largest concerns."
)

st.markdown("---")
st.subheader(f"{group} roster")

table_df = group_df[[
    "player_name", "position",
    "physical_grade", "physical_coverage",
    "statistical_grade", "statistical_coverage",
    "scheme_experience", "experience_bucket",
]].copy()
table_df.columns = [
    "Player", "Pos",
    "Physical", "Phys Coverage",
    "Statistical", "Stat Coverage",
    "Scheme Exp", "NFL Exp",
]
table_df["Physical"]   = table_df["Physical"].round(1)
table_df["Statistical"] = table_df["Statistical"].round(1)
table_df["Phys Coverage"] = table_df["Phys Coverage"].apply(coverage_label)
table_df["Stat Coverage"] = table_df["Stat Coverage"].apply(coverage_label)

st.dataframe(table_df, use_container_width=True, hide_index=True)
