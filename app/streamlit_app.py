"""Las Vegas Raiders 2026 Scheme Fit Analysis: app entry point and navigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    RAIDERS_LOGO,
    coverage_label,
    load_offense_summary,
    load_roster_grades,
)

st.set_page_config(page_title="Las Vegas Raiders 2026 Scheme Fit Analysis", layout="wide")


def home() -> None:
    """Roster overview: headline grades, position-group charts, full roster table."""
    # Header row: logo + title
    logo_col, title_col = st.columns([1, 7])
    with logo_col:
        st.image(RAIDERS_LOGO, width=90)
    with title_col:
        st.title("Las Vegas Raiders 2026 Scheme Fit Analysis")
        st.caption(
            "Grading the 2026 Las Vegas Raiders offensive roster against Klint "
            "Kubiak's offensive scheme tendencies. See the About page for the "
            "reference seasons and full methodology."
        )

    summary  = load_offense_summary()
    phys_sum = summary["physical"]
    stat_sum = summary["statistical"]

    # Top-level metrics
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Scheme Fit (overall)", f"{stat_sum['overall_grade'].iloc[0]:.1f} / 100")
        st.caption("Primary grade. Individual PFF grades + profile vs Kubiak's archetype.")
    with m2:
        st.metric("Athletic Profile (overall)", f"{phys_sum['overall_grade'].iloc[0]:.1f} / 100")
        st.caption("Secondary. Combine measurables vs Kubiak's reference athletes.")

    st.markdown("---")

    # Charts row
    groups = ["QB", "RB", "WR", "TE", "OL"]
    phys_vals = [
        float(phys_sum[f"grade_{g}"].iloc[0]) if f"grade_{g}" in phys_sum.columns else 0
        for g in groups
    ]
    stat_vals = [
        float(stat_sum[f"grade_{g}"].iloc[0]) if f"grade_{g}" in stat_sum.columns else 0
        for g in groups
    ]

    chart_left, chart_right = st.columns([1, 1])

    with chart_left:
        st.subheader("Position group fit")
        st.caption("How well each position matches Kubiak's archetype.")

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=stat_vals + [stat_vals[0]],
            theta=groups + [groups[0]],
            fill="toself",
            name="Scheme Fit",
            fillcolor="rgba(0,0,0,0.20)",
            line=dict(color="#000000", width=2),
        ))
        radar.add_trace(go.Scatterpolar(
            r=phys_vals + [phys_vals[0]],
            theta=groups + [groups[0]],
            fill="toself",
            name="Athletic",
            fillcolor="rgba(165,172,175,0.30)",
            line=dict(color="#A5ACAF", width=2, dash="dot"),
        ))
        radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100],
                                tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=13)),
            ),
            showlegend=True,
            legend=dict(x=0.8, y=1.1),
            margin=dict(l=40, r=40, t=20, b=20),
            height=380,
        )
        st.plotly_chart(radar, use_container_width=True)

    with chart_right:
        st.subheader("Scheme Fit vs Athletic by group")
        st.caption("Bars show unit-grade for each position group.")

        bar_df = pd.DataFrame({
            "Position": groups * 2,
            "Grade":    stat_vals + phys_vals,
            "Model":    ["Scheme Fit"] * 5 + ["Athletic"] * 5,
        })
        bar_fig = px.bar(
            bar_df,
            x="Position",
            y="Grade",
            color="Model",
            barmode="group",
            color_discrete_map={"Scheme Fit": "#000000", "Athletic": "#A5ACAF"},
            range_y=[0, 100],
        )
        bar_fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4)
        bar_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=380,
            legend=dict(x=0.6, y=1.05, orientation="h"),
            yaxis_title="Grade (0-100)",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # Roster table
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
        "Scheme Fit", "Fit Coverage",
        "Athletic", "Ath Coverage",
        "Scheme Exp", "NFL Exp",
    ]
    display_df["Scheme Fit"] = display_df["Scheme Fit"].round(1)
    display_df["Athletic"]   = display_df["Athletic"].round(1)
    display_df["Fit Coverage"] = display_df["Fit Coverage"].apply(coverage_label)
    display_df["Ath Coverage"] = display_df["Ath Coverage"].apply(coverage_label)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "Use Player Detail for per-player breakdowns. "
        "Position Comparison for scatter-plot views. "
        "Methodology for how grades are computed."
    )


pages = [
    st.Page(home, title="Home", default=True),
    st.Page("pages/1_Player_Detail.py", title="Player Detail"),
    st.Page("pages/2_Position_Comparison.py", title="Position Comparison"),
    st.Page("pages/3_Methodology.py", title="Methodology"),
    st.Page("pages/4_About.py", title="About"),
]

st.navigation(pages).run()
