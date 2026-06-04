"""Per-player detail: headshot, grades, feature radar, archetype comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    FEATURE_LABELS,
    PHYSICAL_FEATURES,
    coverage_label,
    load_archetypes,
    load_player_physical_features,
    load_roster_grades,
)

st.set_page_config(page_title="Las Vegas Raiders 2026 | Player Detail", layout="wide")
st.title("Player Detail")

df = load_roster_grades()
player_name = st.selectbox(
    "Select player",
    options=sorted(df["player_name"].dropna().unique()),
)
player = df[df["player_name"] == player_name].iloc[0]
pos_group = player["position_group"]

st.markdown("---")

# Header: photo + bio + grades
photo_col, bio_col, phys_col, stat_col, raw_col = st.columns([1, 3, 1, 1, 1])

with photo_col:
    headshot = player.get("headshot_url")
    if pd.notna(headshot) and headshot:
        st.image(str(headshot), width=120)
    else:
        st.markdown("*(no photo)*")

with bio_col:
    jersey = int(player["jersey_number"]) if pd.notna(player["jersey_number"]) else "--"
    exp    = int(player["years_exp"]) if pd.notna(player["years_exp"]) else "R"
    college = player["college"] if pd.notna(player["college"]) else "unknown"
    st.subheader(player["player_name"])
    st.markdown(f"**#{jersey}** | {player['position']} | {college} | {exp} yr NFL")
    exp_map = {
        "yes":     ":green[YES]",
        "partial": ":orange[PARTIAL]",
        "no":      ":red[NO]",
        "unknown": ":gray[UNKNOWN]",
    }
    sx = player["scheme_experience"]
    st.markdown(f"Scheme experience: {exp_map.get(sx, sx.upper())}")
    st.markdown(f"NFL bucket: **{player['experience_bucket']}**")

with phys_col:
    phys = player["physical_grade"]
    st.metric("Physical Fit", f"{phys:.1f}" if pd.notna(phys) else "N/A")
    st.caption(coverage_label(player["physical_coverage"]))

with stat_col:
    stat = player["statistical_grade"]
    st.metric("Statistical", f"{stat:.1f}" if pd.notna(stat) else "N/A")
    st.caption(coverage_label(player["statistical_coverage"]))

with raw_col:
    raw = player["physical_raw"]
    cov = player["physical_coverage"]
    if pd.notna(raw) and pd.notna(cov):
        st.metric("Pre-penalty", f"{raw:.1f}")
        st.caption(f"x {cov:.0%} = {phys:.1f}")
    else:
        st.metric("Pre-penalty", "N/A")

# Missing features warning
if pd.notna(player["physical_missing_names"]) and player["physical_missing_names"]:
    n_used  = int(player["physical_features_used"])
    n_total = n_used + int(player["physical_features_missing"])
    st.warning(
        f"Missing combine data: **{player['physical_missing_names']}**. "
        f"Grade uses {n_used} of {n_total} features."
    )

st.markdown("---")

# Feature radar: player vs archetype
st.subheader("Physical fit vs Kubiak archetype")
st.caption(
    "Each axis is one physical feature, normalized to a 0-100 scale "
    "(100 = best recorded NFL value; archetype ring = what Kubiak's players averaged). "
    "Player shape filling the archetype ring means a strong physical match."
)

feat_data = load_player_physical_features(player["player_id"], pos_group) if pos_group else {}

if feat_data:
    features    = list(feat_data.keys())
    feat_labels = [FEATURE_LABELS.get(f, f) for f in features]

    player_scores = [
        (feat_data[f]["player_norm"] if feat_data[f]["player_norm"] is not None else 0)
        for f in features
    ]
    arch_scores = [
        (feat_data[f]["arch_norm"] if feat_data[f]["arch_norm"] is not None else 0)
        for f in features
    ]

    # Close the polygon
    theta = feat_labels + [feat_labels[0]]
    r_arch   = arch_scores   + [arch_scores[0]]
    r_player = player_scores + [player_scores[0]]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=r_arch,
        theta=theta,
        fill="toself",
        name="Archetype",
        fillcolor="rgba(165,172,175,0.4)",
        line=dict(color="#A5ACAF", width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra>Archetype</extra>",
    ))
    radar.add_trace(go.Scatterpolar(
        r=r_player,
        theta=theta,
        fill="toself",
        name=player_name,
        fillcolor="rgba(0,0,0,0.25)",
        line=dict(color="#000000", width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra>Player</extra>",
    ))
    radar.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=9),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=True,
        legend=dict(x=0.8, y=1.1),
        margin=dict(l=60, r=60, t=30, b=30),
        height=420,
    )
    st.plotly_chart(radar, use_container_width=True)

    # Raw values table
    with st.expander("Raw feature values"):
        rows = []
        for f in features:
            d = feat_data[f]
            rows.append({
                "Feature":         FEATURE_LABELS.get(f, f),
                "Player value":    round(d["player"], 2) if d["player"] is not None else "N/A",
                "Archetype target": round(d["archetype"], 2) if d["archetype"] is not None else "N/A",
                "Player score":    f"{d['player_norm']:.1f}" if d["player_norm"] is not None else "missing",
                "Archetype score": f"{d['arch_norm']:.1f}"   if d["arch_norm"]   is not None else "--",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Scores normalize each feature to a 0-100 scale across NFL combine ranges. "
            "For speed metrics (40-yard, shuttle, 3-cone), faster = higher score."
        )
else:
    st.info("No physical feature data available for this player.")

st.markdown("---")

# Performance archetype for reference
st.subheader("Statistical archetype targets")
archetypes = load_archetypes()
perf_arch  = archetypes["performance"]

if pos_group and not perf_arch.empty:
    arch_row = perf_arch[perf_arch["position_group"] == pos_group]
    if not arch_row.empty:
        arch = arch_row.iloc[0]
        feat_cols = [c for c in arch.index if c != "position_group"]
        arch_display = pd.DataFrame({
            "Feature":         feat_cols,
            "Kubiak archetype": [
                round(float(arch[f]), 4) if pd.notna(arch[f]) else "N/A"
                for f in feat_cols
            ],
        })
        st.dataframe(arch_display, use_container_width=True, hide_index=True)
        st.caption(
            "Performance archetype from the same reference set. "
            "Low Statistical Similarity often means the player has not run a "
            "compatible scheme yet."
        )
