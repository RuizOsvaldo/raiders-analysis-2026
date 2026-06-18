"""Score current Raiders players against the Kubiak archetypes."""

import math
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/raw/nfl.duckdb"

try:
    from archetype import (PFF_SKILL, PHYSICAL_FEATURES, PHYSICAL_POSITION_MAP,
                           norm_name, norm_sql)
except ImportError:
    from src.archetype import (PFF_SKILL, PHYSICAL_FEATURES, PHYSICAL_POSITION_MAP,
                               norm_name, norm_sql)

# Years of NFL experience determine combine vs performance weighting
# (years counted as completed seasons before 2026)
# Rookies are scored on combine measurables only; veterans on NFL performance
# only. Sophomores (one prior season) blend the two CONTINUOUSLY by how much
# they actually played as a rookie: a sophomore with very few NFL snaps has a
# noisy performance signal, so the blend leans on the more reliable combine
# athleticism. The performance weight is snaps / (snaps + K): a 2-snap rookie
# season contributes almost nothing, a near-full season (~700+) contributes ~0.7.
# This replaces an earlier binary 40%-snap-share cutoff that trusted a 2-snap
# sample exactly as much as a 350-snap one.
SOPHOMORE_SNAP_K = 300

WEIGHTING_RULES = {
    "rookie":   {"combine": 1.00, "performance": 0.00},
    "veteran":  {"combine": 0.00, "performance": 1.00},
}

# Every position's performance features come from PFF (see PFF_SKILL in
# archetype.py). Skill positions are profile-matched (two-sided distance); OL is
# a one-sided quality bar. Must match columns in kubiak_position_archetypes.
POSITION_FEATURES = {g: cfg["features"] for g, cfg in PFF_SKILL.items()}

# Features that are one-directional quality scores (being BETTER than the
# archetype is not a deviation). Only OL is configured this way.
ONE_SIDED_HIGHER_BETTER = {g: set(cfg["features"]) for g, cfg in PFF_SKILL.items()
                           if cfg.get("one_sided")}

# Weight a rookie's college PFF grade gets in the blend (rest is athletic
# combine). College PFF is a real full-season signal but a college-to-NFL
# projection, so it is trusted heavily but not fully.
COLLEGE_PERF_WEIGHT = 0.65

# Combine table uses 'cone' (not 'three_cone'), 'ht' as "F-I" string, no hand_size/arm_length
COMBINE_FEATURES_BY_POSITION = {
    "QB": ["ht", "wt", "forty", "cone", "shuttle"],
    "RB": ["ht", "wt", "forty", "shuttle", "cone", "broad_jump", "vertical"],
    "WR": ["ht", "wt", "forty", "shuttle", "cone", "broad_jump", "vertical"],
    "TE": ["ht", "wt", "forty", "shuttle", "cone", "broad_jump", "vertical"],
    "OL": ["ht", "wt", "shuttle", "cone", "broad_jump"],
}

POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
}

# Grade formula: grade = 100 * exp(-distance / SCALE_FACTOR)
# distance is now the ROOT-MEAN-SQUARE standardized deviation across the
# features actually available for a player (see _compute_grade), so it is on
# a per-feature scale (typically ~0.5-3.0 league std) and is invariant to how
# many features are present. A player whose available traits sit 1 RMS-std
# from the archetype grades ~51; 2 RMS-std grades ~26; on-archetype grades 100.
# Because the metric is now a mean rather than a sum, SCALE_FACTOR no longer
# has to absorb the dimensionality of the feature set, so the old hand-tuned
# 5.0 (calibrated against summed distance) is replaced by a value derived from
# the per-feature std scale.
SCALE_FACTOR = 1.5

# Base uncertainty for confidence interval width
BASE_UNCERTAINTY = 0.15

# Overall offense grade weights (QB 30%, OL 25%, WR 15%, RB 20% covers RB+blocking, TE 10%)
OFFENSE_WEIGHTS = {
    "QB": 0.30,
    "OL": 0.25,
    "WR": 0.15,
    "RB": 0.20,
    "TE": 0.10,
}

# Which model supplies each position's PRIMARY scheme-fit grade.
# Every position's PRIMARY grade is now the PFF performance model: individual
# PFF grades/profile metrics separate Kubiak's players from a control population
# far better than the old combine measurables. Veterans are graded on their NFL
# PFF, rookies on their college PFF, sophomores on a snap-weighted blend; the
# combine/athletic model is retained only as a secondary view and a last-resort
# fallback when a player has no PFF data at all.
PRIMARY_MODEL = {g: "performance" for g in PFF_SKILL}

# Regular season week ceiling
REG_SEASON_MAX_WEEK = 18
RZ_YARDLINE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _height_to_inches(ht: str) -> float:
    """Convert 'F-I' height string to total inches."""
    parts = str(ht).split("-")
    return float(parts[0]) * 12 + float(parts[1])


def _resolve_pfr_id(gsis_id: str, player_name: str, con) -> str | None:
    """Return pfr_id for a player, falling back to name-match in snap_counts.

    OL players in nflreadpy rosters have pfr_id=NULL across all seasons.
    snap_counts (from nfl_data_py/PFR) has their pfr_player_id under the
    player name column, so a name match is the reliable resolution path.
    """
    row = con.execute(
        "SELECT pfr_id FROM rosters WHERE player_id = ? AND pfr_id IS NOT NULL LIMIT 1",
        [gsis_id],
    ).fetchone()
    if row:
        return row[0]

    row = con.execute(
        "SELECT pfr_player_id FROM snap_counts WHERE player = ? LIMIT 1",
        [player_name],
    ).fetchone()
    return row[0] if row else None


def _safe_div(num, denom) -> float:
    try:
        d = float(denom)
        if d != 0 and not math.isnan(d):
            return float(num / d)
    except (TypeError, ValueError):
        pass
    return float("nan")


def _recent_seasons(gsis_id: str, con) -> list[int]:
    """Return up to the 2 most recent seasons this player has weekly_stats data."""
    seasons = con.execute(
        "SELECT DISTINCT season FROM weekly_stats WHERE player_id = ? AND season_type = 'REG' ORDER BY season DESC LIMIT 2",
        [gsis_id],
    ).fetchdf()["season"].tolist()
    return sorted(seasons)


# ---------------------------------------------------------------------------
# Phase A: Player classification and feature building
# ---------------------------------------------------------------------------

def classify_player(pfr_id: str | None) -> tuple[str, float]:
    """Return (experience_label, performance_weight).

    performance_weight is the fraction of the grade driven by NFL performance
    (the rest comes from combine). Rookies -> 0.0 (combine only); veterans ->
    1.0 (performance only); sophomores -> snaps / (snaps + SOPHOMORE_SNAP_K),
    so a sophomore's NFL signal is trusted in proportion to how much they played.
    Players with no pfr_id (common for OL in nflreadpy data) default to rookie.
    """
    if pfr_id is None:
        return "rookie", 0.0

    con = duckdb.connect(DB_PATH)
    prior_seasons = con.execute(
        "SELECT DISTINCT season FROM snap_counts WHERE pfr_player_id = ? AND season < 2026 ORDER BY season",
        [pfr_id],
    ).fetchdf()["season"].tolist()

    if len(prior_seasons) == 0:
        con.close()
        return "rookie", 0.0

    if len(prior_seasons) == 1:
        rookie_year = prior_seasons[0]
        snaps = con.execute(
            "SELECT SUM(offense_snaps) FROM snap_counts WHERE pfr_player_id = ? AND season = ?",
            [pfr_id, rookie_year],
        ).fetchone()[0]
        con.close()
        snaps = float(snaps) if snaps is not None else 0.0
        w_perf = snaps / (snaps + SOPHOMORE_SNAP_K)
        label = f"sophomore ({int(snaps)} snaps)"
        return label, w_perf

    con.close()
    return "veteran", 1.0


def classify_pff(player_name: str, position_group: str, con) -> tuple[str, float]:
    """Experience + performance weight for a player, derived from PFF data.

    Read straight from the PFF report (names match the roster, unlike snap_counts
    nicknames): a player with prior NFL PFF seasons is a veteran (>=2) or snap-
    weighted sophomore (1); one with only a college PFF season is a rookie graded
    on college; one with neither falls back to combine athleticism (weight 0).
    """
    cfg = PFF_SKILL[position_group]
    nkey = norm_name(player_name)
    nfl = con.execute(f"""
        SELECT season, SUM({cfg['snap_col']}) AS snaps
        FROM {cfg['report']}
        WHERE {norm_sql("player")} = ? AND level = 'nfl' AND season < 2026
        GROUP BY season
    """, [nkey]).fetchdf()

    if nfl.empty:
        has_college = con.execute(
            f"SELECT 1 FROM {cfg['report']} WHERE {norm_sql('player')} = ? "
            f"AND level = 'college' LIMIT 1", [nkey],
        ).fetchone()
        if has_college:
            return "rookie (college)", COLLEGE_PERF_WEIGHT
        return "rookie", 0.0

    if len(nfl) >= 2:
        return "veteran", 1.0

    snaps = float(nfl["snaps"].sum() or 0.0)
    return f"sophomore ({int(snaps)} snaps)", snaps / (snaps + SOPHOMORE_SNAP_K)


def get_combine_features(pfr_id: str, position_group: str) -> pd.Series:
    """Return the player's combine measurables. Raises if no combine record exists.

    Join key on combine table is pfr_id. Height is stored as 'F-I' string
    and converted to inches here.
    """
    features = COMBINE_FEATURES_BY_POSITION[position_group]
    con = duckdb.connect(DB_PATH)
    row = con.execute(
        f"SELECT {', '.join(features)} FROM combine WHERE pfr_id = ?",
        [pfr_id],
    ).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(
            f"Player {pfr_id} has no combine record. Cannot score combine component."
        )

    result = {}
    for feat, val in zip(features, row):
        if feat == "ht" and val is not None:
            try:
                result[feat] = _height_to_inches(val)
            except Exception:
                result[feat] = float("nan")
        else:
            result[feat] = float(val) if val is not None else float("nan")

    return pd.Series(result)


# ---------------------------------------------------------------------------
# Per-position performance helpers
# ---------------------------------------------------------------------------

def _qb_performance(gsis_id: str, con, seasons: list[int] | None = None) -> pd.Series:
    if seasons is None:
        seasons = _recent_seasons(gsis_id, con)
    if not seasons:
        raise RuntimeError(f"QB {gsis_id}: no weekly_stats found")

    ws = con.execute("""
        SELECT SUM(completions) AS comp, SUM(attempts) AS att,
               SUM(passing_epa) AS epa, SUM(sacks_suffered) AS sacks
        FROM weekly_stats
        WHERE player_id = ? AND season IN (SELECT UNNEST(?)) AND season_type = 'REG'
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    if ws["att"] == 0:
        raise RuntimeError(f"QB {gsis_id}: zero pass attempts in {seasons}")

    total_db = ws["att"] + ws["sacks"]

    ngs = con.execute("""
        SELECT SUM(attempts * avg_time_to_throw) / NULLIF(SUM(attempts), 0) AS ttt,
               SUM(completions * avg_completed_air_yards) / NULLIF(SUM(completions), 0) AS cay
        FROM ngs_passing
        WHERE player_gsis_id = ? AND season IN (SELECT UNNEST(?)) AND season_type = 'REG'
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    spl = con.execute("""
        SELECT
            SUM(CASE WHEN f.is_play_action = 1 AND p.qb_dropback = 1 THEN 1 ELSE 0 END)  AS pa_db,
            SUM(CASE WHEN f.is_play_action = 1 AND p.complete_pass = 1 THEN 1 ELSE 0 END) AS pa_cp,
            SUM(CASE WHEN f.is_qb_out_of_pocket = 1 AND p.qb_dropback = 1 THEN 1 ELSE 0 END)  AS oop_db,
            SUM(CASE WHEN f.is_qb_out_of_pocket = 1 AND p.complete_pass = 1 THEN 1 ELSE 0 END) AS oop_cp,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.qb_dropback = 1 THEN 1 ELSE 0 END)    AS rz_db,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.complete_pass = 1 THEN 1 ELSE 0 END)  AS rz_cp,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.qb_dropback = 1 THEN p.epa ELSE 0 END) AS rz_epa
        FROM pbp p
        JOIN ftn f ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
        WHERE p.passer_player_id = ? AND p.season IN (SELECT UNNEST(?)) AND p.week <= ?
    """, [RZ_YARDLINE, RZ_YARDLINE, RZ_YARDLINE, gsis_id, seasons, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    return pd.Series({
        "completion_pct":               _safe_div(ws["comp"], ws["att"]),
        "epa_per_dropback":             _safe_div(ws["epa"], total_db),
        "sack_rate":                    _safe_div(ws["sacks"], total_db),
        "time_to_throw":                float(ngs["ttt"]) if ngs["ttt"] is not None and not math.isnan(float(ngs["ttt"])) else float("nan"),
        "completed_air_yards":          float(ngs["cay"]) if ngs["cay"] is not None and not math.isnan(float(ngs["cay"])) else float("nan"),
        "play_action_completion_pct":   _safe_div(spl["pa_cp"], spl["pa_db"]),
        "out_of_pocket_completion_pct": _safe_div(spl["oop_cp"], spl["oop_db"]),
        "rz_completion_pct":            _safe_div(spl["rz_cp"], spl["rz_db"]),
        "rz_epa_per_dropback":          _safe_div(spl["rz_epa"], spl["rz_db"]),
    })


def _rb_performance(gsis_id: str, pfr_id: str | None, con, seasons: list[int] | None = None) -> pd.Series:
    if seasons is None:
        seasons = _recent_seasons(gsis_id, con)
    if not seasons:
        raise RuntimeError(f"RB {gsis_id}: no weekly_stats found")

    ws2 = con.execute("""
        SELECT SUM(w.target_share * w.carries) / NULLIF(SUM(w.carries), 0) AS rb_tgt_share,
               MAX(w.team) AS team
        FROM weekly_stats w
        WHERE w.player_id = ? AND w.season IN (SELECT UNNEST(?)) AND w.season_type = 'REG'
          AND (w.carries + w.targets) > 0
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    ngs = con.execute("""
        SELECT SUM(rush_attempts * rush_yards_over_expected_per_att) / NULLIF(SUM(rush_attempts), 0) AS ryoe
        FROM ngs_rushing
        WHERE player_gsis_id = ? AND season IN (SELECT UNNEST(?)) AND season_type = 'REG'
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    # Snap share from snap_counts (needs pfr_id)
    snap_share = float("nan")
    if pfr_id:
        row = con.execute("""
            SELECT AVG(offense_pct) FROM snap_counts
            WHERE pfr_player_id = ? AND season IN (SELECT UNNEST(?))
        """, [pfr_id, seasons]).fetchone()
        if row and row[0] is not None:
            snap_share = float(row[0])

    pbp = con.execute("""
        SELECT
            SUM(CASE WHEN play_type='run' AND run_gap='end' THEN 1 ELSE 0 END)           AS oz_plays,
            SUM(CASE WHEN play_type='run' AND run_gap='end' AND success=1 THEN 1 ELSE 0 END) AS oz_succ,
            SUM(CASE WHEN play_type='run' AND shotgun=1 THEN 1 ELSE 0 END)                AS sg_plays,
            SUM(CASE WHEN play_type='run' AND shotgun=1 AND success=1 THEN 1 ELSE 0 END) AS sg_succ,
            SUM(CASE WHEN yardline_100 <= ? AND play_type='run' THEN 1 ELSE 0 END)        AS rz_rush,
            SUM(CASE WHEN yardline_100 <= ? AND play_type='run' AND success=1 THEN 1 ELSE 0 END) AS rz_succ
        FROM pbp
        WHERE rusher_player_id = ? AND season IN (SELECT UNNEST(?)) AND week <= ?
    """, [RZ_YARDLINE, RZ_YARDLINE, gsis_id, seasons, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    # RZ target share: player's RZ targets / team total RZ pass plays
    rz_tgt_data = con.execute("""
        WITH player_rz AS (
            SELECT COUNT(*) AS rz_tgt
            FROM pbp
            WHERE receiver_player_id = ? AND yardline_100 <= ?
              AND play_type = 'pass' AND week <= ?
              AND season IN (SELECT UNNEST(?))
        ),
        team_rz AS (
            SELECT COUNT(*) AS team_total
            FROM pbp
            WHERE posteam IN (
                SELECT DISTINCT team FROM weekly_stats
                WHERE player_id = ? AND season IN (SELECT UNNEST(?)) AND season_type='REG'
            )
            AND yardline_100 <= ? AND play_type = 'pass' AND week <= ?
            AND season IN (SELECT UNNEST(?))
        )
        SELECT p.rz_tgt, t.team_total FROM player_rz p, team_rz t
    """, [gsis_id, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons,
          gsis_id, seasons, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons]).fetchdf().iloc[0]

    return pd.Series({
        "rush_yards_over_expected":  float(ngs["ryoe"]) if ngs["ryoe"] is not None and not math.isnan(float(ngs["ryoe"])) else float("nan"),
        "success_rate_outside_zone": _safe_div(pbp["oz_succ"], pbp["oz_plays"]),
        "shotgun_efficiency":        _safe_div(pbp["sg_succ"], pbp["sg_plays"]),
        "rb_target_share":           float(ws2["rb_tgt_share"]) if ws2["rb_tgt_share"] is not None and not math.isnan(float(ws2["rb_tgt_share"])) else float("nan"),
        "snap_share":                snap_share,
        "rz_success_rate":           _safe_div(pbp["rz_succ"], pbp["rz_rush"]),
        "rz_target_share":           _safe_div(rz_tgt_data["rz_tgt"], rz_tgt_data["team_total"]),
    })


def _wr_performance(gsis_id: str, con, seasons: list[int] | None = None) -> pd.Series:
    if seasons is None:
        seasons = _recent_seasons(gsis_id, con)
    if not seasons:
        raise RuntimeError(f"WR {gsis_id}: no weekly_stats found")

    ws = con.execute("""
        SELECT SUM(w.target_share * w.targets) / NULLIF(SUM(w.targets), 0)    AS tgt_share,
               SUM(w.air_yards_share * w.targets) / NULLIF(SUM(w.targets), 0) AS ay_share
        FROM weekly_stats w
        WHERE w.player_id = ? AND w.season IN (SELECT UNNEST(?)) AND w.season_type = 'REG'
          AND w.targets > 0
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    ngs = con.execute("""
        SELECT SUM(n.targets * n.avg_separation) / NULLIF(SUM(n.targets), 0)             AS sep,
               SUM(n.receptions * n.avg_yac_above_expectation) / NULLIF(SUM(n.receptions), 0) AS yac
        FROM ngs_receiving n
        WHERE n.player_gsis_id = ? AND n.season IN (SELECT UNNEST(?)) AND n.season_type = 'REG'
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    motion = con.execute("""
        SELECT
            SUM(CASE WHEN f.is_motion = 1 AND p.play_type = 'pass' THEN 1 ELSE 0 END)  AS motion_tgt,
            SUM(CASE WHEN f.is_motion = 1 AND p.complete_pass = 1 THEN 1 ELSE 0 END)  AS motion_cp
        FROM pbp p
        JOIN ftn f ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
        WHERE p.receiver_player_id = ? AND p.season IN (SELECT UNNEST(?)) AND p.week <= ?
    """, [gsis_id, seasons, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    rz_tgt = con.execute("""
        WITH player_rz AS (
            SELECT COUNT(*) AS rz_tgt
            FROM pbp
            WHERE receiver_player_id = ? AND yardline_100 <= ?
              AND play_type = 'pass' AND week <= ?
              AND season IN (SELECT UNNEST(?))
        ),
        team_rz AS (
            SELECT COUNT(*) AS team_total
            FROM pbp
            WHERE posteam IN (
                SELECT DISTINCT team FROM weekly_stats
                WHERE player_id = ? AND season IN (SELECT UNNEST(?)) AND season_type='REG'
            )
            AND yardline_100 <= ? AND play_type = 'pass' AND week <= ?
            AND season IN (SELECT UNNEST(?))
        )
        SELECT p.rz_tgt, t.team_total FROM player_rz p, team_rz t
    """, [gsis_id, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons,
          gsis_id, seasons, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons]).fetchdf().iloc[0]

    return pd.Series({
        "avg_separation":    float(ngs["sep"]) if ngs["sep"] is not None and not math.isnan(float(ngs["sep"])) else float("nan"),
        "yac_over_expected": float(ngs["yac"]) if ngs["yac"] is not None and not math.isnan(float(ngs["yac"])) else float("nan"),
        "target_share":      float(ws["tgt_share"]) if ws["tgt_share"] is not None and not math.isnan(float(ws["tgt_share"])) else float("nan"),
        "air_yards_share":   float(ws["ay_share"]) if ws["ay_share"] is not None and not math.isnan(float(ws["ay_share"])) else float("nan"),
        "motion_catch_rate": _safe_div(motion["motion_cp"], motion["motion_tgt"]),
        "rz_target_share":   _safe_div(rz_tgt["rz_tgt"], rz_tgt["team_total"]),
    })


def _te_performance(gsis_id: str, pfr_id: str | None, con, seasons: list[int] | None = None) -> pd.Series:
    if seasons is None:
        seasons = _recent_seasons(gsis_id, con)
    if not seasons:
        raise RuntimeError(f"TE {gsis_id}: no weekly_stats found")

    ws = con.execute("""
        SELECT SUM(w.target_share * w.targets) / NULLIF(SUM(w.targets), 0) AS tgt_share,
               SUM(w.receiving_air_yards) AS total_air_yards,
               SUM(w.targets) AS total_targets
        FROM weekly_stats w
        WHERE w.player_id = ? AND w.season IN (SELECT UNNEST(?)) AND w.season_type = 'REG'
          AND w.targets > 0
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    ngs = con.execute("""
        SELECT SUM(n.receptions * n.avg_yac_above_expectation) / NULLIF(SUM(n.receptions), 0) AS yac
        FROM ngs_receiving n
        WHERE n.player_gsis_id = ? AND n.season IN (SELECT UNNEST(?)) AND n.season_type = 'REG'
    """, [gsis_id, seasons]).fetchdf().iloc[0]

    # Snap share from snap_counts (needs pfr_id)
    snap_share = float("nan")
    if pfr_id:
        row = con.execute("""
            SELECT AVG(offense_pct) FROM snap_counts
            WHERE pfr_player_id = ? AND season IN (SELECT UNNEST(?))
        """, [pfr_id, seasons]).fetchone()
        if row and row[0] is not None:
            snap_share = float(row[0])

    rz_tgt = con.execute("""
        WITH player_rz AS (
            SELECT COUNT(*) AS rz_tgt
            FROM pbp
            WHERE receiver_player_id = ? AND yardline_100 <= ?
              AND play_type = 'pass' AND week <= ?
              AND season IN (SELECT UNNEST(?))
        ),
        team_rz AS (
            SELECT COUNT(*) AS team_total
            FROM pbp
            WHERE posteam IN (
                SELECT DISTINCT team FROM weekly_stats
                WHERE player_id = ? AND season IN (SELECT UNNEST(?)) AND season_type='REG'
            )
            AND yardline_100 <= ? AND play_type = 'pass' AND week <= ?
            AND season IN (SELECT UNNEST(?))
        )
        SELECT p.rz_tgt, t.team_total FROM player_rz p, team_rz t
    """, [gsis_id, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons,
          gsis_id, seasons, RZ_YARDLINE, REG_SEASON_MAX_WEEK, seasons]).fetchdf().iloc[0]

    return pd.Series({
        "target_share":         float(ws["tgt_share"]) if ws["tgt_share"] is not None and not math.isnan(float(ws["tgt_share"])) else float("nan"),
        "air_yards_per_target": _safe_div(ws["total_air_yards"], ws["total_targets"]),
        "yac_over_expected":    float(ngs["yac"]) if ngs["yac"] is not None and not math.isnan(float(ngs["yac"])) else float("nan"),
        "snap_share":           snap_share,
        "rz_target_share":      _safe_div(rz_tgt["rz_tgt"], rz_tgt["team_total"]),
    })


_COLLEGE_OFFSET_CACHE: dict[str, dict] = {}


def _college_offsets(position_group: str, con) -> dict:
    """Per-feature (college league mean - NFL league mean) among regular players.
    PFF grades are curved within league, so this is small; it re-centers college
    metrics onto the NFL population before comparison to the (NFL) archetype."""
    if position_group in _COLLEGE_OFFSET_CACHE:
        return _COLLEGE_OFFSET_CACHE[position_group]
    cfg = PFF_SKILL[position_group]
    feats = cfg["features"]
    sel = ", ".join(f"avg({f}) AS {f}" for f in feats)
    nfl = con.execute(f"SELECT {sel} FROM {cfg['report']} WHERE level='nfl' AND {cfg['snap_col']} >= {cfg['min_snaps']}").fetchdf().iloc[0]
    col = con.execute(f"SELECT {sel} FROM {cfg['report']} WHERE level='college' AND {cfg['snap_col']} >= {cfg['min_snaps']}").fetchdf().iloc[0]
    offsets = {f: float(col[f]) - float(nfl[f]) for f in feats}
    _COLLEGE_OFFSET_CACHE[position_group] = offsets
    return offsets


def _pff_performance(player_name: str, position_group: str, con,
                     seasons: list[int] | None = None) -> pd.Series:
    """A player's PFF performance vector for their group.

    Uses the most recent NFL season's row if available; otherwise the final
    college season, re-centered onto the NFL population. `seasons` pins the NFL
    lookup to a reference player's season; None means most recent. Matched by
    normalized name; ordered by snaps so a primary stint wins name collisions.
    """
    cfg = PFF_SKILL[position_group]
    feats = cfg["features"]
    if seasons is None:
        seasons = [2025, 2024]

    def _f(v):
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")

    nkey = norm_name(player_name)
    sel = ", ".join(feats)

    rows = con.execute(f"""
        SELECT {sel} FROM {cfg['report']}
        WHERE {norm_sql("player")} = ? AND level = 'nfl' AND season IN (SELECT UNNEST(?))
        ORDER BY season DESC, {cfg['snap_col']} DESC
    """, [nkey, seasons]).fetchdf()

    if not rows.empty:
        r = rows.iloc[0]
        return pd.Series({f: _f(r[f]) for f in feats})

    col = con.execute(f"""
        SELECT {sel} FROM {cfg['report']}
        WHERE {norm_sql("player")} = ? AND level = 'college'
        ORDER BY {cfg['snap_col']} DESC
    """, [nkey]).fetchdf()

    if col.empty:
        raise RuntimeError(f"{position_group} {player_name}: no NFL or college PFF row")

    offsets = _college_offsets(position_group, con)
    r = col.iloc[0]
    return pd.Series({
        f: (_f(r[f]) - offsets[f]) if not math.isnan(_f(r[f])) else float("nan")
        for f in feats
    })


def get_performance_features(gsis_id: str | None, position_group: str, pfr_id: str | None = None, player_name: str = "", seasons: list[int] | None = None) -> pd.Series:
    """Return the player's performance features matching POSITION_FEATURES[position_group].

    Uses the most recent 2 seasons of data available for the player.
    Sets features to NaN when data exists but a specific metric has zero
    plays in that situation. Raises RuntimeError when the player has no
    usable data at all for this position group.
    """
    if position_group not in PFF_SKILL:
        raise RuntimeError(f"Unknown position_group: {position_group}")

    con = duckdb.connect(DB_PATH)
    try:
        result = _pff_performance(player_name, position_group, con, seasons)
    finally:
        con.close()

    return result


# ---------------------------------------------------------------------------
# Phase B: Scaler and grade
# ---------------------------------------------------------------------------

def build_position_scaler(position_group: str) -> tuple[pd.Series, StandardScaler, pd.Series, StandardScaler]:
    """Return (perf_archetype, perf_scaler, comb_archetype, comb_scaler).

    Both scalers are fit on the reference player population (kubiak_reference_players),
    so distances are in units of natural variation within Kubiak's player set.
    """
    con = duckdb.connect(DB_PATH)

    # Pull archetype vectors
    arch_row = con.execute(
        "SELECT * FROM kubiak_position_archetypes WHERE position_group = ?",
        [position_group],
    ).fetchdf()
    if arch_row.empty:
        raise RuntimeError(f"No archetype found for position_group={position_group}")
    arch_row = arch_row.iloc[0]

    perf_features = POSITION_FEATURES[position_group]
    comb_features = COMBINE_FEATURES_BY_POSITION[position_group]

    perf_archetype = pd.Series({f: arch_row[f] for f in perf_features})

    # Pull reference players and get their features to fit the scalers
    ref_players = con.execute(
        "SELECT player_id, gsis_id, player, position, team, season FROM kubiak_reference_players WHERE position IN (SELECT UNNEST(?))",
        [_positions_for_group(position_group)],
    ).fetchdf()
    con.close()

    comb_matrix = []

    for _, rp in ref_players.iterrows():
        pfr_id = rp["player_id"]  # this is pfr_player_id from snap_counts
        # Combine features (for the secondary athletic model / rookie fallback)
        if pfr_id:
            try:
                cf = get_combine_features(pfr_id, position_group)
                row = [cf.get(f, float("nan")) for f in comb_features]
                if not all(math.isnan(v) for v in row):
                    comb_matrix.append(row)
            except RuntimeError:
                pass

    # Fit the performance scaler on the LEAGUE-WIDE PFF population (regular
    # players, >= min_snaps) so a standard deviation is a stable league unit.
    cfg = PFF_SKILL[position_group]
    with duckdb.connect(DB_PATH) as _c:
        league = _c.execute(
            f"SELECT {', '.join(perf_features)} FROM {cfg['report']} "
            f"WHERE level = 'nfl' AND {cfg['snap_col']} >= {cfg['min_snaps']}"
        ).fetchdf()
    league = league[perf_features].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    perf_scaler = StandardScaler()
    perf_scaler.fit(league.fillna(league.mean()).values)
    perf_scaler.feature_names_in_ = np.array(perf_features)

    # Combine scaler and archetype (mean of reference player measurables)
    if comb_matrix:
        comb_df = pd.DataFrame(comb_matrix, columns=comb_features).dropna(axis=1, how="all")
        comb_scaler = StandardScaler()
        comb_scaler.fit(comb_df.fillna(comb_df.mean()))
        comb_archetype = pd.Series(comb_df.mean())
    else:
        comb_scaler = StandardScaler()
        comb_scaler.fit(np.zeros((2, len(comb_features))))
        comb_archetype = pd.Series({f: float("nan") for f in comb_features})

    return perf_archetype, perf_scaler, comb_archetype, comb_scaler


def _positions_for_group(group: str) -> list[str]:
    return [pos for pos, grp in POSITION_GROUP_MAP.items() if grp == group]


def _compute_grade(player_vec: pd.Series, archetype_vec: pd.Series, scaler: StandardScaler,
                   features: list[str],
                   one_sided_higher_better: set | None = None) -> tuple[float, dict, list[str], list[str]]:
    """Compute grade, per-feature contributions, features used, and features missing.

    Features named in `one_sided_higher_better` are treated as one-directional
    quality scores: a player who is ABOVE the archetype on them contributes zero
    distance (being better than Kubiak's reference is a perfect fit, not a
    deviation), while being below still counts. Used for PFF OL grades.

    Returns: (grade, feature_contributions, features_used, features_missing)
    """
    one_sided_higher_better = one_sided_higher_better or set()
    scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else features

    features_used = []
    features_missing = []
    contributions = {}
    sq_dists = []

    for feat in features:
        p_val = player_vec.get(feat, float("nan"))
        a_val = archetype_vec.get(feat, float("nan"))

        if feat not in scaler_features:
            features_missing.append(feat)
            continue

        feat_idx = scaler_features.index(feat)
        std = scaler.scale_[feat_idx] if scaler.scale_[feat_idx] > 0 else 1.0
        mean = scaler.mean_[feat_idx]

        if p_val is None or math.isnan(float(p_val)):
            features_missing.append(feat)
            continue
        if a_val is None or math.isnan(float(a_val)):
            features_missing.append(feat)
            continue

        p_std = (float(p_val) - mean) / std
        a_std = (float(a_val) - mean) / std
        diff = p_std - a_std
        # One-sided quality feature: exceeding the archetype is not a deviation.
        if feat in one_sided_higher_better and diff > 0:
            diff = 0.0
        sq_dists.append(diff ** 2)
        contributions[feat] = diff
        features_used.append(feat)

    if not sq_dists:
        # No usable features -> the grade is undefined, not zero. Returning NaN
        # lets callers exclude the player from aggregates instead of dragging the
        # unit toward 0 (a missing measurement is not a poor fit).
        return float("nan"), {}, [], features

    # Root-mean-square standardized deviation. Using the MEAN (not the sum) of
    # squared deviations makes the point estimate invariant to the number of
    # available features: a player scored on 2 traits and one scored on 7 sit on
    # the same scale. Missing features therefore no longer bias the grade down;
    # they only widen the confidence interval (see score_player).
    distance = math.sqrt(sum(sq_dists) / len(sq_dists))
    grade = min(100.0, 100.0 * math.exp(-distance / SCALE_FACTOR))
    return grade, contributions, features_used, features_missing


def _blend(perf_grade: float, comb_grade: float,
           w_perf: float, w_comb: float) -> float:
    """Weighted blend of two component grades, NaN-safe.

    If one component is NaN (e.g. a sophomore with no combine record), the blend
    renormalizes onto the component that does exist rather than propagating NaN
    or treating the missing component as a zero.
    """
    parts = []
    if w_perf > 0 and not (perf_grade is None or math.isnan(perf_grade)):
        parts.append((perf_grade, w_perf))
    if w_comb > 0 and not (comb_grade is None or math.isnan(comb_grade)):
        parts.append((comb_grade, w_comb))
    if not parts:
        return float("nan")
    total_w = sum(w for _, w in parts)
    return sum(g * w for g, w in parts) / total_w


def _roster_ht_wt(player_id: str, con) -> tuple[float, float]:
    """Return (height, weight) averaged over a player's roster rows, NaN if absent."""
    row = con.execute(
        "SELECT AVG(height), AVG(weight) FROM rosters WHERE player_id = ?",
        [player_id],
    ).fetchone()
    ht = float(row[0]) if row and row[0] is not None else float("nan")
    wt = float(row[1]) if row and row[1] is not None else float("nan")
    return ht, wt


def score_player(gsis_id: str, position_group: str | None = None) -> dict:
    """Score any player against the Kubiak archetype for their position group.

    If position_group is None, it is looked up from the player's most recent
    roster row (any team, any season). Does NOT require the player to be on
    the 2026 Raiders roster.

    Returns a dict with grade, confidence_interval, feature_contributions,
    features_used, and features_missing.
    """
    con = duckdb.connect(DB_PATH)

    if position_group is None:
        player_row = con.execute("""
            SELECT player_name, pfr_id, position
            FROM rosters
            WHERE player_id = ?
            ORDER BY season DESC, week DESC NULLS LAST
            LIMIT 1
        """, [gsis_id]).fetchone()
        if player_row is None:
            con.close()
            raise RuntimeError(f"Player {gsis_id} not found in rosters table")
        player_name, _, position = player_row
        position_group = POSITION_GROUP_MAP.get(position)
        if position_group is None:
            con.close()
            raise RuntimeError(f"Player {gsis_id} ({player_name}) position '{position}' not in POSITION_GROUP_MAP")
    else:
        player_row = con.execute("""
            SELECT player_name, pfr_id
            FROM rosters
            WHERE player_id = ?
            ORDER BY season DESC, week DESC NULLS LAST
            LIMIT 1
        """, [gsis_id]).fetchone()
        if player_row is None:
            con.close()
            raise RuntimeError(f"Player {gsis_id} not found in rosters table")
        player_name, _ = player_row

    # Resolve pfr_id (rosters may have NULL for OL)
    pfr_id = _resolve_pfr_id(gsis_id, player_name, con)
    con.close()

    if position_group not in POSITION_FEATURES:
        raise RuntimeError(f"Player {gsis_id} ({player_name}) position_group '{position_group}' not in POSITION_FEATURES")

    # Experience + performance weight come from the PFF data (reliable names):
    # veteran -> NFL PFF, rookie -> college PFF, sophomore -> snap-weighted blend.
    with duckdb.connect(DB_PATH) as _c:
        experience, w_perf = classify_pff(player_name, position_group, _c)

    weights = {"performance": w_perf, "combine": 1.0 - w_perf}

    perf_arch, perf_scaler, comb_arch, comb_scaler = build_position_scaler(position_group)

    # Fetch reference player count for CI
    with duckdb.connect(DB_PATH) as con2:
        n_ref = con2.execute(
            "SELECT COUNT(*) FROM kubiak_reference_players WHERE position IN (SELECT UNNEST(?))",
            [_positions_for_group(position_group)],
        ).fetchone()[0]

    perf_features = list(POSITION_FEATURES[position_group])
    comb_features = list(COMBINE_FEATURES_BY_POSITION[position_group])

    grade = 0.0
    contributions = {}
    features_used = []
    features_missing = []

    if weights["performance"] > 0:
        try:
            pf = get_performance_features(gsis_id, position_group, pfr_id, player_name)
        except RuntimeError:
            pf = pd.Series({f: float("nan") for f in perf_features})

        perf_grade, perf_contrib, p_used, p_missing = _compute_grade(
            pf, perf_arch, perf_scaler, perf_features,
            one_sided_higher_better=ONE_SIDED_HIGHER_BETTER.get(position_group),
        )
    else:
        perf_grade, perf_contrib, p_used, p_missing = float("nan"), {}, [], perf_features

    if weights["combine"] > 0:
        try:
            cf = get_combine_features(pfr_id, position_group) if pfr_id else pd.Series({f: float("nan") for f in comb_features})
        except RuntimeError:
            cf = pd.Series({f: float("nan") for f in comb_features})

        # Height/weight always exist in rosters; fall back to them so a player
        # who simply never attended the combine (common for rookie UDFAs) is
        # still scored on ht/wt instead of collapsing to an all-missing 0.
        if "ht" in comb_features or "wt" in comb_features:
            if (cf.get("ht") is None or math.isnan(float(cf.get("ht", float("nan"))))) or \
               (cf.get("wt") is None or math.isnan(float(cf.get("wt", float("nan"))))):
                with duckdb.connect(DB_PATH) as _c:
                    r_ht, r_wt = _roster_ht_wt(gsis_id, _c)
                if "ht" in comb_features and (cf.get("ht") is None or math.isnan(float(cf.get("ht", float("nan"))))):
                    cf["ht"] = r_ht
                if "wt" in comb_features and (cf.get("wt") is None or math.isnan(float(cf.get("wt", float("nan"))))):
                    cf["wt"] = r_wt

        comb_grade, comb_contrib, c_used, c_missing = _compute_grade(
            cf, comb_arch, comb_scaler, comb_features
        )
    else:
        comb_grade, comb_contrib, c_used, c_missing = float("nan"), {}, [], comb_features

    # Blend grades (NaN-safe: a missing component renormalizes onto the other).
    if weights["performance"] > 0 and weights["combine"] > 0:
        grade = _blend(perf_grade, comb_grade, weights["performance"], weights["combine"])
        contributions = {**{f: v * weights["performance"] for f, v in perf_contrib.items()},
                         **{f: v * weights["combine"] for f, v in comb_contrib.items()}}
        features_used = p_used + c_used
        features_missing = p_missing + c_missing
    elif weights["performance"] > 0:
        grade, contributions, features_used, features_missing = perf_grade, perf_contrib, p_used, p_missing
    else:
        grade, contributions, features_used, features_missing = comb_grade, comb_contrib, c_used, c_missing

    # Coverage is reported as a data-completeness diagnostic and feeds the
    # confidence interval, but it is NOT multiplied into the grade: missing a
    # measurement should widen uncertainty, not lower the fit estimate.
    _n_used = len(features_used)
    _n_total = _n_used + len(features_missing)
    coverage = _n_used / _n_total if _n_total > 0 else 0.0
    raw_grade = grade

    # Confidence interval
    total_features = len(perf_features) if weights["performance"] > 0 else len(comb_features)
    n_missing = len([f for f in features_missing if f in (perf_features if weights["performance"] > 0 else comb_features)])
    ci_width = BASE_UNCERTAINTY * math.sqrt(n_missing / max(total_features, 1) + 1 / max(n_ref, 1))
    ci = (max(0.0, grade - 100 * ci_width), min(100.0, grade + 100 * ci_width))

    return {
        "player_id":            gsis_id,
        "player_name":          player_name,
        "position_group":       position_group,
        "experience_bucket":    experience,
        "grade":                round(grade, 2),
        "raw_grade":            round(raw_grade, 2),
        "coverage":             round(coverage, 4),
        "confidence_interval":  (round(ci[0], 2), round(ci[1], 2)),
        "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "features_used":        features_used,
        "features_missing":     features_missing,
        "model":                "performance",
    }


def score_raiders_player(gsis_id: str) -> dict:
    """Score a player who must be on the 2026 LV Raiders roster.

    Looks up the player in rosters where season=2026 and team='LV', extracts
    their position group from that row, then delegates to score_player.
    Raises if the player is not found on the 2026 roster.
    """
    con = duckdb.connect(DB_PATH)
    row = con.execute("""
        SELECT position FROM rosters
        WHERE player_id = ? AND team = 'LV' AND season = 2026
        LIMIT 1
    """, [gsis_id]).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Player {gsis_id} not found in 2026 LV Raiders roster")

    position_group = POSITION_GROUP_MAP.get(row[0])
    if position_group is None:
        raise RuntimeError(f"Player {gsis_id} position '{row[0]}' not in POSITION_GROUP_MAP")

    return score_player(gsis_id, position_group)


# ---------------------------------------------------------------------------
# Phase C: Position-group and overall grades
# ---------------------------------------------------------------------------

def score_position_group(group: str, players_df: pd.DataFrame) -> dict:
    """Score all Raiders players in a position group.

    Free data has no 2026 depth chart or snap share, so we do NOT fabricate a
    single "starter" from an arbitrary ordering (the previous version sorted by
    player name alphabetically). Instead we report two honest numbers:
      - top_grade: the best fit available at the position (the ceiling)
      - unit_grade: the mean over players whose grade is actually defined.
    Players whose grade is undefined (NaN: no usable features for this model)
    are excluded from the mean rather than counted as zero, so depth bodies with
    no data don't drag the unit down.
    """
    scored = []
    for _, row in players_df.iterrows():
        try:
            result = score_raiders_player(row["player_id"])
            result["depth_order"] = row.get("depth_order", 99)
            scored.append(result)
        except RuntimeError as exc:
            # Surface as warning; don't block the group grade
            print(f"  WARNING: could not score {row['player_name']} ({group}): {exc}")

    defined = [r for r in scored if not math.isnan(r["grade"])]

    if not defined:
        return {
            "position_group": group,
            "players": scored,
            "top_grade": float("nan"),
            "unit_grade": float("nan"),
        }

    scored.sort(key=lambda r: (math.isnan(r["grade"]), -r["grade"] if not math.isnan(r["grade"]) else 0))
    top_grade = max(r["grade"] for r in defined)
    unit_grade = sum(r["grade"] for r in defined) / len(defined)

    return {
        "position_group": group,
        "players":        scored,
        "top_grade":      round(top_grade, 2),
        "unit_grade":     round(unit_grade, 2),
    }


def score_offense() -> dict:
    """Score the entire 2026 Raiders offense against the Kubiak archetypes.

    OFFENSE_WEIGHTS reflect analytical consensus for a wide-zone / play-action
    system (QB 30%, OL 25%, WR 15%, RB 20%, TE 10%). These are designed weights,
    not learned from data. The 20% RB figure absorbs both rushing and pass-protection
    contributions since individual pass-pro grades are not available in free data.
    """
    con = duckdb.connect(DB_PATH)
    roster = con.execute("""
        SELECT player_id, player_name, position, depth_chart_position,
               ROW_NUMBER() OVER (PARTITION BY position ORDER BY player_name) AS depth_order
        FROM rosters
        WHERE team = 'LV' AND season = 2026 AND status IN ('ACT', 'RES')
          AND position IN (SELECT UNNEST(?))
        ORDER BY position, player_name
    """, [list(POSITION_GROUP_MAP.keys())]).fetchdf()
    con.close()

    roster["position_group"] = roster["position"].map(POSITION_GROUP_MAP)

    groups = {}
    for group in OFFENSE_WEIGHTS:
        group_players = roster[roster["position_group"] == group].copy()
        if group_players.empty:
            print(f"  No {group} players found on 2026 LV roster")
            continue
        print(f"Scoring {group} ({len(group_players)} players)...")
        groups[group] = score_position_group(group, group_players)

    # Overall grade
    total_weight = 0.0
    overall = 0.0
    for group, result in groups.items():
        unit_grade = result.get("unit_grade", float("nan"))
        if not math.isnan(unit_grade):
            w = OFFENSE_WEIGHTS.get(group, 0)
            overall += unit_grade * w
            total_weight += w

    if total_weight > 0:
        overall = overall / total_weight

    return {
        "overall_grade": round(overall, 2),
        "groups":        groups,
        "computed_at":   datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Physical Fit model
# ---------------------------------------------------------------------------

def get_physical_features(player_id: str, position_group: str,
                           pfr_id: str | None = None,
                           player_name: str = "") -> pd.Series:
    """Return a player's physical features in PHYSICAL_FEATURES[group] order.

    ht/wt from rosters (most recent rows, averaged). Combine measurables from
    combine table joined via rosters.pfr_id, with a fallback to snap_counts
    name lookup for OL players whose rosters.pfr_id is NULL.
    An explicit pfr_id parameter is tried last if both roster joins fail.
    """
    features = PHYSICAL_FEATURES[position_group]
    con = duckdb.connect(DB_PATH)

    row = con.execute("""
        SELECT
            (SELECT AVG(height) FROM rosters WHERE player_id = ?) AS ht,
            (SELECT AVG(weight) FROM rosters WHERE player_id = ?) AS wt,
            MAX(c.forty)      AS forty,
            MAX(c.shuttle)    AS shuttle,
            MAX(c.cone)       AS cone,
            MAX(c.vertical)   AS vertical,
            MAX(c.broad_jump) AS broad_jump
        FROM (SELECT ? AS pid) p
        LEFT JOIN rosters r ON r.player_id = p.pid
        LEFT JOIN combine c ON c.pfr_id = COALESCE(
            r.pfr_id,
            (SELECT pfr_player_id FROM snap_counts WHERE player = r.player_name LIMIT 1),
            ?
        )
    """, [player_id, player_id, player_id, pfr_id]).fetchone()
    con.close()

    if row is None or all(v is None for v in row):
        raise RuntimeError(
            f"Player {player_id} ({player_name}) has no physical data in rosters or combine"
        )

    _all = {"ht": row[0], "wt": row[1], "forty": row[2], "shuttle": row[3],
            "cone": row[4], "vertical": row[5], "broad_jump": row[6]}
    return pd.Series({f: float(_all[f]) if _all[f] is not None else float("nan")
                      for f in features})


# Combine position labels that map into each physical position group.
# Used to fit the scaler on a league-wide population rather than the tiny
# Kubiak reference set (n=2-3), so a "standard deviation" is a stable league
# unit instead of the spread of two players.
COMBINE_POS_FOR_GROUP = {
    "QB": ["QB"],
    "RB": ["RB", "FB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "OL": ["OT", "OG", "C", "G", "OL", "T"],
}


def build_physical_scaler(position_group: str) -> tuple[pd.Series, StandardScaler]:
    """Return the physical archetype mean vector and a fitted StandardScaler.

    The scaler is fit on the LEAGUE-WIDE combine population for the position
    group (every player at those positions across all combine years in the DB),
    not on the handful of Kubiak reference players. This makes each feature's
    standard deviation a stable, league-representative unit: a distance of "1"
    means one league std, regardless of how few reference players Kubiak had.
    It also removes the old n>=3 fragility and the need to hand-tune SCALE_FACTOR
    against a 2-player spread.

    Players with partial combine records still contribute: missing cells are
    imputed with the column (league) mean before fitting, so the std reflects
    the players who actually have each measurement.
    """
    con = duckdb.connect(DB_PATH)

    arch_row = con.execute(
        "SELECT * FROM kubiak_physical_archetypes WHERE position_group = ?",
        [position_group],
    ).fetchdf()
    if arch_row.empty:
        con.close()
        raise RuntimeError(f"No physical archetype found for position_group={position_group}")
    arch_row = arch_row.iloc[0]

    features = PHYSICAL_FEATURES[position_group]
    combine_positions = COMBINE_POS_FOR_GROUP[position_group]

    league = con.execute("""
        SELECT ht, wt, forty, shuttle, cone, vertical, broad_jump
        FROM combine
        WHERE pos IN (SELECT UNNEST(?))
    """, [combine_positions]).fetchdf()
    con.close()

    if league.empty:
        raise RuntimeError(f"No league combine population for {position_group}")

    # Convert 'F-I' height strings to inches.
    league["ht"] = league["ht"].apply(
        lambda v: _height_to_inches(v) if isinstance(v, str) and "-" in v else float("nan")
    )

    matrix = league[features].apply(pd.to_numeric, errors="coerce")
    # Keep rows with at least one present feature; impute the rest with the
    # league mean so partial-combine players still inform the variance.
    matrix = matrix.dropna(how="all")
    means = matrix.mean()
    matrix = matrix.fillna(means)

    if len(matrix) < 10:
        raise RuntimeError(
            f"Only {len(matrix)} league combine records for {position_group}; "
            f"expected a large population. Check combine ingestion."
        )

    scaler = StandardScaler()
    scaler.fit(matrix.values)

    archetype_vec = pd.Series({f: arch_row[f] for f in features})
    return archetype_vec, scaler


def score_player_physical(player_id: str, position_group: str | None = None) -> dict:
    """Score any player's physical fit against the Kubiak physical archetype.

    Mirrors score_player() in shape and return structure. Does NOT require the
    player to be on the 2026 Raiders roster.
    """
    con = duckdb.connect(DB_PATH)
    if position_group is None:
        row = con.execute("""
            SELECT player_name, position FROM rosters
            WHERE player_id = ?
            ORDER BY season DESC, week DESC NULLS LAST LIMIT 1
        """, [player_id]).fetchone()
        if row is None:
            con.close()
            raise RuntimeError(f"Player {player_id} not found in rosters")
        player_name, position = row
        position_group = PHYSICAL_POSITION_MAP.get(position)
        if position_group is None:
            con.close()
            raise RuntimeError(
                f"Player {player_id} ({player_name}) position '{position}' not in PHYSICAL_POSITION_MAP"
            )
    else:
        row = con.execute("""
            SELECT player_name FROM rosters
            WHERE player_id = ?
            ORDER BY season DESC, week DESC NULLS LAST LIMIT 1
        """, [player_id]).fetchone()
        if row is None:
            con.close()
            raise RuntimeError(f"Player {player_id} not found in rosters")
        player_name = row[0]

    pfr_id = _resolve_pfr_id(player_id, player_name, con)
    con.close()

    pf = get_physical_features(player_id, position_group, pfr_id, player_name)
    archetype_vec, scaler = build_physical_scaler(position_group)

    with duckdb.connect(DB_PATH) as con2:
        n_ref = con2.execute(
            "SELECT COUNT(*) FROM kubiak_reference_players WHERE position IN (SELECT UNNEST(?))",
            [[p for p, g in PHYSICAL_POSITION_MAP.items() if g == position_group]],
        ).fetchone()[0]

    features = PHYSICAL_FEATURES[position_group]
    grade, contributions, features_used, features_missing = _compute_grade(
        pf, archetype_vec, scaler, features
    )

    # Coverage is a reported data-completeness diagnostic feeding the confidence
    # interval; it is NOT multiplied into the grade (a missing combine number
    # widens uncertainty, it does not make a player a worse physical fit).
    _n_used = len(features_used)
    _n_total = _n_used + len(features_missing)
    coverage = _n_used / _n_total if _n_total > 0 else 0.0
    raw_grade = grade

    total_features = len(features)
    n_missing = len(features_missing)
    ci_width = BASE_UNCERTAINTY * math.sqrt(
        n_missing / max(total_features, 1) + 1 / max(n_ref, 1)
    )
    ci = (max(0.0, grade - 100 * ci_width), min(100.0, grade + 100 * ci_width))

    return {
        "player_id":             player_id,
        "player_name":           player_name,
        "position_group":        position_group,
        "grade":                 round(grade, 2),
        "raw_grade":             round(raw_grade, 2),
        "coverage":              round(coverage, 4),
        "confidence_interval":   (round(ci[0], 2), round(ci[1], 2)),
        "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "features_used":         features_used,
        "features_missing":      features_missing,
        "model":                 "physical",
    }


def score_raiders_player_physical(player_id: str) -> dict:
    """Look up a player on the 2026 LV roster and score their physical fit."""
    con = duckdb.connect(DB_PATH)
    row = con.execute("""
        SELECT position FROM rosters
        WHERE player_id = ? AND team = 'LV' AND season = 2026
        LIMIT 1
    """, [player_id]).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Player {player_id} not found in 2026 LV Raiders roster")

    position_group = PHYSICAL_POSITION_MAP.get(row[0])
    if position_group is None:
        raise RuntimeError(f"Position '{row[0]}' not in PHYSICAL_POSITION_MAP")

    return score_player_physical(player_id, position_group)


def score_offense_physical() -> dict:
    """Score the entire 2026 Raiders offense on physical fit.

    Returns a structure parallel to score_offense() so the UI can render both.
    """
    con = duckdb.connect(DB_PATH)
    roster = con.execute("""
        SELECT player_id, player_name, position,
               ROW_NUMBER() OVER (PARTITION BY position ORDER BY player_name) AS depth_order
        FROM rosters
        WHERE team = 'LV' AND season = 2026 AND status IN ('ACT', 'RES')
          AND position IN (SELECT UNNEST(?))
        ORDER BY position, player_name
    """, [list(PHYSICAL_POSITION_MAP.keys())]).fetchdf()
    con.close()

    roster["position_group"] = roster["position"].map(PHYSICAL_POSITION_MAP)

    groups = {}
    for group in OFFENSE_WEIGHTS:
        group_players = roster[roster["position_group"] == group].copy()
        if group_players.empty:
            continue

        # This is now the SECONDARY athletic/combine view (the primary scheme-fit
        # grade is the PFF performance model in score_offense). Always the
        # physical/trait scorer.
        scorer = score_raiders_player_physical

        scored = []
        for _, row in group_players.iterrows():
            try:
                result = scorer(row["player_id"])
                result["primary_model"] = "athletic"
                result["depth_order"] = row.get("depth_order", 99)
                scored.append(result)
            except RuntimeError as exc:
                print(f"  WARNING: could not score {row['player_name']} ({group}) [athletic]: {exc}")

        defined = [r for r in scored if not math.isnan(r["grade"])]
        if not defined:
            groups[group] = {
                "position_group": group, "players": scored,
                "top_grade": float("nan"), "unit_grade": float("nan"),
            }
            continue

        scored.sort(key=lambda r: (math.isnan(r["grade"]), -r["grade"] if not math.isnan(r["grade"]) else 0))
        groups[group] = {
            "position_group": group,
            "players":        scored,
            "top_grade":      round(max(r["grade"] for r in defined), 2),
            "unit_grade":     round(sum(r["grade"] for r in defined) / len(defined), 2),
        }

    total_weight = 0.0
    overall = 0.0
    for group, result in groups.items():
        ug = result.get("unit_grade", float("nan"))
        if not math.isnan(ug):
            w = OFFENSE_WEIGHTS.get(group, 0)
            overall += ug * w
            total_weight += w

    if total_weight > 0:
        overall = overall / total_weight

    return {
        "overall_grade": round(overall, 2),
        "groups":        groups,
        "computed_at":   datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Phase D: Persist results
# ---------------------------------------------------------------------------

def write_grades() -> None:
    """Compute both Statistical Similarity and Physical Fit grades, persist to DuckDB."""
    result = score_offense()
    physical_result = score_offense_physical()

    player_rows = []
    for group, grp_result in result["groups"].items():
        for p in grp_result.get("players", []):
            player_rows.append({
                "player_id":           p["player_id"],
                "player_name":         p["player_name"],
                "position_group":      p["position_group"],
                "experience_bucket":   p["experience_bucket"],
                "grade":               p["grade"],
                "raw_grade":           p["raw_grade"],
                "coverage":            p["coverage"],
                "ci_low":              p["confidence_interval"][0],
                "ci_high":             p["confidence_interval"][1],
                "features_used":       len(p["features_used"]),
                "features_missing":    len(p["features_missing"]),
                "missing_feature_names": ", ".join(p["features_missing"]),
            })

    group_rows = []
    for group, grp_result in result["groups"].items():
        group_rows.append({
            "position_group": group,
            "n_players":      len(grp_result.get("players", [])),
            "top_grade":      grp_result.get("top_grade", float("nan")),
            "unit_grade":     grp_result.get("unit_grade", float("nan")),
        })

    players_df = pd.DataFrame(player_rows)
    summary_df = pd.DataFrame([{
        "overall_grade": result["overall_grade"],
        "computed_at":   result["computed_at"],
        **{f"grade_{g}": result["groups"].get(g, {}).get("unit_grade", float("nan"))
           for g in OFFENSE_WEIGHTS},
    }])
    groups_df = pd.DataFrame(group_rows)

    # Flatten physical grades
    physical_player_rows = []
    for group, grp_result in physical_result["groups"].items():
        for p in grp_result.get("players", []):
            physical_player_rows.append({
                "player_id":             p["player_id"],
                "player_name":           p["player_name"],
                "position_group":        p["position_group"],
                "primary_model":         p.get("primary_model", "physical"),
                "grade":                 p["grade"],
                "raw_grade":             p["raw_grade"],
                "coverage":              p["coverage"],
                "ci_low":                p["confidence_interval"][0],
                "ci_high":               p["confidence_interval"][1],
                "features_used":         len(p["features_used"]),
                "features_missing":      len(p["features_missing"]),
                "missing_feature_names": ", ".join(p["features_missing"]),
            })

    physical_players_df = pd.DataFrame(physical_player_rows)
    physical_summary_df = pd.DataFrame([{
        "overall_grade": physical_result["overall_grade"],
        "computed_at":   physical_result["computed_at"],
        **{f"grade_{g}": physical_result["groups"].get(g, {}).get("unit_grade", float("nan"))
           for g in OFFENSE_WEIGHTS},
    }])

    con = duckdb.connect(DB_PATH)
    for table, df in [
        ("raiders_player_grades",           players_df),
        ("raiders_offense_summary",         summary_df),
        ("raiders_position_grades",         groups_df),
        ("raiders_physical_player_grades",  physical_players_df),
        ("raiders_physical_offense_summary", physical_summary_df),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        if not df.empty:
            con.register("df", df)
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.close()


if __name__ == "__main__":
    write_grades()
    print("Raiders offense scored and persisted.")
