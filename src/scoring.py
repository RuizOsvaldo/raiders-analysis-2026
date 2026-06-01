"""Score current Raiders players against the Kubiak archetypes."""

import math
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/raw/nfl.duckdb"

try:
    from archetype import PHYSICAL_FEATURES, PHYSICAL_POSITION_MAP
except ImportError:
    from src.archetype import PHYSICAL_FEATURES, PHYSICAL_POSITION_MAP

# Years of NFL experience determine combine vs performance weighting
# (years counted as completed seasons before 2026)
WEIGHTING_RULES = {
    "rookie":               {"combine": 1.00, "performance": 0.00},
    "sophomore_high_snap":  {"combine": 0.30, "performance": 0.70},
    "sophomore_low_snap":   {"combine": 0.70, "performance": 0.30},
    "veteran":              {"combine": 0.00, "performance": 1.00},
}

SOPHOMORE_SNAP_THRESHOLD = 0.40

# Feature lists per position. Must match columns in kubiak_position_archetypes.
POSITION_FEATURES = {
    "QB": [
        "completion_pct", "epa_per_dropback", "sack_rate", "time_to_throw",
        "completed_air_yards", "play_action_completion_pct",
        "out_of_pocket_completion_pct", "rz_completion_pct", "rz_epa_per_dropback",
    ],
    "RB": [
        "rush_yards_over_expected", "success_rate_outside_zone",
        "shotgun_efficiency", "rb_target_share", "snap_share",
        "rz_success_rate", "rz_target_share",
    ],
    "WR": [
        "avg_separation", "yac_over_expected", "target_share",
        "air_yards_share", "motion_catch_rate", "rz_target_share",
    ],
    "TE": [
        "target_share", "air_yards_per_target", "yac_over_expected",
        "snap_share", "rz_target_share",
    ],
    "OL": [
        "rush_epa_guard", "rush_epa_tackle", "rush_epa_end",
        "team_pressure_rate_allowed", "avg_snap_share",
    ],
}

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
# SCALE_FACTOR=5.0: primary reference QB (Darnold, 60% weight) grades ~76;
# a player 5 sd from archetype grades ~37. Raised from the handoff's 2.0
# because the blended 40/60 archetype cannot be exactly matched by any real
# player, so 2.0 compressed all grades into the 5-50 range.
SCALE_FACTOR = 5.0

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

def classify_player(pfr_id: str | None) -> str:
    """Return 'rookie', 'sophomore_high_snap', 'sophomore_low_snap', or 'veteran'.

    Classification based on snap_counts presence before 2026. Players with no
    pfr_id (common for OL in nflreadpy data) default to 'rookie'.
    """
    if pfr_id is None:
        return "rookie"

    con = duckdb.connect(DB_PATH)
    prior_seasons = con.execute(
        "SELECT DISTINCT season FROM snap_counts WHERE pfr_player_id = ? AND season < 2026 ORDER BY season",
        [pfr_id],
    ).fetchdf()["season"].tolist()

    if len(prior_seasons) == 0:
        con.close()
        return "rookie"

    if len(prior_seasons) == 1:
        rookie_year = prior_seasons[0]
        snap_share = con.execute(
            "SELECT AVG(offense_pct) FROM snap_counts WHERE pfr_player_id = ? AND season = ?",
            [pfr_id, rookie_year],
        ).fetchone()[0]
        con.close()
        if snap_share is None:
            raise RuntimeError(
                f"Player {pfr_id} has 1 prior season but no snap_share. Investigate."
            )
        return "sophomore_high_snap" if snap_share >= SOPHOMORE_SNAP_THRESHOLD else "sophomore_low_snap"

    con.close()
    return "veteran"


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


def _ol_performance(gsis_id: str, pfr_id: str | None, player_name: str, con, seasons: list[int] | None = None) -> pd.Series:
    """OL performance using team-level metrics from games the player participated in.

    Individual OL blocking grades are not available in free data. These
    team-level proxies are computed only for games the player was on the field,
    which is the tightest available signal for individual contribution.
    """
    if seasons is None:
        seasons = [2024, 2025]

    # Resolve games via snap_counts (pfr_player_id or name)
    if pfr_id:
        games = con.execute("""
            SELECT game_id, team, season FROM snap_counts
            WHERE pfr_player_id = ? AND season IN (SELECT UNNEST(?))
              AND position IN ('C','G','T') AND offense_snaps > 0
        """, [pfr_id, seasons]).fetchdf()
        snap_pct_row = con.execute("""
            SELECT AVG(offense_pct) FROM snap_counts
            WHERE pfr_player_id = ? AND season IN (SELECT UNNEST(?))
              AND position IN ('C','G','T')
        """, [pfr_id, seasons]).fetchone()
    else:
        games = con.execute("""
            SELECT game_id, team, season FROM snap_counts
            WHERE player = ? AND season IN (SELECT UNNEST(?))
              AND position IN ('C','G','T') AND offense_snaps > 0
        """, [player_name, seasons]).fetchdf()
        snap_pct_row = con.execute("""
            SELECT AVG(offense_pct) FROM snap_counts
            WHERE player = ? AND season IN (SELECT UNNEST(?))
              AND position IN ('C','G','T')
        """, [player_name, seasons]).fetchone()

    if games.empty:
        # Last resort: use rosters to find prior team(s), compute full-season team stats
        team_seasons = con.execute("""
            SELECT DISTINCT team, season FROM rosters
            WHERE player_id = ? AND season IN (SELECT UNNEST(?))
        """, [gsis_id, seasons]).fetchdf()
        if team_seasons.empty:
            raise RuntimeError(
                f"OL {gsis_id} ({player_name}): no snap data or roster history in 2024-2025"
            )
        # Use full-season team data (less precise but better than nothing)
        cond = " OR ".join(
            f"(posteam='{r.team}' AND season={r.season})"
            for _, r in team_seasons.iterrows()
        )
        game_filter = f"({cond})"
        snap_share = float("nan")
    else:
        game_ids = games["game_id"].tolist()
        teams = games["team"].unique().tolist()
        game_filter = f"game_id IN ({', '.join(repr(g) for g in game_ids)}) AND posteam IN ({', '.join(repr(t) for t in teams)})"
        snap_share = float(snap_pct_row[0]) if snap_pct_row and snap_pct_row[0] is not None else float("nan")

    rush_epa = con.execute(f"""
        SELECT run_gap, AVG(epa) AS avg_epa, COUNT(*) AS n
        FROM pbp
        WHERE {game_filter} AND play_type='run' AND run_gap IS NOT NULL AND week <= {REG_SEASON_MAX_WEEK}
        GROUP BY run_gap
    """).fetchdf()

    epa_by_gap = {row["run_gap"]: row["avg_epa"] for _, row in rush_epa.iterrows()}

    pressure = con.execute(f"""
        SELECT SUM(CAST(was_pressure AS DOUBLE)) AS pressures,
               COUNT(*) AS dropbacks
        FROM pbp
        WHERE {game_filter} AND qb_dropback = 1 AND week <= {REG_SEASON_MAX_WEEK}
    """).fetchdf().iloc[0]

    return pd.Series({
        "rush_epa_guard":             epa_by_gap.get("guard", float("nan")),
        "rush_epa_tackle":            epa_by_gap.get("tackle", float("nan")),
        "rush_epa_end":               epa_by_gap.get("end", float("nan")),
        "team_pressure_rate_allowed": _safe_div(pressure["pressures"], pressure["dropbacks"]),
        "avg_snap_share":             snap_share,
    })


def get_performance_features(gsis_id: str | None, position_group: str, pfr_id: str | None = None, player_name: str = "", seasons: list[int] | None = None) -> pd.Series:
    """Return the player's performance features matching POSITION_FEATURES[position_group].

    Uses the most recent 2 seasons of data available for the player.
    Sets features to NaN when data exists but a specific metric has zero
    plays in that situation. Raises RuntimeError when the player has no
    usable data at all for this position group.
    """
    if position_group != "OL" and gsis_id is None:
        raise RuntimeError(f"gsis_id required for {position_group} performance features")

    con = duckdb.connect(DB_PATH)
    try:
        if position_group == "QB":
            result = _qb_performance(gsis_id, con, seasons)
        elif position_group == "RB":
            result = _rb_performance(gsis_id, pfr_id, con, seasons)
        elif position_group == "WR":
            result = _wr_performance(gsis_id, con, seasons)
        elif position_group == "TE":
            result = _te_performance(gsis_id, pfr_id, con, seasons)
        elif position_group == "OL":
            result = _ol_performance(gsis_id, pfr_id, player_name, con, seasons)
        else:
            raise RuntimeError(f"Unknown position_group: {position_group}")
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

    perf_matrix = []
    comb_matrix = []
    comb_archetype_rows = []

    for _, rp in ref_players.iterrows():
        gsis_id = None if pd.isna(rp["gsis_id"]) else rp["gsis_id"]
        pfr_id = rp["player_id"]  # this is pfr_player_id from snap_counts
        player_name = rp["player"]

        # OL reference players have NULL gsis_id (OL pfr_id not in nflreadpy rosters).
        # They can still be scored via pfr_id in snap_counts.
        if gsis_id is None and not pfr_id:
            continue

        # Use only the player's reference season for the scaler
        # (avoids mixing pre-Kubiak seasons into the reference distribution)
        _ref_map = {"NO": 2024, "SEA": 2025}
        ref_season = [_ref_map[rp["team"]]] if rp["team"] in _ref_map else None

        # Performance features
        try:
            pf = get_performance_features(gsis_id, position_group, pfr_id, player_name, ref_season)
            row = [pf.get(f, float("nan")) for f in perf_features]
            if not all(math.isnan(v) for v in row):
                perf_matrix.append(row)
        except RuntimeError:
            pass

        # Combine features
        if pfr_id:
            try:
                cf = get_combine_features(pfr_id, position_group)
                row = [cf.get(f, float("nan")) for f in comb_features]
                if not all(math.isnan(v) for v in row):
                    comb_matrix.append(row)
                    comb_archetype_rows.append(row)
            except RuntimeError:
                pass

    if not perf_matrix:
        raise RuntimeError(f"No performance data for any reference player in {position_group}")

    perf_df = pd.DataFrame(perf_matrix, columns=perf_features).dropna(axis=1, how="all")
    perf_scaler = StandardScaler()
    perf_scaler.fit(perf_df.fillna(perf_df.mean()))

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
                   features: list[str]) -> tuple[float, dict, list[str], list[str]]:
    """Compute grade, per-feature contributions, features used, and features missing.

    Returns: (grade, feature_contributions, features_used, features_missing)
    """
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
        sq_dists.append(diff ** 2)
        contributions[feat] = diff
        features_used.append(feat)

    if not sq_dists:
        return 0.0, {}, [], features

    distance = math.sqrt(sum(sq_dists))
    grade = min(100.0, 100.0 * math.exp(-distance / SCALE_FACTOR))
    return grade, contributions, features_used, features_missing


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

    experience = classify_player(pfr_id)
    weights = WEIGHTING_RULES[experience]

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
            pf, perf_arch, perf_scaler, perf_features
        )
    else:
        perf_grade, perf_contrib, p_used, p_missing = 0.0, {}, [], perf_features

    if weights["combine"] > 0:
        try:
            cf = get_combine_features(pfr_id, position_group) if pfr_id else pd.Series({f: float("nan") for f in comb_features})
        except RuntimeError:
            cf = pd.Series({f: float("nan") for f in comb_features})

        comb_grade, comb_contrib, c_used, c_missing = _compute_grade(
            cf, comb_arch, comb_scaler, comb_features
        )
    else:
        comb_grade, comb_contrib, c_used, c_missing = 0.0, {}, [], comb_features

    # Blend grades
    if weights["performance"] > 0 and weights["combine"] > 0:
        grade = weights["performance"] * perf_grade + weights["combine"] * comb_grade
        contributions = {**{f: v * weights["performance"] for f, v in perf_contrib.items()},
                         **{f: v * weights["combine"] for f, v in comb_contrib.items()}}
        features_used = p_used + c_used
        features_missing = p_missing + c_missing
    elif weights["performance"] > 0:
        grade, contributions, features_used, features_missing = perf_grade, perf_contrib, p_used, p_missing
    else:
        grade, contributions, features_used, features_missing = comb_grade, comb_contrib, c_used, c_missing

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
        "confidence_interval":  (round(ci[0], 2), round(ci[1], 2)),
        "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "features_used":        features_used,
        "features_missing":     features_missing,
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

    Starter is identified by depth_chart_position order (v1 limitation:
    no 2026 snap data available pre-season, so depth chart is the proxy).
    Unit grade is snap-share-weighted across the rotation, but since 2026
    snap shares don't exist yet, we weight equally among scored players.
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

    if not scored:
        return {
            "position_group": group,
            "players": [],
            "starter_grade": float("nan"),
            "unit_grade": float("nan"),
        }

    scored.sort(key=lambda r: r.get("depth_order", 99))
    starter_grade = scored[0]["grade"]
    unit_grade = sum(r["grade"] for r in scored) / len(scored)

    return {
        "position_group": group,
        "players":        scored,
        "starter_grade":  round(starter_grade, 2),
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
        WHERE team = 'LV' AND season = 2026 AND status IN ('ACT', 'UFA')
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


def build_physical_scaler(position_group: str) -> tuple[pd.Series, StandardScaler]:
    """Return the physical archetype mean vector and a fitted StandardScaler.

    Scaler is fit on reference players' physical features for the group.
    Only players with complete records for all group features are used.
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

    positions_in_group = [p for p, g in PHYSICAL_POSITION_MAP.items() if g == position_group]
    ref_players = con.execute(
        "SELECT player_id, gsis_id, player FROM kubiak_reference_players WHERE position IN (SELECT UNNEST(?))",
        [positions_in_group],
    ).fetchdf()

    con.register("ref_df", ref_players)
    physical = con.execute("""
        SELECT
            m.player_id,
            AVG(r.height)     AS ht,
            AVG(r.weight)     AS wt,
            MAX(c.forty)      AS forty,
            MAX(c.shuttle)    AS shuttle,
            MAX(c.cone)       AS cone,
            MAX(c.vertical)   AS vertical,
            MAX(c.broad_jump) AS broad_jump
        FROM ref_df m
        LEFT JOIN rosters r ON (
            (m.gsis_id IS NOT NULL AND r.player_id = m.gsis_id)
            OR (m.gsis_id IS NULL AND r.player_name = m.player)
        )
        LEFT JOIN combine c ON c.pfr_id = m.player_id
        GROUP BY m.player_id
    """).fetchdf()
    con.close()

    features = PHYSICAL_FEATURES[position_group]
    feature_matrix = []
    for _, row in physical.iterrows():
        vals = [row.get(f) for f in features]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            continue
        feature_matrix.append([float(v) for v in vals])

    if len(feature_matrix) < 3:
        raise RuntimeError(
            f"Only {len(feature_matrix)} complete physical records for {position_group}. "
            f"Need at least 3 to fit a scaler."
        )

    scaler = StandardScaler()
    scaler.fit(feature_matrix)

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
        WHERE team = 'LV' AND season = 2026 AND status IN ('ACT', 'UFA')
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

        scored = []
        for _, row in group_players.iterrows():
            try:
                result = score_raiders_player_physical(row["player_id"])
                result["depth_order"] = row.get("depth_order", 99)
                scored.append(result)
            except RuntimeError as exc:
                print(f"  WARNING: could not score {row['player_name']} ({group}) physical: {exc}")

        if not scored:
            groups[group] = {
                "position_group": group, "players": [],
                "starter_grade": float("nan"), "unit_grade": float("nan"),
            }
            continue

        scored.sort(key=lambda r: r.get("depth_order", 99))
        groups[group] = {
            "position_group": group,
            "players":        scored,
            "starter_grade":  round(scored[0]["grade"], 2),
            "unit_grade":     round(sum(r["grade"] for r in scored) / len(scored), 2),
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
            "starter_grade":  grp_result.get("starter_grade", float("nan")),
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
                "grade":                 p["grade"],
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
