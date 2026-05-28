"""Build Kubiak scheme profile and position archetypes from 2024 Saints and 2025 Seahawks."""

import duckdb
import pandas as pd

DB_PATH = "data/raw/nfl.duckdb"

# Reference teams, weighted per handoff design
REFERENCE = {
    "NO": {"season": 2024, "weight": 0.40},
    "SEA": {"season": 2025, "weight": 0.60},
}

# Inclusion floor: a player must have at least 20% offensive snap share
# in at least 4 games of the season to feed the archetype
MIN_SNAP_SHARE = 0.20
MIN_GAMES = 4

# Red zone definition
RZ_YARDLINE = 20  # yardline_100 <= 20 is red zone

# Regular season week ceiling (NFL has 18 regular season weeks since 2021)
REG_SEASON_MAX_WEEK = 18

# OL positions as labeled in snap_counts
OL_POSITIONS = ("C", "G", "T")


def get_reference_players() -> pd.DataFrame:
    """Return players who met the snap floor on the reference Kubiak teams.

    Output columns: player_id, player, position, team, season, games_qualified,
                    avg_snap_share, total_offensive_snaps, weight, gsis_id
    """
    con = duckdb.connect(DB_PATH)
    query = """
        WITH per_game AS (
            SELECT
                pfr_player_id AS player_id,
                player,
                position,
                team,
                season,
                game_id,
                offense_pct AS snap_share,
                offense_snaps AS snaps
            FROM snap_counts
            WHERE (team = 'NO' AND season = 2024)
               OR (team = 'SEA' AND season = 2025)
        ),
        qualifying AS (
            SELECT
                player_id,
                player,
                position,
                team,
                season,
                COUNT(*) FILTER (WHERE snap_share >= ?) AS games_qualified,
                AVG(snap_share) AS avg_snap_share,
                SUM(snaps) AS total_offensive_snaps
            FROM per_game
            GROUP BY player_id, player, position, team, season
        ),
        id_map AS (
            SELECT pfr_id,
                   FIRST(player_id ORDER BY season DESC, week DESC NULLS LAST) AS gsis_id
            FROM rosters
            WHERE pfr_id IS NOT NULL AND player_id IS NOT NULL
            GROUP BY pfr_id
        )
        SELECT q.*, i.gsis_id
        FROM qualifying q
        LEFT JOIN id_map i ON q.player_id = i.pfr_id
        WHERE q.games_qualified >= ?
    """
    df = con.execute(query, [MIN_SNAP_SHARE, MIN_GAMES]).fetchdf()
    con.close()

    if df.empty:
        raise RuntimeError("No reference players passed the snap floor. Check snap_counts data.")

    df["weight"] = df["team"].map(lambda t: REFERENCE[t]["weight"])
    return df


def build_scheme_profile() -> pd.DataFrame:
    """Build Kubiak's scheme profile from PBP + FTN, weighted 40/60.

    Output: one row per red_zone value (0/1), with columns for each metric.
    Columns include: play_action_rate, motion_rate, shotgun_rate, no_huddle_rate,
                     screen_rate, rpo_rate, pass_rate, avg_epa, play_count
    """
    con = duckdb.connect(DB_PATH)
    query = """
        WITH plays AS (
            SELECT
                p.season,
                p.posteam AS team,
                p.play_id,
                p.epa,
                p.shotgun,
                p.no_huddle,
                p.pass,
                p.rush,
                CASE WHEN p.yardline_100 <= ? THEN 1 ELSE 0 END AS red_zone,
                f.is_play_action,
                f.is_motion,
                f.is_screen_pass,
                f.is_rpo
            FROM pbp p
            LEFT JOIN ftn f
                ON p.game_id = f.nflverse_game_id
                AND p.play_id = f.nflverse_play_id
            WHERE
                ((p.posteam = 'NO' AND p.season = 2024)
                 OR (p.posteam = 'SEA' AND p.season = 2025))
                AND p.play_type IN ('pass', 'run')
                AND p.week <= ?
        )
        SELECT
            team,
            season,
            red_zone,
            COUNT(*) AS play_count,
            AVG(epa) AS avg_epa,
            AVG(CAST(pass AS DOUBLE)) AS pass_rate,
            AVG(CAST(shotgun AS DOUBLE)) AS shotgun_rate,
            AVG(CAST(no_huddle AS DOUBLE)) AS no_huddle_rate,
            AVG(CAST(is_play_action AS DOUBLE)) AS play_action_rate,
            AVG(CAST(is_motion AS DOUBLE)) AS motion_rate,
            AVG(CAST(is_screen_pass AS DOUBLE)) AS screen_rate,
            AVG(CAST(is_rpo AS DOUBLE)) AS rpo_rate
        FROM plays
        GROUP BY team, season, red_zone
    """
    raw = con.execute(query, [RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf()
    con.close()

    if raw.empty:
        raise RuntimeError("scheme profile query returned empty rows")

    # Apply 40/60 weighting per team
    raw["weight"] = raw["team"].map(lambda t: REFERENCE[t]["weight"])

    metric_cols = [
        "avg_epa", "pass_rate", "shotgun_rate", "no_huddle_rate",
        "play_action_rate", "motion_rate", "screen_rate", "rpo_rate"
    ]

    rows = []
    for rz_value in [0, 1]:
        subset = raw[raw["red_zone"] == rz_value]
        if subset.empty:
            raise RuntimeError(f"scheme profile missing red_zone={rz_value}")
        row = {"red_zone": rz_value, "total_plays": int(subset["play_count"].sum())}
        for col in metric_cols:
            row[col] = (subset[col] * subset["weight"]).sum() / subset["weight"].sum()
        rows.append(row)

    return pd.DataFrame(rows)


def _qb_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build QB archetype feature vector from reference players.

    Features: completion_pct, epa_per_dropback, sack_rate, time_to_throw,
              completed_air_yards, play_action_completion_pct,
              out_of_pocket_completion_pct, rz_completion_pct, rz_epa_per_dropback
    """
    qb_refs = ref_players[ref_players["position"] == "QB"].copy()
    if qb_refs.empty:
        raise RuntimeError("No QB reference players found")

    qb_weight = qb_refs[["gsis_id", "season", "total_offensive_snaps", "weight"]].dropna(subset=["gsis_id"])
    if qb_weight.empty:
        raise RuntimeError("QB reference players have no gsis_id mapping. Check rosters table.")

    con = duckdb.connect(DB_PATH)
    con.register("qb_ref", qb_weight)

    # Season totals from weekly_stats, weighted by snaps * team_weight
    ws = con.execute("""
        SELECT
            SUM(w.completions * r.total_offensive_snaps * r.weight)   AS wt_completions,
            SUM(w.attempts   * r.total_offensive_snaps * r.weight)   AS wt_attempts,
            SUM(w.passing_epa * r.total_offensive_snaps * r.weight)  AS wt_passing_epa,
            SUM(w.sacks_suffered * r.total_offensive_snaps * r.weight) AS wt_sacks,
            SUM(r.total_offensive_snaps * r.weight)                  AS total_weight
        FROM weekly_stats w
        JOIN qb_ref r ON w.player_id = r.gsis_id AND w.season = r.season
        WHERE w.season_type = 'REG'
    """).fetchdf().iloc[0]

    total_weight = ws["total_weight"]
    if total_weight == 0:
        raise RuntimeError("QB archetype: no weekly_stats data joined to reference QBs")

    total_dropbacks_wt = ws["wt_attempts"] + ws["wt_sacks"]

    # NGS passing: time_to_throw and completed_air_yards, weighted by attempts
    ngs = con.execute("""
        SELECT
            SUM(n.attempts * n.avg_time_to_throw * r.weight)          AS wt_ttt_num,
            SUM(n.attempts * r.weight)                                 AS wt_att_ttt,
            SUM(n.completions * n.avg_completed_air_yards * r.weight) AS wt_cay_num,
            SUM(n.completions * r.weight)                              AS wt_comp_cay
        FROM ngs_passing n
        JOIN qb_ref r ON n.player_gsis_id = r.gsis_id AND n.season = r.season
        WHERE n.season_type = 'REG'
    """).fetchdf().iloc[0]

    # PBP+FTN splits: play-action, OOP, red zone. Weight by team_weight per play.
    spl = con.execute("""
        SELECT
            SUM(CASE WHEN f.is_play_action = 1 AND p.qb_dropback = 1 THEN r.weight ELSE 0 END)  AS pa_db_wt,
            SUM(CASE WHEN f.is_play_action = 1 AND p.complete_pass = 1 THEN r.weight ELSE 0 END) AS pa_cp_wt,
            SUM(CASE WHEN f.is_qb_out_of_pocket = 1 AND p.qb_dropback = 1 THEN r.weight ELSE 0 END)  AS oop_db_wt,
            SUM(CASE WHEN f.is_qb_out_of_pocket = 1 AND p.complete_pass = 1 THEN r.weight ELSE 0 END) AS oop_cp_wt,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.qb_dropback = 1 THEN r.weight ELSE 0 END)         AS rz_db_wt,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.complete_pass = 1 THEN r.weight ELSE 0 END)        AS rz_cp_wt,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.qb_dropback = 1 THEN p.epa * r.weight ELSE 0 END) AS rz_epa_wt
        FROM pbp p
        JOIN ftn f ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
        JOIN qb_ref r ON p.passer_player_id = r.gsis_id AND p.season = r.season
        WHERE p.week <= ?
    """, [RZ_YARDLINE, RZ_YARDLINE, RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    con.close()

    def _safe_div(num, denom):
        return float(num / denom) if denom and denom > 0 else None

    return pd.Series({
        "completion_pct":               _safe_div(ws["wt_completions"], ws["wt_attempts"]),
        "epa_per_dropback":             _safe_div(ws["wt_passing_epa"], total_dropbacks_wt),
        "sack_rate":                    _safe_div(ws["wt_sacks"], total_dropbacks_wt),
        "time_to_throw":                _safe_div(ngs["wt_ttt_num"], ngs["wt_att_ttt"]),
        "completed_air_yards":          _safe_div(ngs["wt_cay_num"], ngs["wt_comp_cay"]),
        "play_action_completion_pct":   _safe_div(spl["pa_cp_wt"], spl["pa_db_wt"]),
        "out_of_pocket_completion_pct": _safe_div(spl["oop_cp_wt"], spl["oop_db_wt"]),
        "rz_completion_pct":            _safe_div(spl["rz_cp_wt"], spl["rz_db_wt"]),
        "rz_epa_per_dropback":          _safe_div(spl["rz_epa_wt"], spl["rz_db_wt"]),
    })


def _rb_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build RB archetype feature vector from reference players.

    Features: rush_yards_over_expected, success_rate_outside_zone,
              shotgun_efficiency, rb_target_share, snap_share,
              rz_success_rate, rz_target_share
    """
    rb_refs = ref_players[ref_players["position"] == "RB"].copy()
    if rb_refs.empty:
        raise RuntimeError("No RB reference players found")

    rb_weight = rb_refs[["gsis_id", "season", "total_offensive_snaps", "weight"]].dropna(subset=["gsis_id"])
    if rb_weight.empty:
        raise RuntimeError("RB reference players have no gsis_id mapping. Check rosters table.")

    con = duckdb.connect(DB_PATH)
    con.register("rb_ref", rb_weight)

    # RYOE per attempt from ngs_rushing, weighted by rush_attempts * team_weight
    ngs = con.execute("""
        SELECT
            SUM(n.rush_attempts * n.rush_yards_over_expected_per_att * r.weight) AS wt_ryoe_num,
            SUM(n.rush_attempts * r.weight)                                        AS wt_rush_att
        FROM ngs_rushing n
        JOIN rb_ref r ON n.player_gsis_id = r.gsis_id AND n.season = r.season
        WHERE n.season_type = 'REG'
    """).fetchdf().iloc[0]

    # Target share from weekly_stats (already team-share), weighted by snaps * team_weight
    ws = con.execute("""
        SELECT
            SUM(w.target_share * r.total_offensive_snaps * r.weight) AS wt_target_share,
            SUM(r.total_offensive_snaps * r.weight)                  AS total_weight
        FROM weekly_stats w
        JOIN rb_ref r ON w.player_id = r.gsis_id AND w.season = r.season
        WHERE w.season_type = 'REG' AND w.carries + w.targets > 0
    """).fetchdf().iloc[0]

    # PBP (rusher join): outside-zone success, shotgun efficiency, RZ rush success
    pbp = con.execute("""
        SELECT
            SUM(CASE WHEN p.play_type='run' AND p.run_gap='end' THEN r.weight ELSE 0 END)                    AS oz_plays_wt,
            SUM(CASE WHEN p.play_type='run' AND p.run_gap='end' AND p.success=1 THEN r.weight ELSE 0 END)    AS oz_success_wt,
            SUM(CASE WHEN p.play_type='run' AND p.shotgun=1 THEN r.weight ELSE 0 END)                        AS sg_plays_wt,
            SUM(CASE WHEN p.play_type='run' AND p.shotgun=1 AND p.success=1 THEN r.weight ELSE 0 END)        AS sg_success_wt,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.play_type='run' THEN r.weight ELSE 0 END)                AS rz_rush_wt,
            SUM(CASE WHEN p.yardline_100 <= ? AND p.play_type='run' AND p.success=1 THEN r.weight ELSE 0 END) AS rz_rush_success_wt
        FROM pbp p
        JOIN rb_ref r ON p.rusher_player_id = r.gsis_id AND p.season = r.season
        WHERE p.week <= ?
    """, [RZ_YARDLINE, RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    # RZ target share: per-player share (receiver_player_id), then weighted average
    rz_share = con.execute("""
        WITH player_rz AS (
            SELECT p.receiver_player_id AS gsis_id, p.season,
                   p.posteam,
                   COUNT(*) AS rz_targets
            FROM pbp p
            WHERE p.play_type = 'pass' AND p.yardline_100 <= ?
                AND ((p.posteam = 'NO' AND p.season = 2024) OR (p.posteam = 'SEA' AND p.season = 2025))
                AND p.week <= ? AND p.receiver_player_id IS NOT NULL
            GROUP BY p.receiver_player_id, p.season, p.posteam
        ),
        team_rz AS (
            SELECT posteam, season, COUNT(*) AS team_total
            FROM pbp
            WHERE play_type = 'pass' AND yardline_100 <= ?
                AND ((posteam = 'NO' AND season = 2024) OR (posteam = 'SEA' AND season = 2025))
                AND week <= ?
            GROUP BY posteam, season
        )
        SELECT
            SUM(COALESCE(pr.rz_targets, 0)::DOUBLE / tt.team_total
                * r.total_offensive_snaps * r.weight) AS wt_rz_share,
            SUM(r.total_offensive_snaps * r.weight)   AS total_weight
        FROM rb_ref r
        JOIN team_rz tt ON r.season = tt.season
            AND tt.posteam = CASE WHEN r.season = 2024 THEN 'NO' ELSE 'SEA' END
        LEFT JOIN player_rz pr ON pr.gsis_id = r.gsis_id AND pr.season = r.season
    """, [RZ_YARDLINE, REG_SEASON_MAX_WEEK, RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    # Snap share from the ref DataFrame (already computed in get_reference_players)
    snap_share = (rb_refs["avg_snap_share"] * rb_refs["total_offensive_snaps"] * rb_refs["weight"]).sum() / \
                 (rb_refs["total_offensive_snaps"] * rb_refs["weight"]).sum()

    con.close()

    def _safe_div(num, denom):
        return float(num / denom) if denom and denom > 0 else None

    return pd.Series({
        "rush_yards_over_expected":  _safe_div(ngs["wt_ryoe_num"], ngs["wt_rush_att"]),
        "success_rate_outside_zone": _safe_div(pbp["oz_success_wt"], pbp["oz_plays_wt"]),
        "shotgun_efficiency":        _safe_div(pbp["sg_success_wt"], pbp["sg_plays_wt"]),
        "rb_target_share":           _safe_div(ws["wt_target_share"], ws["total_weight"]),
        "snap_share":                float(snap_share) if snap_share is not None else None,
        "rz_success_rate":           _safe_div(pbp["rz_rush_success_wt"], pbp["rz_rush_wt"]),
        "rz_target_share":           _safe_div(rz_share["wt_rz_share"], rz_share["total_weight"]),
    })


def _wr_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build WR archetype feature vector from reference players.

    Features: avg_separation, yac_over_expected, target_share,
              air_yards_share, motion_catch_rate, rz_target_share
    """
    wr_refs = ref_players[ref_players["position"] == "WR"].copy()
    if wr_refs.empty:
        raise RuntimeError("No WR reference players found")

    wr_weight = wr_refs[["gsis_id", "season", "total_offensive_snaps", "weight"]].dropna(subset=["gsis_id"])
    if wr_weight.empty:
        raise RuntimeError("WR reference players have no gsis_id mapping. Check rosters table.")

    con = duckdb.connect(DB_PATH)
    con.register("wr_ref", wr_weight)

    # NGS receiving: separation and YAC above expected, weighted by targets
    ngs = con.execute("""
        SELECT
            SUM(n.targets * n.avg_separation * r.weight)          AS wt_sep_num,
            SUM(n.targets * r.weight)                              AS wt_tgt_sep,
            SUM(n.receptions * n.avg_yac_above_expectation * r.weight) AS wt_yac_num,
            SUM(n.receptions * r.weight)                           AS wt_rec_yac
        FROM ngs_receiving n
        JOIN wr_ref r ON n.player_gsis_id = r.gsis_id AND n.season = r.season
        WHERE n.season_type = 'REG'
    """).fetchdf().iloc[0]

    # Weekly stats: target_share and air_yards_share, weighted by snaps * team_weight
    ws = con.execute("""
        SELECT
            SUM(w.target_share    * r.total_offensive_snaps * r.weight) AS wt_tgt_share,
            SUM(w.air_yards_share * r.total_offensive_snaps * r.weight) AS wt_ay_share,
            SUM(r.total_offensive_snaps * r.weight)                      AS total_weight
        FROM weekly_stats w
        JOIN wr_ref r ON w.player_id = r.gsis_id AND w.season = r.season
        WHERE w.season_type = 'REG' AND w.targets > 0
    """).fetchdf().iloc[0]

    # PBP+FTN: motion catch rate
    motion = con.execute("""
        SELECT
            SUM(CASE WHEN f.is_motion = 1 AND p.play_type = 'pass' THEN r.weight ELSE 0 END) AS motion_tgt_wt,
            SUM(CASE WHEN f.is_motion = 1 AND p.complete_pass = 1 THEN r.weight ELSE 0 END)  AS motion_catch_wt
        FROM pbp p
        JOIN ftn f ON p.game_id = f.nflverse_game_id AND p.play_id = f.nflverse_play_id
        JOIN wr_ref r ON p.receiver_player_id = r.gsis_id AND p.season = r.season
        WHERE p.week <= ?
    """, [REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    # RZ target share: per-player share, then weighted average (includes zeros)
    rz_share = con.execute("""
        WITH player_rz AS (
            SELECT p.receiver_player_id AS gsis_id, p.season, p.posteam,
                   COUNT(*) AS rz_targets
            FROM pbp p
            WHERE p.play_type = 'pass' AND p.yardline_100 <= ?
                AND ((p.posteam = 'NO' AND p.season = 2024) OR (p.posteam = 'SEA' AND p.season = 2025))
                AND p.week <= ? AND p.receiver_player_id IS NOT NULL
            GROUP BY p.receiver_player_id, p.season, p.posteam
        ),
        team_rz AS (
            SELECT posteam, season, COUNT(*) AS team_total
            FROM pbp
            WHERE play_type = 'pass' AND yardline_100 <= ?
                AND ((posteam = 'NO' AND season = 2024) OR (posteam = 'SEA' AND season = 2025))
                AND week <= ?
            GROUP BY posteam, season
        )
        SELECT
            SUM(COALESCE(pr.rz_targets, 0)::DOUBLE / tt.team_total
                * r.total_offensive_snaps * r.weight) AS wt_rz_share,
            SUM(r.total_offensive_snaps * r.weight)   AS total_weight
        FROM wr_ref r
        JOIN team_rz tt ON r.season = tt.season
            AND tt.posteam = CASE WHEN r.season = 2024 THEN 'NO' ELSE 'SEA' END
        LEFT JOIN player_rz pr ON pr.gsis_id = r.gsis_id AND pr.season = r.season
    """, [RZ_YARDLINE, REG_SEASON_MAX_WEEK, RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    con.close()

    def _safe_div(num, denom):
        return float(num / denom) if denom and denom > 0 else None

    return pd.Series({
        "avg_separation":    _safe_div(ngs["wt_sep_num"], ngs["wt_tgt_sep"]),
        "yac_over_expected": _safe_div(ngs["wt_yac_num"], ngs["wt_rec_yac"]),
        "target_share":      _safe_div(ws["wt_tgt_share"], ws["total_weight"]),
        "air_yards_share":   _safe_div(ws["wt_ay_share"], ws["total_weight"]),
        "motion_catch_rate": _safe_div(motion["motion_catch_wt"], motion["motion_tgt_wt"]),
        "rz_target_share":   _safe_div(rz_share["wt_rz_share"], rz_share["total_weight"]),
    })


def _te_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build TE archetype feature vector from reference players.

    Features: target_share, air_yards_per_target, yac_over_expected,
              snap_share, rz_target_share
    """
    te_refs = ref_players[ref_players["position"] == "TE"].copy()
    if te_refs.empty:
        raise RuntimeError("No TE reference players found")

    te_weight = te_refs[["gsis_id", "season", "total_offensive_snaps", "weight"]].dropna(subset=["gsis_id"])
    if te_weight.empty:
        raise RuntimeError("TE reference players have no gsis_id mapping. Check rosters table.")

    con = duckdb.connect(DB_PATH)
    con.register("te_ref", te_weight)

    # Weekly stats: target_share and air_yards_per_target, weighted by snaps * team_weight
    ws = con.execute("""
        SELECT
            SUM(w.target_share          * r.total_offensive_snaps * r.weight) AS wt_tgt_share,
            SUM(w.receiving_air_yards   * r.weight)                            AS wt_air_yards,
            SUM(w.targets               * r.weight)                            AS wt_targets,
            SUM(r.total_offensive_snaps * r.weight)                            AS total_weight
        FROM weekly_stats w
        JOIN te_ref r ON w.player_id = r.gsis_id AND w.season = r.season
        WHERE w.season_type = 'REG' AND w.targets > 0
    """).fetchdf().iloc[0]

    # NGS receiving: YAC above expected, weighted by receptions * team_weight
    ngs = con.execute("""
        SELECT
            SUM(n.receptions * n.avg_yac_above_expectation * r.weight) AS wt_yac_num,
            SUM(n.receptions * r.weight)                                AS wt_rec
        FROM ngs_receiving n
        JOIN te_ref r ON n.player_gsis_id = r.gsis_id AND n.season = r.season
        WHERE n.season_type = 'REG'
    """).fetchdf().iloc[0]

    # RZ target share: per-player share, then weighted average (includes zeros)
    rz_tgt = con.execute("""
        WITH player_rz AS (
            SELECT p.receiver_player_id AS gsis_id, p.season, p.posteam,
                   COUNT(*) AS rz_targets
            FROM pbp p
            WHERE p.play_type = 'pass' AND p.yardline_100 <= ?
                AND ((p.posteam = 'NO' AND p.season = 2024) OR (p.posteam = 'SEA' AND p.season = 2025))
                AND p.week <= ? AND p.receiver_player_id IS NOT NULL
            GROUP BY p.receiver_player_id, p.season, p.posteam
        ),
        team_rz AS (
            SELECT posteam, season, COUNT(*) AS team_total
            FROM pbp
            WHERE play_type = 'pass' AND yardline_100 <= ?
                AND ((posteam = 'NO' AND season = 2024) OR (posteam = 'SEA' AND season = 2025))
                AND week <= ?
            GROUP BY posteam, season
        )
        SELECT
            SUM(COALESCE(pr.rz_targets, 0)::DOUBLE / tt.team_total
                * r.total_offensive_snaps * r.weight) AS wt_rz_share,
            SUM(r.total_offensive_snaps * r.weight)   AS total_weight
        FROM te_ref r
        JOIN team_rz tt ON r.season = tt.season
            AND tt.posteam = CASE WHEN r.season = 2024 THEN 'NO' ELSE 'SEA' END
        LEFT JOIN player_rz pr ON pr.gsis_id = r.gsis_id AND pr.season = r.season
    """, [RZ_YARDLINE, REG_SEASON_MAX_WEEK, RZ_YARDLINE, REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    # Snap share from ref DataFrame
    snap_share = (te_refs["avg_snap_share"] * te_refs["total_offensive_snaps"] * te_refs["weight"]).sum() / \
                 (te_refs["total_offensive_snaps"] * te_refs["weight"]).sum()

    con.close()

    def _safe_div(num, denom):
        return float(num / denom) if denom and denom > 0 else None

    return pd.Series({
        "target_share":         _safe_div(ws["wt_tgt_share"], ws["total_weight"]),
        "air_yards_per_target": _safe_div(ws["wt_air_yards"], ws["wt_targets"]),
        "yac_over_expected":    _safe_div(ngs["wt_yac_num"], ngs["wt_rec"]),
        "snap_share":           float(snap_share) if snap_share is not None else None,
        "rz_target_share":      _safe_div(rz_tgt["wt_rz_share"], rz_tgt["total_weight"]),
    })


def _ol_archetype(ref_players: pd.DataFrame) -> pd.Series:
    """Build OL archetype feature vector using team-level proxies.

    Individual OL grade data requires paid services (PFF). These features
    use team-level PBP proxies instead. position_slot is limited to C/G/T
    because LT/LG/RG/RT granularity is not available in free data sources.

    Features: rush_epa_guard, rush_epa_tackle, rush_epa_end,
              team_pressure_rate_allowed, avg_snap_share
    """
    ol_refs = ref_players[ref_players["position"].isin(OL_POSITIONS)].copy()
    if ol_refs.empty:
        raise RuntimeError("No OL reference players found")

    con = duckdb.connect(DB_PATH)

    # Team-level rush EPA by run gap, weighted 40/60
    rush_epa = con.execute("""
        SELECT
            run_gap,
            SUM(CASE WHEN posteam='NO'  AND season=2024 THEN epa*0.40 ELSE
                     CASE WHEN posteam='SEA' AND season=2025 THEN epa*0.60 ELSE 0 END END) AS wt_epa,
            SUM(CASE WHEN posteam='NO'  AND season=2024 THEN 0.40 ELSE
                     CASE WHEN posteam='SEA' AND season=2025 THEN 0.60 ELSE 0 END END) AS wt_plays
        FROM pbp
        WHERE ((posteam='NO' AND season=2024) OR (posteam='SEA' AND season=2025))
            AND play_type = 'run'
            AND run_gap IS NOT NULL
            AND week <= ?
        GROUP BY run_gap
    """, [REG_SEASON_MAX_WEEK]).fetchdf()

    rush_epa_by_gap = {row["run_gap"]: row["wt_epa"] / row["wt_plays"]
                       for _, row in rush_epa.iterrows() if row["wt_plays"] > 0}

    # Team pressure rate on dropbacks, weighted 40/60
    pressure = con.execute("""
        SELECT
            SUM(CASE WHEN posteam='NO'  AND season=2024 THEN CAST(was_pressure AS DOUBLE)*0.40 ELSE
                     CASE WHEN posteam='SEA' AND season=2025 THEN CAST(was_pressure AS DOUBLE)*0.60 ELSE 0 END END) AS wt_pressures,
            SUM(CASE WHEN posteam='NO'  AND season=2024 THEN 0.40 ELSE
                     CASE WHEN posteam='SEA' AND season=2025 THEN 0.60 ELSE 0 END END) AS wt_dropbacks
        FROM pbp
        WHERE ((posteam='NO' AND season=2024) OR (posteam='SEA' AND season=2025))
            AND qb_dropback = 1
            AND week <= ?
    """, [REG_SEASON_MAX_WEEK]).fetchdf().iloc[0]

    con.close()

    # Average snap share of OL players who passed the floor, weighted by snaps * team_weight
    snap_share = (ol_refs["avg_snap_share"] * ol_refs["total_offensive_snaps"] * ol_refs["weight"]).sum() / \
                 (ol_refs["total_offensive_snaps"] * ol_refs["weight"]).sum()

    def _safe_div(num, denom):
        return float(num / denom) if denom and denom > 0 else None

    return pd.Series({
        "rush_epa_guard":             rush_epa_by_gap.get("guard"),
        "rush_epa_tackle":            rush_epa_by_gap.get("tackle"),
        "rush_epa_end":               rush_epa_by_gap.get("end"),
        "team_pressure_rate_allowed": _safe_div(pressure["wt_pressures"], pressure["wt_dropbacks"]),
        "avg_snap_share":             float(snap_share) if snap_share is not None else None,
    })


def build_position_archetypes() -> pd.DataFrame:
    """Build all five position-group archetypes and return as a DataFrame."""
    ref_players = get_reference_players()
    rows = [
        _qb_archetype(ref_players).rename("QB"),
        _rb_archetype(ref_players).rename("RB"),
        _wr_archetype(ref_players).rename("WR"),
        _te_archetype(ref_players).rename("TE"),
        _ol_archetype(ref_players).rename("OL"),
    ]
    df = pd.concat(rows, axis=1).T
    df.index.name = "position_group"
    return df.reset_index()


def write_archetypes() -> None:
    """Run all archetype builds and persist to DuckDB."""
    scheme = build_scheme_profile()
    archetypes = build_position_archetypes()
    ref_players = get_reference_players()

    con = duckdb.connect(DB_PATH)
    for table, df in [
        ("kubiak_scheme_profile", scheme),
        ("kubiak_position_archetypes", archetypes),
        ("kubiak_reference_players", ref_players),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.register("df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.close()


if __name__ == "__main__":
    write_archetypes()
    print("Archetypes built and persisted.")
