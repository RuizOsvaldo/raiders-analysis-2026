"""Pull data from nfl_data_py, cache to DuckDB, expose query functions."""

import pathlib

import duckdb
import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd

DB_PATH = str(pathlib.Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")
RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"


def ingest_pff_blocking(seasons: list[int]) -> None:
    """Load manually-exported PFF offensive-line blocking CSVs into DuckDB.

    PFF is a paid source with no API at the consumer tier, so these files are
    exported by hand from the PFF Premium 'Blocking Grades' report (one CSV per
    season, league-wide) and dropped in data/raw/ as pff_blocking_<season>.csv.
    They provide the per-player blocking signal that free data cannot: pass- and
    run-block grades, pass-block efficiency (pbe), and pressures allowed. The
    files are gitignored (proprietary). Join key downstream is player name +
    team + season, since rosters.pff_id is unpopulated.
    """
    keep = [
        "player", "position", "team_name", "player_game_count",
        "grades_offense", "grades_pass_block", "grades_run_block", "pbe",
        "pressures_allowed", "sacks_allowed", "hits_allowed", "hurries_allowed",
        "penalties", "snap_counts_offense", "snap_counts_pass_block",
        "snap_counts_run_block",
    ]
    frames = []
    for yr in seasons:
        path = RAW_DIR / f"pff_blocking_{yr}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing PFF export {path}. Export it from PFF Premium.")
        df = pd.read_csv(path)
        df = df[[c for c in keep if c in df.columns]].copy()
        df["season"] = yr
        df["level"] = "nfl"
        frames.append(df)

    # Optional PFF College blocking (same columns, same 0-100 PFF scale). Used to
    # grade rookie linemen on their final college season, since they have no NFL
    # blocking yet. PFF grades are curved within league, so college and NFL grades
    # are roughly comparable; a small mean-offset correction is applied at scoring.
    college_path = RAW_DIR / "pff_blocking_ncaa_2025.csv"
    if college_path.exists():
        c = pd.read_csv(college_path)
        c = c[[col for col in keep if col in c.columns]].copy()
        c["season"] = 2025
        c["level"] = "college"
        frames.append(c)
        print(f"  + PFF College 2025: {len(c)} rows")

    out = pd.concat(frames, ignore_index=True)
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS pff_blocking")
    con.register("df", out)
    con.execute("CREATE TABLE pff_blocking AS SELECT * FROM df")
    con.close()
    print(f"  pff_blocking: {len(out)} rows (NFL seasons {seasons} + college)")


def ingest_pff_skill(report: str, seasons: list[int]) -> None:
    """Load a PFF skill report (passing / receiving / rushing) into DuckDB.

    NFL files: data/raw/<report>_summary_<season>.csv (level='nfl').
    College : data/raw/ncaa_<report>_summary_2025.csv (level='college').
    All columns are kept; scoring selects the per-position feature subset.
    Joined to players by normalized name + team + season (no nflverse id in PFF).
    """
    frames = []
    for yr in seasons:
        path = RAW_DIR / f"{report}_summary_{yr}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing PFF export {path}.")
        df = pd.read_csv(path)
        df["season"] = yr
        df["level"] = "nfl"
        frames.append(df)

    college_path = RAW_DIR / f"ncaa_{report}_summary_2025.csv"
    if college_path.exists():
        c = pd.read_csv(college_path)
        c["season"] = 2025
        c["level"] = "college"
        frames.append(c)
        print(f"  + PFF College {report} 2025: {len(c)} rows")

    out = pd.concat(frames, ignore_index=True)
    # Derived zone-run rate: the most scheme-specific RB feature for a zone
    # (wide-zone) offense like Kubiak's. zone_attempts / (zone + gap).
    if {"zone_attempts", "gap_attempts"}.issubset(out.columns):
        denom = out["zone_attempts"].fillna(0) + out["gap_attempts"].fillna(0)
        out["zone_rate"] = (out["zone_attempts"] / denom).where(denom > 0)
    table = f"pff_{report}"
    con = duckdb.connect(DB_PATH)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.register("df", out)
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.close()
    print(f"  {table}: {len(out)} rows (NFL {seasons} + college)")


def ingest_snap_counts(years: list[int]) -> None:
    """Pull NFL snap counts for given years and write to DuckDB."""
    print(f"Pulling snap_counts for {years}...")
    df = nfl.import_snap_counts(years)
    if df.empty:
        raise RuntimeError(f"snap_counts pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS snap_counts")
    con.register("df", df)
    con.execute("CREATE TABLE snap_counts AS SELECT * FROM df")
    con.close()


def ingest_weekly_stats(years: list[int]) -> None:
    """Pull weekly player stats from nflreadpy and write to DuckDB."""
    df = nflr.load_player_stats(years, summary_level="week").to_pandas()
    if df.empty:
        raise RuntimeError(f"weekly_stats pull returned empty for years {years}")
    present = sorted(df["season"].unique().tolist())
    for y in years:
        if y not in present:
            raise RuntimeError(f"weekly_stats missing season {y}; got {present}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS weekly_stats")
    con.register("df", df)
    con.execute("CREATE TABLE weekly_stats AS SELECT * FROM df")
    con.close()


def ingest_combine(years: list[int]) -> None:
    """Pull NFL combine data for given years and write to DuckDB."""
    print(f"Pulling combine for {years}...")
    df = nfl.import_combine_data(years)
    if df.empty:
        raise RuntimeError(f"combine pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS combine")
    con.register("df", df)
    con.execute("CREATE TABLE combine AS SELECT * FROM df")
    con.close()


def ingest_ngs_passing(years: list[int]) -> None:
    """Pull NFL Next Gen Stats passing data for given years and write to DuckDB."""
    print(f"Pulling ngs_passing for {years}...")
    df = nfl.import_ngs_data("passing", years)
    if df.empty:
        raise RuntimeError(f"ngs_passing pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS ngs_passing")
    con.register("df", df)
    con.execute("CREATE TABLE ngs_passing AS SELECT * FROM df")
    con.close()


def ingest_ngs_rushing(years: list[int]) -> None:
    """Pull NFL Next Gen Stats rushing data for given years and write to DuckDB."""
    print(f"Pulling ngs_rushing for {years}...")
    df = nfl.import_ngs_data("rushing", years)
    if df.empty:
        raise RuntimeError(f"ngs_rushing pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS ngs_rushing")
    con.register("df", df)
    con.execute("CREATE TABLE ngs_rushing AS SELECT * FROM df")
    con.close()


def ingest_ngs_receiving(years: list[int]) -> None:
    """Pull NFL Next Gen Stats receiving data for given years and write to DuckDB."""
    print(f"Pulling ngs_receiving for {years}...")
    df = nfl.import_ngs_data("receiving", years)
    if df.empty:
        raise RuntimeError(f"ngs_receiving pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS ngs_receiving")
    con.register("df", df)
    con.execute("CREATE TABLE ngs_receiving AS SELECT * FROM df")
    con.close()


def ingest_rosters(years: list[int]) -> None:
    """Pull NFL seasonal roster data for given years and write to DuckDB."""
    print(f"Pulling rosters for {years}...")
    df = nfl.import_seasonal_rosters(years)
    if df.empty:
        raise RuntimeError(f"rosters pull returned empty for years {years}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS rosters")
    con.register("df", df)
    con.execute("CREATE TABLE rosters AS SELECT * FROM df")
    con.close()


PBP_COLUMNS = [
    "play_id", "game_id", "season", "week", "posteam", "defteam",
    "yardline_100", "down", "ydstogo", "qtr", "score_differential",
    "play_type", "shotgun", "no_huddle", "qb_dropback", "pass", "rush",
    "run_location", "run_gap", "epa", "success",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "complete_pass", "yards_gained",
]


def ingest_pbp(years: list[int]) -> None:
    """Pull play-by-play data with filtered columns and write to DuckDB."""
    print(f"Pulling pbp for {years}...")
    df = nfl.import_pbp_data(years, columns=PBP_COLUMNS)
    if df.empty:
        raise RuntimeError(f"pbp pull returned empty for years {years}")
    present = sorted(df["season"].unique().tolist())
    for y in years:
        if y not in present:
            raise RuntimeError(f"pbp missing season {y}; got {present}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS pbp")
    con.register("df", df)
    con.execute("CREATE TABLE pbp AS SELECT * FROM df")
    con.close()


def ingest_ftn(years: list[int]) -> None:
    """Pull FTN charting data and write to DuckDB."""
    print(f"Pulling ftn for {years}...")
    df = nfl.import_ftn_data(years)
    if df.empty:
        raise RuntimeError(f"ftn pull returned empty for years {years}")
    present = sorted(df["season"].unique().tolist())
    for y in years:
        if y not in present:
            raise RuntimeError(f"ftn missing season {y}; got {present}")
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS ftn")
    con.register("df", df)
    con.execute("CREATE TABLE ftn AS SELECT * FROM df")
    con.close()


def main() -> None:
    """Run all ingest functions with years matched to what nflverse has published.

    Reference seasons: 2021 MIN (Klint Kubiak's first OC year), 2024 NO, 2025 SEA.
    FTN charting only exists from 2022 on, so it is pulled for 2024/2025 only;
    2021 reference players contribute physical and non-FTN performance features.
    Combine is pulled broadly because veterans/reference players span many draft
    classes (a 2024-only combine pull would leave most players with no testing).
    """
    ref_years = [2021, 2024, 2025]
    ingest_snap_counts(ref_years)
    ingest_weekly_stats(ref_years)
    ingest_combine(list(range(2010, 2027)))
    ingest_ngs_passing(ref_years)
    ingest_ngs_rushing(ref_years)
    ingest_ngs_receiving(ref_years)
    ingest_rosters([2021, 2024, 2025, 2026])
    ingest_pbp(ref_years)
    ingest_ftn([2024, 2025])
    ingest_pff_blocking(ref_years)
    for report in ("passing", "receiving", "rushing"):
        ingest_pff_skill(report, ref_years)
    # Restore 2026 OTA additions wiped by the roster re-ingest above.
    from ota_roster_patch import apply_ota_patch
    apply_ota_patch()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
