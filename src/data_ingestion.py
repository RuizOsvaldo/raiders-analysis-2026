"""Pull data from nfl_data_py, cache to DuckDB, expose query functions."""

import pathlib

import duckdb
import nfl_data_py as nfl
import nflreadpy as nflr

DB_PATH = str(pathlib.Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")


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
    """Run all ingest functions with years matched to what nflverse has published."""
    ingest_snap_counts([2024, 2025])
    ingest_weekly_stats([2024, 2025])
    ingest_combine([2024, 2025])
    ingest_ngs_passing([2024, 2025])
    ingest_ngs_rushing([2024, 2025])
    ingest_ngs_receiving([2024, 2025])
    ingest_rosters([2024, 2025, 2026])
    ingest_pbp([2024, 2025])
    ingest_ftn([2024, 2025])
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
