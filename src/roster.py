"""Manage the current 2026 Raiders roster, including the Scheme Experience lookup."""

import duckdb
import pandas as pd

DB_PATH = "data/raw/nfl.duckdb"
SCHEME_CSV = "data/raw/scheme_experience.csv"


def load_scheme_experience() -> None:
    """Read the manually-curated CSV and persist as a DuckDB table."""
    df = pd.read_csv(SCHEME_CSV)

    valid = {"yes", "partial", "no"}
    if not df["scheme_experience"].isin(valid).all():
        raise RuntimeError(f"scheme_experience must be one of {valid}")

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS scheme_experience")
    con.register("df", df)
    con.execute("CREATE TABLE scheme_experience AS SELECT * FROM df")
    con.close()


if __name__ == "__main__":
    load_scheme_experience()
    print("Scheme experience loaded.")
