"""Export app-facing tables from DuckDB to CSV for Streamlit Cloud deployment."""

from pathlib import Path
import duckdb

DB_PATH = Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=True)

# Pre-computed join used by load_roster_grades()
roster_grades = con.execute("""
    SELECT
        r.player_id,
        r.player_name,
        r.position,
        r.jersey_number,
        r.years_exp,
        r.college,
        r.headshot_url,
        stat.position_group,
        -- Headline grade = PFF scheme fit (performance model). Column names kept
        -- as physical_* for app compatibility; they now carry the scheme-fit grade.
        stat.grade        AS physical_grade,
        stat.raw_grade    AS physical_raw,
        stat.coverage     AS physical_coverage,
        stat.ci_low       AS physical_ci_low,
        stat.ci_high      AS physical_ci_high,
        stat.features_used        AS physical_features_used,
        stat.features_missing     AS physical_features_missing,
        stat.missing_feature_names AS physical_missing_names,
        -- Secondary grade = athletic / combine profile.
        phys.grade        AS statistical_grade,
        phys.coverage     AS statistical_coverage,
        phys.ci_low       AS statistical_ci_low,
        phys.ci_high      AS statistical_ci_high,
        phys.features_used        AS statistical_features_used,
        phys.features_missing     AS statistical_features_missing,
        stat.experience_bucket,
        COALESCE(sx.scheme_experience, 'unknown') AS scheme_experience
    FROM rosters r
    LEFT JOIN raiders_physical_player_grades phys ON phys.player_id = r.player_id
    LEFT JOIN raiders_player_grades          stat ON stat.player_id  = r.player_id
    LEFT JOIN scheme_experience              sx   ON sx.player_id    = r.player_id
    WHERE r.season = 2026 AND r.team = 'LV'
      AND r.position IN ('QB','RB','FB','WR','TE','T','G','C','OT','OG','OL')
    ORDER BY stat.position_group, stat.grade DESC NULLS LAST
""").fetchdf()

# Per-player combine measurables (for load_player_physical_features)
player_combine = con.execute("""
    SELECT
        r.player_id,
        AVG(r.height)     AS ht,
        AVG(r.weight)     AS wt,
        MAX(c.forty)      AS forty,
        MAX(c.shuttle)    AS shuttle,
        MAX(c.cone)       AS cone,
        MAX(c.vertical)   AS vertical,
        MAX(c.broad_jump) AS broad_jump
    FROM rosters r
    LEFT JOIN combine c ON c.pfr_id = COALESCE(
        r.pfr_id,
        (SELECT pfr_player_id FROM snap_counts WHERE player = r.player_name LIMIT 1)
    )
    WHERE r.season = 2026 AND r.team = 'LV'
    GROUP BY r.player_id
""").fetchdf()

tables = {
    "roster_grades":              roster_grades,
    "player_combine":             player_combine,
    "physical_offense_summary":   con.execute("SELECT * FROM raiders_physical_offense_summary").fetchdf(),
    "offense_summary":            con.execute("SELECT * FROM raiders_offense_summary").fetchdf(),
    "scheme_profile":             con.execute("SELECT * FROM kubiak_scheme_profile").fetchdf(),
    "physical_archetypes":        con.execute("SELECT * FROM kubiak_physical_archetypes").fetchdf(),
    "position_archetypes":        con.execute("SELECT * FROM kubiak_position_archetypes").fetchdf(),
}

con.close()

for name, df in tables.items():
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")

print("Done.")
