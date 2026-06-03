"""Patch the rosters table with current 2026 Raiders OTA players.

These players are not yet in nflreadpy's seasonal roster data (too recent).
Run this after any ingest_rosters([2026]) call to restore them.

Sources: ESPN Raiders roster (verified June 2026), nfl_data_py player directory.
"""

import duckdb

DB_PATH = "data/raw/nfl.duckdb"

# (player_id, player_name, position, depth_chart_position, height_in, weight_lb, pfr_id)
# pfr_id is used to join combine table. Players without combine records score on ht/wt only.
OTA_PLAYERS = [
    ("MEN516487", "Fernando Mendoza",    "QB", "QB", 77.0, 225, "MendFe00"),
    ("CLA362740", "Jacob Clark",         "QB", "QB", 77.0, 220, None),
    ("HEM079286", "Roman Hemby",         "RB", "RB", 72.0, 210, "HembRo00"),
    ("WAS797326", "Mike Washington Jr.", "RB", "RB", 74.0, 228, "WashMi00"),
    ("BEN601117", "Malik Benson",        "WR", "WR", 73.0, 195, "BensMa00"),
    ("BRA342173", "Jonathan Brady",      "WR", "WR", 70.0, 183, None),
    ("ROB111721", "Chase Roberts",       "WR", "WR", 76.0, 210, "RobeCh00"),
    ("RUC514959", "Corey Rucker",        "WR", "WR", 72.0, 213, None),
    ("WIL577959", "E.J. Williams Jr.",   "WR", "WR", 75.0, 205, None),
    ("GUR131030", "Patrick Gurd",        "TE", "TE", 76.0, 250, None),
    ("ZUH415291", "Trey Zuhn III",       "OL", "OT", 79.0, 320, None),
    ("PIC776223", "Justin Pickett",      "OL", "G",  79.0, 317, None),
    ("HEN583744", "Niklas Henning",      "OL", "OT", 78.0, 287, None),
    ("JAT415291", "Isaiah Jatta",        "OL", "OT", 78.0, 315, None),
    ("MIS655235", "Kamar Missouri",      "OL", "OT", 77.0, 310, None),
]


def apply_ota_patch() -> None:
    """Insert OTA players into rosters if not already present."""
    con = duckdb.connect(DB_PATH)
    cols = [r[1] for r in con.execute("PRAGMA table_info(rosters)").fetchall()]
    has_week = "week" in cols

    inserted = 0
    for player_id, player_name, position, depth_pos, height, weight, pfr_id in OTA_PLAYERS:
        exists = con.execute(
            "SELECT 1 FROM rosters WHERE player_id=? AND team='LV' AND season=2026",
            [player_id],
        ).fetchone()
        if exists:
            continue

        first = player_name.split()[0]
        last = " ".join(player_name.split()[1:])

        if has_week:
            con.execute("""
                INSERT INTO rosters (
                    player_id, player_name, first_name, last_name,
                    position, depth_chart_position,
                    team, season, status, height, weight, pfr_id, week
                ) VALUES (?, ?, ?, ?, ?, ?, 'LV', 2026, 'ACT', ?, ?, ?, NULL)
            """, [player_id, player_name, first, last,
                  position, depth_pos, height, weight, pfr_id])
        else:
            con.execute("""
                INSERT INTO rosters (
                    player_id, player_name, first_name, last_name,
                    position, depth_chart_position,
                    team, season, status, height, weight, pfr_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'LV', 2026, 'ACT', ?, ?, ?)
            """, [player_id, player_name, first, last,
                  position, depth_pos, height, weight, pfr_id])
        inserted += 1

    con.close()
    print(f"OTA patch: inserted {inserted} players ({len(OTA_PLAYERS) - inserted} already present).")


if __name__ == "__main__":
    apply_ota_patch()
