"""Validation harness for the scheme-fit model.

The model has no labeled target, so we cannot compute a conventional accuracy.
What we CAN check is internal validity: the players who define Kubiak's archetype
(his actual reference starters) should grade clearly higher than an arbitrary
control population. If they don't, the distance metric isn't capturing fit and
no downstream grade should be trusted.

This is an in-sample sanity check (the reference players contribute to the
archetype mean, though the league-wide scaler and the 40/60 blend mean none of
them match it exactly). A large reference-vs-control separation is necessary, not
sufficient, evidence the model works. It is the cheapest backtest available
without paid outcome data.

Run:  uv run python3 scripts/validate.py
"""

import math
import statistics
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from archetype import PHYSICAL_POSITION_MAP  # noqa: E402
from scoring import (  # noqa: E402
    PRIMARY_MODEL,
    score_player,
    score_player_physical,
)

DB_PATH = str(Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")


def _primary_grade(gsis_id: str, position: str):
    """Score a player with the SAME model the production primary grade uses.

    Routes RB to the performance model and everyone else to the physical model,
    matching PRIMARY_MODEL, so the harness validates the grade actually shipped.
    """
    group = PHYSICAL_POSITION_MAP.get(position)
    if group is None:
        return None, None
    model = PRIMARY_MODEL.get(group, "physical")
    scorer = score_player if model == "performance" else score_player_physical
    res = scorer(gsis_id, group)
    return res["grade"], res["position_group"]


def _grade_population(rows) -> dict[str, list[float]]:
    """Score a list of (gsis_id, position) rows; return grades grouped by position_group."""
    by_group: dict[str, list[float]] = {}
    for gsis_id, position in rows:
        if gsis_id is None:
            continue
        try:
            g, grp = _primary_grade(gsis_id, position)
        except RuntimeError:
            continue
        if g is None or math.isnan(g):
            continue
        by_group.setdefault(grp, []).append(g)
    return by_group


def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)

    # Reference players (the archetype-definers). OL have NULL gsis_id and are skipped.
    ref_rows = con.execute("""
        SELECT gsis_id, position FROM kubiak_reference_players
        WHERE gsis_id IS NOT NULL
    """).fetchall()

    # Control: other offensive skill players on a 2024/2025 roster who are NOT
    # reference players. Deterministic (ordered + capped) so the separation
    # numbers are stable run-to-run rather than depending on a random sample.
    control_rows = con.execute("""
        SELECT player_id, position FROM (
            SELECT DISTINCT r.player_id, r.position
            FROM rosters r
            WHERE r.season IN (2024, 2025)
              AND r.position IN ('QB','RB','WR','TE')
              AND r.player_id NOT IN (
                  SELECT gsis_id FROM kubiak_reference_players WHERE gsis_id IS NOT NULL
              )
        )
        ORDER BY player_id
        LIMIT 250
    """).fetchall()
    con.close()

    ref = _grade_population(ref_rows)
    ctl = _grade_population(control_rows)

    groups = sorted(set(ref) | set(ctl))
    print(f"{'group':<6}{'ref_n':>6}{'ref_mean':>10}{'ctl_n':>7}{'ctl_mean':>10}{'separation':>12}")
    print("-" * 51)
    all_ref, all_ctl = [], []
    for grp in groups:
        r = ref.get(grp, [])
        c = ctl.get(grp, [])
        all_ref += r
        all_ctl += c
        rm = statistics.mean(r) if r else float("nan")
        cm = statistics.mean(c) if c else float("nan")
        sep = rm - cm if r and c else float("nan")
        print(f"{grp:<6}{len(r):>6}{rm:>10.1f}{len(c):>7}{cm:>10.1f}{sep:>12.1f}")

    print("-" * 51)
    rm = statistics.mean(all_ref) if all_ref else float("nan")
    cm = statistics.mean(all_ctl) if all_ctl else float("nan")
    print(f"{'ALL':<6}{len(all_ref):>6}{rm:>10.1f}{len(all_ctl):>7}{cm:>10.1f}{rm - cm:>12.1f}")
    print()
    if rm - cm > 0:
        print(f"PASS: reference players grade {rm - cm:.1f} points above control on average.")
        print("      (Necessary internal-validity signal; not a substitute for outcome backtesting.)")
    else:
        print("FAIL: reference players do NOT grade above control. The distance metric")
        print("      is not separating fit from non-fit. Investigate features/archetype.")


if __name__ == "__main__":
    main()
