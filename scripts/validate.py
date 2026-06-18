"""Validation harness for the scheme-fit model.

The model has no labeled target, so we cannot compute a conventional accuracy.
What we CAN check is internal validity: the players who define Kubiak's archetype
(his actual reference starters) should grade clearly higher than an arbitrary
control population. If they don't, the distance metric isn't capturing fit and
no downstream grade should be trusted.

Two layers of evidence, weakest to strongest:

1. HEADLINE (in-sample). Reference players scored with the production grade
   against the full archetype. They contribute to that archetype, so this is a
   sanity floor, not proof -- a model can look good in-sample by memorizing.

2. RIGOROUS (out-of-sample + significance):
   - Leave-one-out: rebuild each position archetype with one reference player
     removed, then score that player against the archetype built from everyone
     else. This breaks the circularity. The drop from in-sample to LOO is the
     amount of separation that was just memorization.
   - Permutation test: shuffle the reference/control labels thousands of times
     to get a null distribution for the separation, yielding a p-value -- "could
     a gap this big happen by chance?"
   - Effect size: Cohen's d, so the gap is reported in standard-deviation units,
     not just raw grade points.

None of this is a substitute for backtesting against real future production,
which would require outcome data the free sources don't provide. It is the best
internal evidence available.

Run:  uv run python3 scripts/validate.py
"""

import math
import statistics
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from archetype import (  # noqa: E402
    PFF_SKILL,
    PHYSICAL_POSITION_MAP,
    get_reference_players,
    norm_name,
    norm_sql,
)
from scoring import (  # noqa: E402
    ONE_SIDED_HIGHER_BETTER,
    POSITION_FEATURES,
    PRIMARY_MODEL,
    _compute_grade,
    _pff_performance,
    build_position_scaler,
    score_player,
    score_player_physical,
)

DB_PATH = str(Path(__file__).parent.parent / "data" / "raw" / "nfl.duckdb")

# Position groups that have a meaningful reference-vs-control comparison. OL is
# excluded: its grade is one-sided (beating Kubiak's linemen is a perfect fit,
# not a deviation), so it has no natural "control" the way skill positions do.
COMPARABLE_GROUPS = ["QB", "RB", "WR", "TE"]
CONTROL_LIMIT = 250
N_PERMUTATIONS = 10000


# ---------------------------------------------------------------------------
# Layer 1: headline in-sample separation (production grade path)
# ---------------------------------------------------------------------------

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


def headline_separation() -> None:
    """Print the in-sample reference-vs-control table (the documented ~9-pt gap)."""
    con = duckdb.connect(DB_PATH, read_only=True)
    ref_rows = con.execute("""
        SELECT gsis_id, position FROM kubiak_reference_players
        WHERE gsis_id IS NOT NULL
    """).fetchall()
    control_rows = con.execute(f"""
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
        LIMIT {CONTROL_LIMIT}
    """).fetchall()
    con.close()

    ref = _grade_population(ref_rows)
    ctl = _grade_population(control_rows)

    groups = sorted(set(ref) | set(ctl))
    print("LAYER 1 -- headline in-sample separation (production grade path)")
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
    print("  (In-sample: reference players help build the archetype they score against.)")


# ---------------------------------------------------------------------------
# Layer 2 helpers: exact leave-one-out on the PFF performance archetype
# ---------------------------------------------------------------------------

def _reference_pff_matrix(group: str, ref_players) -> "object":
    """One row per reference player with their PFF feature values + agg_weight.

    Replicates archetype._pff_skill_archetype's join exactly (normalized name +
    season + team, NFL level) so the weighted mean of this matrix reproduces the
    stored archetype. `agg_weight` = snaps * recency weight, the per-player weight
    that defines each player's pull on the centroid.
    """
    cfg = PFF_SKILL[group]
    positions = [p for p, g in PHYSICAL_POSITION_MAP.items() if g == group]
    refs = ref_players[ref_players["position"].isin(positions)].copy()
    if refs.empty:
        import pandas as pd
        return pd.DataFrame()
    refs["name_key"] = refs["player"].map(norm_name)

    with duckdb.connect(DB_PATH, read_only=True) as con:
        con.register("skill_ref", refs[["name_key", "team", "season", "weight"]])
        sel = ", ".join(f"p.{f}" for f in cfg["features"])
        pff = con.execute(f"""
            SELECT r.name_key, {sel}, p.{cfg['snap_col']} AS snaps, r.weight
            FROM {cfg['report']} p
            JOIN skill_ref r ON {norm_sql('p.player')} = r.name_key
                            AND p.season = r.season AND p.team_name = r.team
            WHERE p.level = 'nfl'
        """).fetchdf()
    if pff.empty:
        return pff
    pff["agg_weight"] = pff["snaps"].fillna(0).astype(float) * pff["weight"]
    return pff


def _weighted_centroid(pff, features, exclude_idx=None):
    """Per-feature snap*recency-weighted mean over the matrix, optionally dropping one row."""
    import pandas as pd
    df = pff.drop(index=exclude_idx) if exclude_idx is not None else pff
    out = {}
    for f in features:
        valid = df[df[f].notna() & (df["agg_weight"] > 0)]
        w = valid["agg_weight"].sum()
        out[f] = float((valid[f] * valid["agg_weight"]).sum() / w) if w > 0 else float("nan")
    return pd.Series(out)


def _grade_against(player_vec, centroid, scaler, features, one_sided) -> float:
    g, _, _, _ = _compute_grade(player_vec, centroid, scaler, features,
                                one_sided_higher_better=one_sided)
    return g


def leave_one_out(group: str, ref_players):
    """Return in-sample and leave-one-out grade lists for a group's reference players.

    Also returns the full centroid + scaler (reused for the control comparison)
    and `max_dev`: how far the locally-recomputed full centroid sits from the
    stored archetype. A large max_dev means this replication has drifted from the
    production build and the LOO numbers should not be trusted.
    """
    features = list(POSITION_FEATURES[group])
    one_sided = ONE_SIDED_HIGHER_BETTER.get(group)
    pff = _reference_pff_matrix(group, ref_players)
    if len(pff) < 2:
        return None

    perf_arch, perf_scaler, _, _ = build_position_scaler(group)
    full = _weighted_centroid(pff, features)
    max_dev = max(
        (abs(full[f] - perf_arch[f]) for f in features
         if not math.isnan(full[f]) and not math.isnan(float(perf_arch[f]))),
        default=0.0,
    )

    insample, loo = [], []
    for i in pff.index:
        pvec = pff.loc[i, features]
        insample.append(_grade_against(pvec, full, perf_scaler, features, one_sided))
        loo_centroid = _weighted_centroid(pff, features, exclude_idx=i)
        loo.append(_grade_against(pvec, loo_centroid, perf_scaler, features, one_sided))

    return {
        "insample": insample,
        "loo": loo,
        "full_centroid": full,
        "scaler": perf_scaler,
        "max_dev": max_dev,
    }


def _control_pff_grades(group, full_centroid, scaler, names) -> list[float]:
    """Grade control players for a group via the same manual PFF path (for fairness)."""
    features = list(POSITION_FEATURES[group])
    one_sided = ONE_SIDED_HIGHER_BETTER.get(group)
    grades = []
    with duckdb.connect(DB_PATH, read_only=True) as con:
        for name in names:
            try:
                vec = _pff_performance(name, group, con)
            except RuntimeError:
                continue
            g = _grade_against(vec, full_centroid, scaler, features, one_sided)
            if not math.isnan(g):
                grades.append(g)
    return grades


# ---------------------------------------------------------------------------
# Layer 2 statistics
# ---------------------------------------------------------------------------

def permutation_test(ref, ctl, n=N_PERMUTATIONS, seed=0) -> tuple[float, float]:
    """One-sided label-permutation test. Returns (observed_separation, p_value)."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    observed = ref.mean() - ctl.mean()
    pool = np.concatenate([ref, ctl])
    n_ref = len(ref)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        if pool[:n_ref].mean() - pool[n_ref:].mean() >= observed:
            count += 1
    # +1 smoothing so a perfect separation never reports p=0 exactly.
    return observed, (count + 1) / (n + 1)


def cohens_d(ref, ctl) -> float:
    ref = np.asarray(ref, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    nx, ny = len(ref), len(ctl)
    pooled_var = ((nx - 1) * ref.var(ddof=1) + (ny - 1) * ctl.var(ddof=1)) / (nx + ny - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")
    return (ref.mean() - ctl.mean()) / pooled_sd if pooled_sd else float("nan")


def rigorous_validation() -> None:
    """Print the leave-one-out shrinkage table plus a pooled significance test."""
    ref_players = get_reference_players()

    with duckdb.connect(DB_PATH, read_only=True) as con:
        control = con.execute(f"""
            SELECT player_id,
                   ANY_VALUE(player_name) AS player_name,
                   ANY_VALUE(position)    AS position
            FROM rosters r
            WHERE r.season IN (2024, 2025)
              AND r.position IN ('QB','RB','WR','TE')
              AND r.player_id NOT IN (
                  SELECT gsis_id FROM kubiak_reference_players WHERE gsis_id IS NOT NULL
              )
            GROUP BY player_id
            ORDER BY player_id
            LIMIT {CONTROL_LIMIT}
        """).fetchdf()
    control["group"] = control["position"].map(PHYSICAL_POSITION_MAP)

    print("\nLAYER 2 -- leave-one-out (out-of-sample) + significance")
    print(f"{'group':<6}{'n':>4}{'in-sample':>11}{'LOO':>8}{'shrink':>9}{'max_dev':>10}")
    print("-" * 48)

    pooled_loo, pooled_ctl = [], []
    for grp in ("QB", "RB", "WR", "TE", "OL"):
        res = leave_one_out(grp, ref_players)
        if res is None:
            continue
        ins = statistics.mean(res["insample"])
        loo = statistics.mean(res["loo"])
        flag = "  <-- DRIFT" if res["max_dev"] > 0.5 else ""
        print(f"{grp:<6}{len(res['loo']):>4}{ins:>11.1f}{loo:>8.1f}{ins - loo:>9.1f}{res['max_dev']:>10.4f}{flag}")

        if grp in COMPARABLE_GROUPS:
            names = control.loc[control["group"] == grp, "player_name"].tolist()
            ctl = _control_pff_grades(grp, res["full_centroid"], res["scaler"], names)
            pooled_loo += res["loo"]
            pooled_ctl += ctl

    print("-" * 48)
    print("  shrink = how many points of separation were in-sample memorization.")
    print("  max_dev = recomputed-vs-stored archetype gap (should be ~0).")

    if pooled_loo and pooled_ctl:
        observed, p = permutation_test(pooled_loo, pooled_ctl)
        d = cohens_d(pooled_loo, pooled_ctl)
        loo_mean = statistics.mean(pooled_loo)
        ctl_mean = statistics.mean(pooled_ctl)
        print(f"\nPooled skill positions (LOO reference vs control), "
              f"n_ref={len(pooled_loo)}, n_ctl={len(pooled_ctl)}:")
        print(f"  LOO reference mean : {loo_mean:.1f}")
        print(f"  control mean       : {ctl_mean:.1f}")
        print(f"  separation         : {observed:.1f} points")
        print(f"  Cohen's d          : {d:.2f}  "
              f"({'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'} effect)")
        print(f"  permutation p      : {p:.4f}  ({N_PERMUTATIONS} shuffles)")
        print()
        if observed > 0 and p < 0.05:
            print(f"PASS: held-out reference players still grade {observed:.1f} pts above control")
            print(f"      (p={p:.4f}), so the separation is not just in-sample memorization.")
        elif observed > 0:
            print(f"WEAK: held-out reference players grade {observed:.1f} pts above control, but")
            print(f"      the permutation test (p={p:.4f}) can't rule out chance at alpha=0.05.")
        else:
            print("FAIL: out-of-sample, reference players do NOT grade above control. The")
            print("      in-sample separation was memorization. Investigate features/archetype.")


def main() -> None:
    headline_separation()
    rigorous_validation()


if __name__ == "__main__":
    main()
