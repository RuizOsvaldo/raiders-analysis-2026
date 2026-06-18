"""About this project: the reasoning behind every design decision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.title("About This Project")
st.caption("Why it was built this way, and what every number means.")

st.markdown("""
## Why this project exists

Klint Kubiak was hired as the Las Vegas Raiders head coach in early 2026.
His offensive system -- wide-zone running, heavy pre-snap motion, play-action
off the run game -- has a specific physical and athletic profile that it rewards.
The question this project answers is: **does the current Raiders roster have
the right players for that offense?**

The project builds a measurable answer using only free, public data.

---

## Why these three reference seasons

Kubiak's scheme is defined by his three seasons as an NFL offensive coordinator,
each weighted by recency:

- **2021 Minnesota Vikings** (weighted **15%**) -- Kubiak's first OC year, Kirk
  Cousins at QB. He was still working inside his father's (Gary Kubiak) framework
  with less autonomy, so it gets the lightest weight, but it adds a third season
  of physical and run-game signal.
- **2024 New Orleans Saints** (weighted **35%**) -- Derek Carr (then Spencer
  Rattler) at QB. A lower-scoring offense, but where Kubiak established his own
  wide-zone and motion identity.
- **2025 Seattle Seahawks** (weighted **50%**) -- Kubiak's most recent season,
  Sam Darnold at QB. The most roster control and the best reflection of what
  Kubiak would build given full authority.

More recent seasons are weighted higher because recency matters: if Kubiak ran
something in 2025 that he did not run in 2021, the 2025 version is a better
predictor of what he will install in Las Vegas. A simple average would treat a
four-year-old season as equally predictive as last year.

**One caveat on the 2021 season.** FTN play-level charting -- which feeds the
play-action, pre-snap motion, RPO, and out-of-pocket features -- only exists in
nflverse from 2022 onward. So the 2021 Vikings players contribute to the
physical and non-charted performance features but *abstain* from the
FTN-derived ones, and the scheme-tendency profile itself is built from the
2024 and 2025 seasons only. Adding another reference season is a one-line
change to the `REFERENCE` config in `src/archetype.py`.

---

## Why Physical Fit is the primary grade

A player's statistics are a product of three things at once: their physical
traits, the scheme they ran, and recent performance noise. When a player grades
low on Statistical Similarity, it is almost always because they have not run
a Kubiak-compatible offense, not because they are a poor player.

Physical traits are different. Height, weight, speed, and explosiveness do not
change based on what offense someone ran. A wide receiver either has the build
and athleticism Kubiak's wide-zone play-action system rewards, or they do not --
regardless of whether they played for Kyle Shanahan or a spread-RPO coordinator.

Physical Fit isolates what we can actually evaluate before a player takes a snap
in the new system. It is the primary grade because it answers the actual question.
Statistical Similarity is preserved as secondary context -- useful for identifying
players who have already proven they can produce in a compatible system.

**The running back exception.** Validation (see below) showed combine
measurables separate Kubiak's players from a control population at QB, WR and
TE, but *not* at running back -- every NFL back is a similar athlete, so
40-time and vertical carry no zone-scheme-fit signal. Running back fit is a
running-*style* question: cutback vision, outside-zone success, work as a
receiver. So at RB the primary grade is sourced from the **performance model**
(outside-zone success rate, rush yards over expected, shotgun efficiency, red
zone usage) instead of physical traits. Every other position stays on Physical
Fit.

---

## How we know the grades mean something

The model has no labeled outcome, so it can't report a conventional accuracy.
What it can show is internal validity: the players who *define* Kubiak's
archetype should grade higher than an arbitrary control population. They do --
run by `scripts/validate.py`, Kubiak's reference players average about **9
points** above a deterministic 250-player control set:

| Group | Separation (ref − control) |
|---|---|
| QB | +29 |
| TE | +7 |
| WR | +6 |
| RB | +3 |
| **All** | **+9** |

The separation is strongest at quarterback and thinnest at running back. RB
stays the hardest position to separate even on the performance model, because
back production is noisy and NFL backs are broadly interchangeable -- a useful
reminder that the RB grade carries the least signal of any position. (Offensive
line is excluded from this test; see the OL section for why its one-sided grade
doesn't fit a reference-vs-control comparison.) This is a necessary sanity
check, not a substitute for backtesting against real future production -- which
would require outcome data the free sources don't provide.

---

## How missing data is handled

Earlier versions multiplied the grade by a "coverage" factor
(`grade = raw_grade * features_present / features_total`), so a player scored on
2 of 5 features could keep at most 40% of their grade. That conflated two
different things: *how well a player fits* and *how much we know about them*.
It systematically buried exactly the players we had least data on -- a genuinely
great physical match missing a combine record ranked below a mediocre one with a
complete record.

The model now keeps those two ideas separate:

- **The grade (point estimate)** is the root-mean-square standardized distance to
  the archetype over the features that *are* available. Because it is a mean
  rather than a sum, it is on the same scale whether a player has 2 features or
  7 -- missing data no longer biases the grade up or down.
- **The confidence interval** carries the uncertainty. It widens when features
  are missing and when the reference set has few players at a position. A player
  graded on 2 of 5 traits gets the same kind of point estimate as anyone else,
  but a visibly wider band.

Height and weight always exist in roster data, so a player who simply never
attended the combine is still scored on ht/wt instead of collapsing to zero.
Players with no usable features at all for a given model are reported as
undefined and excluded from unit averages rather than counted as a zero.

Coverage is still shown next to every grade as an honest data-completeness
indicator -- read a high grade with low coverage as "promising but uncertain,"
which is what the wide confidence band is telling you.

---

## How the grade formula works

```
distance  = sqrt( mean( standardized_deviation_i^2 ) )   # over available features
grade     = 100 * exp(-distance / 1.5)
```

Standardization uses a StandardScaler fit on the **league-wide** combine
population at each position, so a distance of "1" means one league standard
deviation -- a stable unit, not the spread of the two or three players Kubiak
happened to coach. A player who sits one league std from the archetype grades
about 51; two league std grades about 26; an exact match grades 100.

The scale factor is 1.5. Because distance is now a per-feature mean rather than a
sum, the scale factor no longer has to absorb how many features a position uses,
so the old hand-tuned 5.0 (and the 2.0 before it) are retired.

Confidence intervals widen when features are missing and when the reference
set has few players at a position. They are shown as context but the primary
number is the grade.

---

## Why these position group weights

| Group | Weight | Reasoning |
|---|---|---|
| QB | 30% | The quarterback runs the offense; Kubiak's system is especially QB-dependent because of play-action timing and pre-snap reads from motion |
| OL | 25% | Wide-zone blocking requires specific athletic and technique profiles; the line is the engine of the run game |
| RB | 20% | Zone schemes demand backs who can identify the cutback lane and contribute as receivers; weight absorbs both rushing and pass-protection since individual pass-pro grades are not in free data |
| WR | 15% | Separation and YAC matter but Kubiak's receivers are asked to do less than in a pure passing system |
| TE | 10% | Primarily a run-blocking and seam-route role in this system; important but narrower contribution |

These are designed weights based on analytical consensus for wide-zone play-action
offenses, not learned from data. A model that learned these weights would need
far more reference seasons than the three available here.

---

## Why the Scheme Experience tag is manual

No free data source classifies coaching lineages. The tag is applied by reviewing
each player's 2022-2025 team history and identifying whether their offensive
coordinator came from the Shanahan-tree (Kyle Shanahan at SF, Sean McVay at LAR,
Matt LaFleur at GB, Kevin O'Connell at MIN, Mike McDaniel at MIA, Klint Kubiak
directly, Luke Getsy at LV 2024 who came from LaFleur's staff, Zac Robinson at
ATL who came from McVay's staff).

Three tiers are coarser than ideal but sufficient:
- **yes**: strong prior exposure to the system
- **partial**: adjacent scheme with meaningful overlap (zone-heavy, play-action)
- **no**: materially different scheme (power run, Air Raid, RPO-heavy spread)

A player tagged **no** with a high Physical Fit grade is the most interesting
case -- they have the traits but have never run this system. High upside,
unknown execution risk.

---

## Offensive line: real PFF grades

OL fit is the one place this project uses **paid** data. Individual blocking
grades aren't in any free source, so the linemen are graded on PFF Premium's
**pass-block grade, run-block grade, and pass-block efficiency**, exported by
hand and matched by name. Two things to read carefully:

- The grade is a **meet-or-exceed** measure, not a "be exactly average like
  Kubiak's line" measure. A lineman who blocks at least as well as Kubiak's
  reference linemen scores at the top; only blocking *below* that level pulls the
  grade down. (Blocking quality is universal -- being great isn't a "bad fit" --
  so the usual reference-vs-control archetype test doesn't apply to OL the way it
  does to skill positions.)
- **Rookies have no NFL blocking grade**, so a rookie lineman is graded on
  athletic measurables only and carries a wide confidence band. Read a high
  rookie OL grade as an athletic projection, not demonstrated blocking.

## What free data still cannot do

These limitations are gaps in what is publicly available:

- **Personnel groupings** (11/12/21 personnel rates): paywalled. We know
  Kubiak runs a lot of 12-personnel but cannot measure it.
- **Route-level receiver data**: separation, route usage by type. Available in
  paid NextGen Stats tiers but not the free API.
- **More reference seasons**: Kubiak has three NFL seasons as a full offensive
  coordinator (2021, 2024, 2025), and this project uses all three. More would
  make the archetype more stable, but they don't exist yet.

The project is designed so that when better data becomes available, the feature
lists in `src/archetype.py` can be extended without changing the scoring
architecture.
""")
