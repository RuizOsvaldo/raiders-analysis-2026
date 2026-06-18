"""About this project: the reasoning behind every design decision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="Las Vegas Raiders 2026 | About", layout="wide")
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

## Why these two reference seasons

Kubiak's scheme is defined by his two most recent full OC seasons:

- **2024 New Orleans Saints** -- Kubiak's first full offensive coordinator role,
  Derek Carr (then Spencer Rattler) as QB. The Saints had a lower-scoring offense
  but Kubiak established his wide-zone and motion identity.
- **2025 Seattle Seahawks** -- Kubiak's most recent season, Sam Darnold as QB.
  Greater roster control and more reflection of what Kubiak would build given
  full authority.

The Seahawks season is weighted **60%** and the Saints season **40%**
because recency matters: if Kubiak ran something in 2025 that he did not run
in 2024, the 2025 version is a better predictor of what he will install in
Las Vegas. A simple average would treat a two-year-old season as equally
predictive as last year.

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
across QB, RB, WR and TE the reference players average roughly 5-12 points above
a 250-player control set, with RB the strongest separator now that it uses the
performance model. This is run by `scripts/validate.py`. It is a necessary
sanity check, not a substitute for backtesting against real future production --
which would require outcome data the free sources don't provide.

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
far more reference seasons than two.

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
- **More reference seasons**: Kubiak was a position coach and then assistant before
  his two OC years. We have two seasons. More would make the archetype more stable.

The project is designed so that when better data becomes available, the feature
lists in `src/archetype.py` can be extended without changing the scoring
architecture.
""")
