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

---

## Why the coverage penalty exists

During testing, the top-ranked players by raw Physical Fit grade were all players
with only 2 of 5 features on file (height and weight, no combine data). Their
distance to the archetype was computed on only those 2 features, and happened to
be small -- but a player who matches on height and weight alone tells you almost
nothing. Raw grades of 93 and 87 for undrafted players with no combine records
were clearly artifacts, not real signal.

The fix:

```
coverage = features_with_data / total_features_for_position
grade = raw_grade * coverage
```

A player graded on 2 of 5 features can score at most 40% of their raw grade.
A player with full combine data keeps their full grade. This is the most important
safety rail in the model: high coverage means the grade is trustworthy; low
coverage means it is a weak signal that should not drive decisions.

The same penalty applies to Statistical Similarity for the same reason.

---

## How the grade formula works

```
distance = Euclidean distance in standardized feature space
raw_grade = 100 * exp(-distance / 5.0)
grade = raw_grade * coverage
```

Standardization uses a StandardScaler fit on Kubiak's reference players, so
distances are in units of "natural variation within Kubiak's actual rosters."
A player who is one standard deviation away from the archetype on every feature
grades roughly 82. A player two standard deviations away on every feature grades
around 67. The exponential decay means the penalty accelerates as players get
further from the archetype.

The scale factor is 5.0. The handoffs originally used 2.0, but the 40/60
archetype blend cannot be exactly matched by any real player -- the archetype
is an average, not an individual. With scale 2.0, almost every player graded
between 5 and 50, which compressed the rankings to noise. Scale 5.0 spreads
grades meaningfully while still punishing large deviations.

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

## What free data cannot do

These limitations are not choices, they are gaps in what is publicly available:

- **Individual OL blocking grades**: require PFF's paid charting. This project
  uses team-level rushing EPA and pressure rate for games the lineman participated
  in -- a noisier proxy.
- **Personnel groupings** (11/12/21 personnel rates): also paywalled. We know
  Kubiak runs a lot of 12-personnel but cannot measure it.
- **Route-level receiver data**: separation, route usage by type. Available in
  paid NextGen Stats tiers but not the free API.
- **More reference seasons**: Kubiak was a position coach and then assistant before
  his two OC years. We have two seasons. More would make the archetype more stable.

The project is designed so that when better data becomes available, the feature
lists in `src/archetype.py` can be extended without changing the scoring
architecture.
""")
