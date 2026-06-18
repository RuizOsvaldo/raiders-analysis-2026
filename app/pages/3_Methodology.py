"""Explain how grades are computed and what they mean."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from utils import load_scheme_profile

st.title("Methodology")

st.markdown("""
## What this project measures

This app grades the 2026 Las Vegas Raiders offensive roster against Klint
Kubiak's scheme tendencies from his three OC seasons, recency-weighted: the
2021 Minnesota Vikings (15%), the 2024 New Orleans Saints (35%), and the 2025
Seattle Seahawks (50%). Newer seasons are weighted higher because they reflect
the offense Kubiak built with the most autonomy. (The 2021 season abstains from
the FTN-charted motion/play-action features, which only exist from 2022 on.)
See the About page for the full reasoning.

## Two grades, one Scheme Experience tag

**Physical Fit (primary)**

Trait-based. Compares a player's height, weight, and combine measurables
(forty, shuttle, three-cone, vertical, broad jump) against the snap-weighted
mean of those same metrics across Kubiak's reference players at the position.
Each feature is standardized against the **league-wide** spread at that
position, so a distance of "1" means one league standard deviation -- a stable
unit, not the spread of the handful of players Kubiak happened to coach. The
grade is the root-mean-square of those standardized deviations over the
features the player actually has, converted with:

```
distance = sqrt( mean( standardized_deviation_i^2 ) )   # over available features
grade    = 100 * exp(-distance / 1.5)
```

A player one league std from the archetype grades about 51; two std grades
about 26; an exact match grades 100. Because the distance is a per-feature
*mean* rather than a sum, the scale factor (1.5) no longer has to absorb how
many features a position uses, and missing measurements no longer bias the
grade up or down -- see "Missing data and coverage" below.

This grade is scheme-agnostic. It does not care what offense the player has
run before. It only asks: do they have the physical and athletic traits
Kubiak's scheme has historically rewarded?

**Statistical Similarity (secondary)**

Performance-based. Compares a player's recent NFL statistics (last two
seasons) against Kubiak's reference players' statistics at that position.
Same distance calculation and grade formula as Physical Fit.

This grade conflates three things: physical traits, system experience, and
recent performance. Low Statistical Similarity often means a player has not
run a compatible scheme yet, not that they cannot perform in one. Read it
together with the Physical Fit grade, not in isolation.

**Scheme Experience (informational)**

A manually-curated tag indicating prior NFL experience in a Kubiak-tree offense:

- **yes**: ran a wide-zone Shanahan-tree system (SF under Shanahan, SEA under Kubiak,
  LAR under McVay, GB/LV under LaFleur/Getsy, MIN under O'Connell, ATL under Robinson,
  LV under Getsy 2024 or Kubiak 2025)
- **partial**: ran something with significant overlap (zone-blocking emphasis,
  heavy play-action) or played under a scheme with mixed influences
- **no**: different scheme entirely (power run, Air Raid, RPO-heavy spread, etc.)

## Missing data and coverage

A missing measurement makes a grade *less certain*, not *worse*. Earlier
versions multiplied the grade by coverage (`grade = raw_grade * features_present
/ features_total`), which penalized the players we knew least about. That penalty
has been removed.

Instead, the grade is a root-mean-square distance over the features that are
available, so it is on the same scale regardless of how many features a player
has on file. The uncertainty lives entirely in the confidence interval, which
widens when data is missing. Height and weight always exist in roster data, so a
player who never attended the combine is still scored on ht/wt rather than
dropping to zero.

Coverage is still shown next to each grade as a data-completeness indicator:
read a high grade with low coverage as "promising but uncertain" -- and look at
the width of the confidence band, which is now where that uncertainty is
expressed.

## Data sources

All data is from the open-source nflverse project, accessed via
nfl_data_py and nflreadpy:

- Snap counts, NextGen Stats, rosters, play-by-play, FTN charting, combine: nfl_data_py
- Weekly player stats: nflreadpy

## Known limitations

- Personnel groupings (11/12/21 personnel) require paid data (PFF) and are not used
- Individual OL blocking grades come from PFF Premium (pass-block grade, run-block grade, pass-block efficiency), manually exported and joined by name
- Roughly 9% of plays did not match between PBP and FTN charting and were excluded
- 24% of reference players have no combine record; these contribute only height and weight
- Combine coverage is weakest for offensive linemen
- Reference set is 3 seasons (2021, 2024, 2025); archetypes are noisier than with 4-5 years of data

## Kubiak's actual play-calling tendencies
""")

scheme = load_scheme_profile()

# Rename and reformat for readability
display = scheme.copy()
display["Context"]         = display["red_zone"].map({0: "Outside red zone", 1: "Red zone (inside 20)"})
display["Total plays"]     = display["total_plays"]
display["Avg EPA / play"]  = display["avg_epa"].round(3)
display["Pass rate"]       = (display["pass_rate"] * 100).round(1).astype(str) + "%"
display["Shotgun rate"]    = (display["shotgun_rate"] * 100).round(1).astype(str) + "%"
display["No-huddle rate"]  = (display["no_huddle_rate"] * 100).round(1).astype(str) + "%"
display["Play-action rate"] = (display["play_action_rate"] * 100).round(1).astype(str) + "%"
display["Motion rate"]     = (display["motion_rate"] * 100).round(1).astype(str) + "%"
display["Screen rate"]     = (display["screen_rate"] * 100).round(1).astype(str) + "%"
display["RPO rate"]        = (display["rpo_rate"] * 100).round(1).astype(str) + "%"

cols = ["Context", "Total plays", "Avg EPA / play", "Pass rate",
        "Shotgun rate", "No-huddle rate", "Play-action rate",
        "Motion rate", "Screen rate", "RPO rate"]
st.dataframe(display[cols], use_container_width=True, hide_index=True)
st.caption(
    "All rates are as a percentage of total offensive plays (pass + run). "
    "Play-action rate as a fraction of pass plays only is roughly pass-action / pass-rate "
    "(e.g., 13% / 57% = ~23% of dropbacks outside the red zone). "
    "Motion rate reflects FTN charting of any pre-snap motion; Shanahan-tree offenses "
    "routinely exceed 50%. "
    "Source: Saints 2024 and Seahawks 2025 play-by-play, recency-weighted "
    "(35% and 50%, renormalized to ~41/59 between them). The 2021 Vikings are "
    "excluded from this table because FTN charting only starts in 2022. "
    "Regular season only."
)
