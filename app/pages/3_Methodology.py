"""Explain how grades are computed and what they mean."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from utils import load_scheme_profile

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology")

st.markdown("""
## What this project measures

This app grades the 2026 Las Vegas Raiders offensive roster against Klint
Kubiak's scheme tendencies from his two most recent OC seasons: the 2024
New Orleans Saints (weighted 40%) and the 2025 Seattle Seahawks (weighted 60%).
The newer season is weighted higher because it reflects the offense Kubiak
built with the most recent autonomy.

## Two grades, one Scheme Experience tag

**Physical Fit (primary)**

Trait-based. Compares a player's height, weight, and combine measurables
(forty, shuttle, three-cone, vertical, broad jump) against the snap-weighted
mean of those same metrics across Kubiak's reference players at the position.
Distance to the archetype is converted to a 0-100 grade using
`100 * exp(-distance / scale)`, then multiplied by the fraction of features
the player has data for. A player with only height and weight on file gets
penalized to roughly 40% of their raw grade.

This grade is scheme-agnostic. It does not care what offense the player has
run before. It only asks: do they have the physical and athletic traits
Kubiak's scheme has historically rewarded?

**Statistical Similarity (secondary)**

Performance-based. Compares a player's recent NFL statistics (last two
seasons) against Kubiak's reference players' statistics at that position.
Same distance calculation, same grade formula, same coverage penalty.

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

## Coverage penalty

Every grade is penalized when the underlying data is incomplete:

```
coverage = features_with_data / total_features_for_position
grade = raw_grade * coverage
```

A player missing three of five combine events gets a grade worth at most
60% of what it would be with full data. This is the most important
user-facing safety rail: never trust a high grade with low coverage.

## Data sources

All data is from the open-source nflverse project, accessed via
nfl_data_py and nflreadpy:

- Snap counts, NextGen Stats, rosters, play-by-play, FTN charting, combine: nfl_data_py
- Weekly player stats: nflreadpy

## Known limitations

- Personnel groupings (11/12/21 personnel) require paid data (PFF) and are not used
- Individual OL blocking grades require paid charting; team-level proxies are used instead
- Roughly 9% of plays did not match between PBP and FTN charting and were excluded
- 24% of reference players have no combine record; these contribute only height and weight
- Combine coverage is weakest for offensive linemen
- Reference set is only 2 seasons; archetypes are noisier than with 4-5 years of data

## Kubiak's actual play-calling tendencies
""")

scheme = load_scheme_profile()
st.dataframe(scheme, use_container_width=True, hide_index=True)
st.caption(
    "Play-call rates from the weighted Saints 2024 / Seahawks 2025 reference set, "
    "split between red zone (inside 20) and the rest of the field."
)
