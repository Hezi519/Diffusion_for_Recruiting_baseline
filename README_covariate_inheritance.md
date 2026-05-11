# Empirical Covariate Inheritance Probabilities

This document describes the change made to the synthetic covariate transition models (`SyntheticCovariateModel` and `TunnelVisionCovariateModel`) to make parent-to-child covariate simulation more realistic, grounded in the actual ICPSR 22140 dataset.

---

## Background

In the synthetic environment, when a recruited individual (child) enters the frontier, their covariate vector must be generated from their recruiter's (parent's) covariate vector. The previous approach used a single global `inherit_prob = 0.7` for every covariate group: each of the 17 categorical fields had a 70% chance of being inherited from the parent and a 30% chance of being sampled uniformly at random. This was a reasonable placeholder but had no empirical basis.

Lingkai's recommendation was to make this more realistic by grounding the per-group inheritance rates in actual HIV/STI referral network data.

---

## Methodology

### Data source

We use **ICPSR 22140 DS0002**, which contains dyad-level data: each row represents a pair of individuals in the referral network, with both members' full covariate vectors recorded (suffixed `_1` for the recruiter and `_2` for the recruit).

We filter to **recruitment dyads** only:
- `NTYPE1 = 1`: person 1 is the recruiter in this dyad
- `NTYPE2 = 3`: person 2 is the recruited individual

This yields **73,669 valid recruiter–recruit pairs** across all disease subnetworks in the dataset (HIV, Gonorrhea, Chlamydia, Syphilis, Hepatitis). All networks are included because the covariate inheritance pattern reflects general social network dynamics, not disease-specific biology.

### Computing inheritance probability from match rate

For each covariate group with `K` categories, we compute the empirical **match rate** — the fraction of recruiter–recruit pairs where both members have the same active category:

```
match_rate = P(child category == parent category)
```

The synthetic model's generative process is:

```
P(child = c | parent = p) = inherit_prob * 1[c == p]  +  (1 - inherit_prob) * (1/K)
```

So the expected match rate under this model is:

```
match_rate = inherit_prob + (1 - inherit_prob) / K
```

Solving for `inherit_prob`:

```
inherit_prob = (match_rate - 1/K) / (1 - 1/K)
```

This formula converts the raw empirical match rate into the probability parameter used by the model. Values are clipped to `[0, 1]`.

### Handling missing data

ORIENT2 is almost entirely missing (`-8` / not applicable) across all 73,669 pairs, leaving 0 valid pairs for that group. We fall back to the **mean of all other groups' inherit_probs** (0.744) and note this explicitly in the code.

---

## Results

| Group | K | N valid pairs | Match rate | inherit_prob |
|-------|---|---------------|------------|--------------|
| LOCAL | 4 | 10,468 | 0.825 | 0.766 |
| RACE | 7 | 29,061 | 0.549 | 0.474 |
| ETHN | 4 | 28,852 | 0.896 | 0.861 |
| SEX | 3 | 69,304 | 0.482 | 0.223 |
| ORIENT | 6 | 0 | — | 0.744 *(fallback)* |
| BEHAV | 3 | 26,085 | 0.841 | 0.762 |
| PRO | 4 | 11,069 | 0.680 | 0.573 |
| PIMP | 4 | 11,078 | 0.918 | 0.891 |
| JOHN | 4 | 12,002 | 0.760 | 0.680 |
| DEALER | 4 | 11,071 | 0.831 | 0.775 |
| DRUGMAN | 4 | 11,075 | 0.984 | 0.979 |
| THIEF | 4 | 11,075 | 0.955 | 0.940 |
| RETIRED | 4 | 11,075 | 0.970 | 0.960 |
| HWIFE | 4 | 11,084 | 0.895 | 0.861 |
| DISABLE | 5 | 11,075 | 0.892 | 0.865 |
| UNEMP | 4 | 11,078 | 0.505 | 0.339 |
| STREETS | 4 | 11,063 | 0.964 | 0.952 |

---

## Interpretation of notable values

**SEX (0.223) — lowest inheritance.** Cross-sex recruitment is common in STI/HIV networks, where sexual contact bridges gender groups. The low value reflects that recruiters frequently refer people of a different sex than themselves.

**DRUGMAN (0.979) — highest inheritance.** Drug use networks are extremely tightly clustered. People who use drugs recruit almost exclusively within the same drug-use community, resulting in near-perfect inheritance.

**RACE (0.474) — moderate.** While some racial homophily exists, this network spans multiple racial groups and recruitment crosses racial lines more than might be expected.

**UNEMP (0.339) — surprisingly low.** Employment status is weakly correlated between recruiter and recruit, suggesting it is not a strong axis of social clustering in this population.

**STREETS (0.952) and THIEF (0.940) — very high.** Street-based and criminally-involved social networks are tight-knit, with strong within-group recruitment.

---

## Code changes

### `src/data/covariate_spec.py`

Added `EMPIRICAL_INHERIT_PROBS` as a module-level constant — the single source of truth for all downstream models:

```python
EMPIRICAL_INHERIT_PROBS: dict[str, float] = {
    "LOCAL":   0.766,
    "RACE":    0.474,
    ...
}
```

### `src/models/covariate_model/synthetic_covariate_model.py`

`inherit_prob` parameter changed from `float = 0.7` to `float | None = None`.

- `None` (default): uses `EMPIRICAL_INHERIT_PROBS` per group
- `float`: overrides all groups uniformly (backward-compatible for ablations)

```python
# Default — empirical per-group probs
model = SyntheticCovariateModel()

# Ablation — uniform override
model = SyntheticCovariateModel(inherit_prob=0.7)
```

### `src/models/covariate_model/tunnel_vision_covariate_model.py`

Same `inherit_prob` change applied to the non-LOCAL groups. The LOCAL group is unaffected — it continues to use the designed type-transition rules (`A→C`, `B→B`, `C→C`) controlled by `p_cross`.

```python
# Default — empirical probs for RACE, ETHN, SEX, ... (LOCAL uses p_cross as before)
model = TunnelVisionCovariateModel(p_cross=0.85)

# Ablation — uniform override for non-LOCAL groups
model = TunnelVisionCovariateModel(p_cross=0.85, inherit_prob=0.7)
```

---

## Reproducibility

To recompute these probabilities from scratch:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("ICPSR_22140/DS0002/22140-0002-Data.tsv", sep="\t", low_memory=False)
mask = (df["NTYPE1"].astype(str).str.strip() == "1") & (df["NTYPE2"] == 3)
pairs = df[mask]

fields_and_K = [
    ("LOCAL", 4), ("RACE", 7), ("ETHN", 4), ("SEX", 3), ("ORIENT", 6),
    ("BEHAV", 3), ("PRO", 4), ("PIMP", 4), ("JOHN", 4), ("DEALER", 4),
    ("DRUGMAN", 4), ("THIEF", 4), ("RETIRED", 4), ("HWIFE", 4),
    ("DISABLE", 5), ("UNEMP", 4), ("STREETS", 4),
]

for field, K in fields_and_K:
    c1 = pd.to_numeric(pairs[field + "1"], errors="coerce")
    c2 = pd.to_numeric(pairs[field + "2"], errors="coerce")
    valid = (~c1.isna()) & (~c2.isna()) & (c1 >= 0) & (c2 >= 0)
    match_rate = (c1[valid] == c2[valid]).mean()
    inherit_prob = max(0.0, min(1.0, (match_rate - 1/K) / (1 - 1/K)))
    print(f"{field}: match_rate={match_rate:.3f}, inherit_prob={inherit_prob:.3f}")
```
