# Nearest-Neighbor Drift Experiment: Findings

## Setup

The experiment asks: **what happens to LLR attack AUC when we replace each individual's real biological drift with drift from someone else?**

For each individual at timepoint t=1, we identify their k nearest neighbors in baseline feature space and assign them one of those neighbors' real drift vectors. We sweep k from 1 (closest neighbor) to 50 and compare three distance metrics: Euclidean, cosine, and correlation.

We also test several Gaussian models to understand what drives the AUC crash.

## Results

### NN sweep (Euclidean; cosine and correlation are identical)

| k | AUC (LLR) | Percentile of pairwise distances |
|---|-----------|----------------------------------|
| 0 (real drift) | 0.563 | 0% (self) |
| 1 | 0.812 | 17% |
| 2 | 0.820 | 21% |
| 5 | 0.845 | 32% |
| 10 | 0.875 | 43% |
| 20 | 0.900 | 51% |
| 50 | 0.903 | 51% |

### Gaussian model comparison

| Model | LLR | L1 |
|-------|:---:|:--:|
| Real drift | 0.563 | 0.572 |
| Cond. mean (memorising, p>>n) | 0.563 | 0.572 |
| **LOO cond. mean** | **0.827** | **0.597** |
| NN k=1 | 0.812 | 0.625 |
| Independent Gaussian (per-feature σ) | 0.998 | 0.910 |

## Key findings

### 1. Any drift swap immediately breaks the AUC crash

Even k=1 — the single closest neighbor — raises AUC from 0.563 to 0.81. The effect is individual-specific: swapping in any other person's drift, no matter how similar, destroys the baseline-drift coupling that degrades the attack.

### 2. The nearest neighbor is not actually close

The k=1 neighbor sits at only the 17th percentile of all pairwise distances. In this high-dimensional feature space, points are roughly equidistant (curse of dimensionality), so the "nearest" neighbor is not a meaningfully close proxy. By k=20 (~51st percentile, i.e., median distance), AUC plateaus — beyond that, neighbors are no more distant than a random draw.

### 3. Distance metric does not matter

Euclidean, cosine, and correlation distances produce identical results for both AUC and percentile rank at every k. The effect is not an artefact of how "nearness" is defined.

### 4. The conditional Gaussian was memorising, not generalising

The non-LOO conditional Gaussian appeared to reproduce the AUC crash (matching real drift at 0.563). However, with ~1205 features and ~18 individuals (p >> n), the least-squares regression perfectly interpolates the training data: `X @ X⁺ @ Y = Y`. It returns each individual's original drift, not a learned relationship.

The LOO test exposes this: when individual i is held out during fitting, predicted drift gives AUC of 0.827 — no better than NN k=1 (0.812). The population-level linear baseline-drift coupling does not generalise to held-out individuals.

### 5. Independent Gaussian fails completely

Zero-mean per-feature Gaussian noise (variance matched to real drift) gives LLR AUC of 0.998 — the attack works almost perfectly. Independent noise has no baseline-drift coupling and provides no protection.

## Implications

The NN experiment shows the AUC crash is individual-specific (any swap breaks it). The LOO test shows a population-level linear model cannot capture it. The independent Gaussian shows that per-feature variance alone is worthless. Together, these confirm that the destructive mechanism is in the nonlinear, individual-specific residual — the part of each person's drift that cannot be predicted from population-level statistics or from biologically similar neighbors.
