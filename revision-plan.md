# Revision plan: Replacing "inter-feature dependencies" with the linear/residual decomposition

## 1. The new paragraph

### Motivation

The current "Inter-feature dependencies as the dominant factor" paragraph (lines 719–728) claims that correlations across features explain the gap between synthetic noise and real temporal drift. The correlation-preserving shuffle is offered as evidence. But the new experiments show this accounts for only a fraction of the gap (AUC rises from ~0.56 to ~0.875, not back to ~0.985). The true dominant factor — individual-specific nonlinear residuals — is absent from the paper.

The paragraph also invites the reviewer question the paper has already received: "have you tried X?" The ddof and Cauchy sentences read as a defensive list of failed alternatives. The linear/residual factorial replaces this defensive posture with a single, clean experiment that *explains why* simple models cannot work, rather than enumerating models that didn't.

### Content

The paragraph answers one question: **why does real biological drift degrade attacks so much more than any synthetic noise model?**

The argument has three steps:

1. **Setup.** For each individual, decompose their drift vector into a linear component (per-individual OLS of drift on baseline: slope β_k, intercept α_k) and a nonlinear residual ε_k. This separates what a regression-to-mean model can capture from what it cannot.

2. **Result.** A 2×2 swap experiment using the nearest neighbor's components reveals a completely one-sided effect. Keeping self residuals but swapping the linear component leaves the AUC crash intact (~0.545). Swapping the residuals but keeping self linear parameters eliminates the crash (~0.80). The linear component — everything a simple drift model can represent — is empirically inert.

3. **Implication.** The destructive mechanism lives entirely in the residual: the individual-specific, nonlinear pattern of which features overshoot or undershoot their predicted drift. No additive noise model (Gaussian, RTM, per-feature-scaled, heavy-tailed) can capture this, because these models operate on exactly the linear or marginal structure that the experiment shows is irrelevant.

### How it fits

The paragraph replaces the current "Inter-feature dependencies as the dominant factor" paragraph and the two sentences about ddof/Cauchy. The preceding paragraph ("Synthetic noise underestimates real drift") remains unchanged — it establishes the gap; the new paragraph explains it. The following paragraph ("Filtering dependent features") can also remain, since it makes an independent point about correlated features inflating apparent AUC.

The subsubsection opener (line 698–699) should be softened: instead of "introduces structured inter-feature dependencies that degrade attack accuracy," something like "degrades attack accuracy more steeply than either noise or missing features alone." This avoids pre-committing to inter-feature dependencies as the explanation before the evidence is presented.

## 2. Impact on other parts of the paper

### Subsection opener (line 620)

Currently ends with: "Real-world temporal drift, which introduces inter-feature dependencies absent from synthetic models, proves more damaging than either perturbation alone." Replace the causal clause. Suggested: "Real-world temporal drift proves more damaging than either perturbation alone, through a mechanism that no simple synthetic model can reproduce."

### Grey box (lines 772–782)

The second paragraph currently reads: "Real-world temporal drift degrades attack accuracy more steeply than synthetic models predict, because inter-feature dependencies drive the gap…" Replace with language reflecting the residual finding: the gap is driven by individual-specific nonlinear drift structure, meaning synthetic noise fundamentally underestimates the natural privacy protection conferred by temporal variation.

### First contribution bullet (lines 173–182)

Currently says drift "introduces structured inter-feature dependencies that cause significantly steeper performance drops than synthetic perturbations predict." Revise to say drift causes steeper drops, and that the destructive mechanism is individual-specific and nonlinear — residing in the component of drift that cannot be captured by any linear or marginal noise model. This is a stronger claim than the current one.

### Conclusion (lines 984–988)

Currently: "structured inter-feature dependencies in longitudinal data cannot be captured by independent Gaussian perturbations." Replace with: the degradation is driven by individual-specific nonlinear drift that resists all simple statistical modeling, not merely by inter-feature correlations. This also strengthens the conclusion — the finding is more fundamental than a violated independence assumption.

### Abstract (line 180, if present)

Check for any mention of inter-feature dependencies and update accordingly.

### Limitations paragraph (lines 1008–1015)

The sentence about extending the bound to "account for feature correlations" can remain as-is — the theoretical bound does assume independence, and extending it is still a valid future direction. But it no longer carries the weight of being the primary explanation for the synthetic-real gap.

## 3. The supporting figure

### What to show

A single figure with four bars (or a small 2×2 table inset) showing mean LLR AUC at t=1 for the four factorial conditions:

| Condition | Linear (β, α) | Residual (ε) | LLR AUC |
|:---------:|:-:|:-:|:-------:|
| self_self | self | self | 0.554 |
| self_nn | self | NN | 0.800 |
| nn_self | NN | self | 0.545 |
| nn_nn | NN | NN | 0.799 |

### Design

A grouped bar chart is the most readable format. The x-axis has two groups: "Self residual" and "NN residual." Within each group, two bars: "Self linear" and "NN linear." This makes the one-sided effect visually immediate — the two bars within each group are nearly identical (linear component doesn't matter), while the two groups are far apart (residual determines everything).

Add a dashed horizontal line at ~0.56 (real drift) and optionally at ~0.985 (Gaussian noise) for reference.

A small schematic inset or caption note should explain the decomposition: δ = β_k · x_k(0) + α_k + ε_k, with β and α fit per-individual.

### Why this figure works

It communicates the entire result at a glance without requiring the reader to parse a long table or follow a chain of experiments. The visual pattern — two pairs of nearly identical bars at very different heights — is unmistakable. A reviewer can look at this figure and immediately understand (a) the experiment design and (b) the conclusion, without needing to read a paragraph of setup.

### Placement

In the temporal drift subsubsection, immediately after (or alongside) the new paragraph. It replaces no existing figure — it is a net addition, but a small one (single-column width is sufficient).
