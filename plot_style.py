"""Shared plot styling for the unified legend convention.

Convention (chosen as Option 18 in plot_legend_options.py):
  - Attack metric (L1 / LLR)   -> colour  (L1 = blue, LLR = red)
  - Pool (case / random)       -> linestyle + marker fill
                                  (case = solid + filled o ;
                                   random = dashed + hollow o)
  - Score (AUC / TPR@FPR)      -> stacked sub-plots (AUC top, TPR bottom)
                                  -- never twin axes.

Use `line_kwargs(metric, pool)` everywhere a series is plotted, and
`stacked_auc_tpr(...)` anywhere a script previously called ax.twinx().
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import colormaps


L1_COLOR  = "#1f77b4"   # matplotlib tab:blue
LLR_COLOR = "#d62728"   # matplotlib tab:red

METRIC_COLOR: dict[str, str] = {"L1": L1_COLOR, "LLR": LLR_COLOR}
POOL_LINESTYLE: dict[str, str] = {"case": "-", "random": "--"}


def line_kwargs(metric: str,
                pool: str | None = None,
                *,
                marker: str | None = "o",
                linewidth: float = 1.8,
                **extra) -> dict:
    """matplotlib `plot` kwargs for the unified convention.

    metric : 'L1' or 'LLR'         -> colour.
    pool   : 'case' | 'random' | None
        case   -> solid line + filled marker.
        random -> dashed line + hollow marker.
        None   -> solid line, marker filled if drawn (no pool dim in this plot).
    marker : marker shape, or None to suppress markers entirely.
    Any extra kwargs (e.g. markersize=4) override the defaults.
    """
    if metric not in METRIC_COLOR:
        raise ValueError(f"metric must be 'L1' or 'LLR', got {metric!r}")
    color = METRIC_COLOR[metric]

    kw: dict = {"color": color, "linewidth": linewidth}
    if pool is None:
        kw["linestyle"] = "-"
        if marker is not None:
            kw["marker"] = marker
            kw["markerfacecolor"] = color
            kw["markeredgecolor"] = color
    else:
        if pool not in POOL_LINESTYLE:
            raise ValueError(f"pool must be 'case', 'random', or None, "
                             f"got {pool!r}")
        kw["linestyle"] = POOL_LINESTYLE[pool]
        if marker is not None:
            kw["marker"] = marker
            kw["markeredgecolor"] = color
            kw["markerfacecolor"] = color if pool == "case" else "white"

    kw.update(extra)
    return kw


def stacked_auc_tpr(figsize: tuple[float, float] = (7.0, 6.0),
                    height_ratios: tuple[float, float] = (1.0, 1.0)):
    """Two vertically stacked axes -- AUC on top, TPR@FPR on bottom.

    Returns (fig, ax_auc, ax_tpr). The x-axis is shared, top tick labels
    hidden. Use this in place of ``ax1 = plt.subplots(); ax2 = ax1.twinx()``.
    """
    fig, (ax_auc, ax_tpr) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios)},
    )
    return fig, ax_auc, ax_tpr


def noise_sequential(metric: str, n_levels: int) -> list[tuple[float, float, float, float]]:
    """Light-to-dark colour ramp anchored at the metric hue.

    For scripts (Ordered_Noise.py) that plot N noise-level curves of a single
    metric. Returns `n_levels` RGBA tuples drawn from a 'Blues' (L1) or 'Reds'
    (LLR) sequential colormap, sampled in [0.35, 0.95] so the lightest end is
    still visible against a white background.
    """
    if metric not in METRIC_COLOR:
        raise ValueError(f"metric must be 'L1' or 'LLR', got {metric!r}")
    cmap_name = "Blues" if metric == "L1" else "Reds"
    cmap = colormaps[cmap_name]
    if n_levels == 1:
        return [cmap(0.75)]
    lo, hi = 0.35, 0.95
    step = (hi - lo) / (n_levels - 1)
    return [cmap(lo + i * step) for i in range(n_levels)]
