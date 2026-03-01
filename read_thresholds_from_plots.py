"""Extract α_{0.6} and α* thresholds from the existing figure PDFs.

Converts each PDF to an image via pdftoppm, identifies the plot area,
traces the L1 and LLR curves by scanning pixel colors, maps pixel
positions back to data coordinates, then interpolates crossings.

Axis limits are calibrated from two known tick positions per plot type,
determined from manual inspection of the tick mark pixel fractions.
"""
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from PIL import Image


def pdf_to_array(pdf_path, dpi=300):
    """Convert first page of PDF to numpy RGB array using pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf_path), str(prefix)],
            check=True, capture_output=True
        )
        img = Image.open(f"{prefix}.png").convert("RGB")
        return np.array(img)


def find_plot_bounds(img):
    """Find the plot area bounding box by looking for the black border lines."""
    gray = np.mean(img, axis=2)
    col_dark_count = np.sum(gray < 50, axis=0)
    dark_cols = np.where(col_dark_count > img.shape[0] * 0.3)[0]
    row_dark_count = np.sum(gray < 50, axis=1)
    dark_rows = np.where(row_dark_count > img.shape[1] * 0.3)[0]
    if len(dark_cols) < 2 or len(dark_rows) < 2:
        raise ValueError("Could not find plot axes")
    return dark_cols[0], dark_cols[-1], dark_rows[0], dark_rows[-1]


def extract_curve(img, left, right, top, bottom, color_rgb, tolerance=60):
    """Extract curve y-positions by scanning columns for matching color pixels."""
    region = img[top + 2:bottom - 2, left + 2:right - 2].astype(np.int16)
    target = np.array(color_rgb, dtype=np.int16)
    diff = np.abs(region - target)
    mask = np.all(diff < tolerance, axis=2)

    xs, ys = [], []
    for col_idx in range(mask.shape[1]):
        matching = np.where(mask[:, col_idx])[0]
        if len(matching) > 0:
            xs.append(col_idx + left + 2)
            ys.append(np.median(matching) + top + 2)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def pixel_to_data(px_x, px_y, left, right, top, bottom,
                  x_min_log, x_max_log, y_min, y_max):
    """Convert pixel coordinates to data coordinates (log x, linear y)."""
    x_frac = (px_x - left) / (right - left)
    data_x = 10 ** (x_min_log + x_frac * (x_max_log - x_min_log))
    y_frac = (bottom - px_y) / (bottom - top)
    data_y = y_min + y_frac * (y_max - y_min)
    return data_x, data_y


def interpolate_crossing(x, y, target, direction="below"):
    """Find x where y crosses target."""
    for i in range(1, len(x)):
        if direction == "below" and y[i] < target <= y[i - 1]:
            frac = (target - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] + frac * (x[i] - x[i - 1])
        if direction == "above" and y[i] >= target > y[i - 1]:
            frac = (target - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] + frac * (x[i] - x[i - 1])
    return None


def smooth_curve(x, y, window=15):
    """Simple moving average to smooth out pixel noise."""
    if len(y) < window:
        return x, y
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode='valid')
    offset = window // 2
    return x[offset:offset + len(y_smooth)], y_smooth


def analyze_plot(pdf_path, label, x_range_log, y_range=(0.45, 1.0),
                 l1_color=(0, 0, 255), llr_color=(255, 0, 0),
                 color_tolerance=60):
    """Extract thresholds from a single plot PDF."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {label} ({Path(pdf_path).name})")

    img = pdf_to_array(pdf_path)
    left, right, top, bottom = find_plot_bounds(img)

    l1_px_x, l1_px_y = extract_curve(img, left, right, top, bottom, l1_color, color_tolerance)
    llr_px_x, llr_px_y = extract_curve(img, left, right, top, bottom, llr_color, color_tolerance)
    print(f"  L1: {len(l1_px_x)} pts, LLR: {len(llr_px_x)} pts")

    if len(l1_px_x) == 0 or len(llr_px_x) == 0:
        print("  WARNING: Could not extract curves!")
        return None

    l1_x, l1_y = pixel_to_data(l1_px_x, l1_px_y, left, right, top, bottom,
                                x_range_log[0], x_range_log[1], y_range[0], y_range[1])
    llr_x, llr_y = pixel_to_data(llr_px_x, llr_px_y, left, right, top, bottom,
                                  x_range_log[0], x_range_log[1], y_range[0], y_range[1])

    l1_x, l1_y = smooth_curve(l1_x, l1_y)
    llr_x, llr_y = smooth_curve(llr_x, llr_y)

    # Baseline AUC sanity check
    n_base = max(1, len(l1_y) // 20)
    l1_base = np.mean(l1_y[:n_base])
    llr_base = np.mean(llr_y[:n_base])
    print(f"  Baseline AUC: L1={l1_base:.3f}, LLR={llr_base:.3f}")

    a06_l1 = interpolate_crossing(l1_x, l1_y, 0.6, "below")
    a06_llr = interpolate_crossing(llr_x, llr_y, 0.6, "below")

    # α*: interpolate both onto common grid
    common_x = np.logspace(x_range_log[0], x_range_log[1], 1000)
    l1_interp = np.interp(common_x, l1_x, l1_y)
    llr_interp = np.interp(common_x, llr_x, llr_y)
    a_star = interpolate_crossing(common_x, l1_interp - llr_interp, 0, "above")

    fmt = lambda v: f"{v:.2f}" if v is not None else ">max"
    print(f"  α_{{0.6}} LLR={fmt(a06_llr)}, L1={fmt(a06_l1)}, α*={fmt(a_star)}")

    return {"label": label, "a06_llr": a06_llr, "a06_l1": a06_l1, "a_star": a_star}


# ── Axis calibration ──────────────────────────────────────────────────
# Determined from tick mark pixel positions at 300 DPI.
#
# Cross-sectional (D3, D17) and FitBit use deviation_range = logspace(-2, 2, 50).
# matplotlib auto-pads the log axis. From tick analysis:
#   10^-2 at pixel frac ~0.071, 10^2 at frac ~0.970
#   => axis limits: 10^-2.316 to 10^2.133
#
# Timestamp uses ranges[2] = arange(0, 20, 0.1), so data spans 0.1..19.9.
# From tick analysis:
#   10^-1 at frac ~0.165, 10^1 at frac ~0.954
#   => axis limits: 10^-1.418 to 10^1.117

XRANGE_CROSS = (-2.317, 2.154)  # D3, D17, FitBit (logspace(-2,2,50))
XRANGE_TIMESTAMP = (-1.179, 1.382)  # Timestamp (arange(0,20,0.1) on log)


if __name__ == "__main__":
    results = []

    for pdf, label in [
        ("fig2a_D3_case.pdf", "D3 case"),
        ("fig2b_D3_random.pdf", "D3 random"),
        ("fig2ai_D17_case_dev.pdf", "D17 case"),
        ("fig2bi_D17_random_dev.pdf", "D17 random"),
    ]:
        if Path(pdf).exists():
            r = analyze_plot(pdf, label, x_range_log=XRANGE_CROSS,
                             l1_color=(0, 0, 255), llr_color=(255, 0, 0))
            if r:
                results.append(r)

    if Path("fig2c_Timestamp.pdf").exists():
        r = analyze_plot("fig2c_Timestamp.pdf", "Timestamp",
                         x_range_log=XRANGE_TIMESTAMP,
                         l1_color=(31, 119, 180), llr_color=(255, 127, 14),
                         color_tolerance=40)
        if r:
            results.append(r)

    if Path("fig2d_FitBit.pdf").exists():
        r = analyze_plot("fig2d_FitBit.pdf", "FitBit",
                         x_range_log=XRANGE_CROSS,
                         l1_color=(31, 119, 180), llr_color=(255, 127, 14),
                         color_tolerance=40)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary of thresholds read from plots:\n")
    fmt = lambda v: f"{v:.2f}" if v is not None else ">max"
    print(f"{'Config':<20} {'α₀.₆ LLR':>10} {'α₀.₆ L1':>10} {'α*':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<20} {fmt(r['a06_llr']):>10} {fmt(r['a06_l1']):>10} {fmt(r['a_star']):>10}")
