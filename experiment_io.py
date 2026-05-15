"""Reproducibility + figure-data persistence helpers shared by the wired
experiment scripts (Noise.py, Ordered_Noise.py, Ordered_FitBit.py,
Ordered_Timestamp.py).

Three things:

  parse_flags(argv)  pops `--seed N` and `--replot PATH` out of argv in place,
                     so the existing positional-arg parsing in each script
                     keeps working unchanged. Default seed is DEFAULT_SEED.

  seed_all(seed)     seeds the numpy and stdlib globals. That's enough:
                     sklearn ShuffleSplit(random_state=None), pandas
                     df.sample(...) and scipy .rvs(...) all consume from
                     numpy's global state when no random_state is passed.

  save_figdata / load_figdata
                     write/read `<output>.npz` + `<output>.meta.json` next
                     to the figure PDF.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

DEFAULT_SEED = 42
RESULTS_DIR = "results"


def resolve_output_path(path: str | None) -> str | None:
    """Bare filenames (no directory component) land in results/. Absolute
    paths and paths that already include a directory are left untouched.
    Creates results/ on demand."""
    if path is None or path == "":
        return path
    if os.path.dirname(path):
        return path
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, path)


def resolve_input_path(path: str) -> str:
    """If `path` doesn't exist as given but exists under results/, return the
    results/ version. Lets --replot fig2a_D3_case.pdf find results/fig2a_D3_case.npz."""
    if os.path.exists(path):
        return path
    if not os.path.dirname(path):
        candidate = os.path.join(RESULTS_DIR, path)
        if os.path.exists(candidate):
            return candidate
        # also try resolving via the npz extension
        base, ext = os.path.splitext(candidate)
        if ext.lower() in (".pdf", ""):
            npz_candidate = base + ".npz"
            if os.path.exists(npz_candidate):
                return candidate
    return path


def parse_flags(argv: list[str]) -> dict:
    """Pop `--seed N` and `--replot PATH` from argv (in place). Returns a
    dict {"seed": int, "replot": str | None}. Unknown flags are left alone."""
    seed = DEFAULT_SEED
    replot = None

    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok == "--seed" and i + 1 < len(argv):
            seed = int(argv[i + 1])
            del argv[i:i + 2]
            continue
        if tok.startswith("--seed="):
            seed = int(tok.split("=", 1)[1])
            del argv[i]
            continue
        if tok == "--replot" and i + 1 < len(argv):
            replot = argv[i + 1]
            del argv[i:i + 2]
            continue
        if tok.startswith("--replot="):
            replot = tok.split("=", 1)[1]
            del argv[i]
            continue
        i += 1

    return {"seed": seed, "replot": replot}


def seed_all(seed: int) -> None:
    import random as _random
    np.random.seed(seed)
    _random.seed(seed)


def _coerce_array(value):
    """Best-effort cast to ndarray; falls back to object dtype for ragged."""
    try:
        return np.asarray(value)
    except ValueError:
        return np.asarray(value, dtype=object)


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _paths_for(output_pdf: str) -> tuple[str, str]:
    base, _ = os.path.splitext(output_pdf)
    return base + ".npz", base + ".meta.json"


def save_figdata(output_pdf: str, data: dict, meta: dict | None = None) -> None:
    """Write `<output>.npz` + `<output>.meta.json` next to output_pdf.
    Bare filenames are written under results/."""
    output_pdf = resolve_output_path(output_pdf)
    npz_path, meta_path = _paths_for(output_pdf)

    coerced = {k: _coerce_array(v) for k, v in data.items()}
    np.savez(npz_path, **coerced)

    full_meta = {
        "script": os.path.basename(sys.argv[0]) if sys.argv else None,
        "argv": list(sys.argv),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
    }
    if meta:
        full_meta.update(meta)
    with open(meta_path, "w") as fh:
        json.dump(full_meta, fh, indent=2, default=str)

    print(f"Saved figure data to {npz_path} (+ {os.path.basename(meta_path)})")


def load_figdata(path: str) -> tuple[dict, dict]:
    """Load a saved dump. `path` may be the .npz, the .pdf, or the basename.
    Falls back to looking under results/ if the bare path doesn't exist."""
    path = resolve_input_path(path)
    base, ext = os.path.splitext(path)
    if ext.lower() == ".npz":
        npz_path = path
    elif ext.lower() == ".pdf" or ext == "":
        npz_path = base + ".npz"
    else:
        npz_path = path
    npz_path = resolve_input_path(npz_path)

    with np.load(npz_path, allow_pickle=True) as nz:
        data = {k: nz[k] for k in nz.files}

    meta_path = os.path.splitext(npz_path)[0] + ".meta.json"
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    return data, meta
