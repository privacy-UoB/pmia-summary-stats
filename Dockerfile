# Reproducible environment for the PETs artifact
# "Revisiting Assumptions for Membership Inference on Summary Statistics".
#
# Base image ships uv + a pinned CPython in the project's supported range
# (>=3.11,<3.14). uv installs every dependency from the committed uv.lock,
# so the build is fully reproducible and needs no network access at run time.
#
# Build:  docker build -t pmia-summary-stats .
# Run:    docker run --rm -it pmia-summary-stats
#         (then, inside the container, e.g.  uv run python fig_min_error.py)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /artifact

# Copy the whole artifact (datasets included) and install the locked deps.
COPY . .
RUN uv sync --frozen

# uv-managed venv is at /artifact/.venv; `uv run` uses it automatically.
ENV UV_NO_SYNC=1

CMD ["bash"]
