# Artifact Appendix

Paper title: **Revisiting Assumptions for Membership Inference on Summary Statistics**

Requested Badge(s):
  - [X] **Available**
  - [X] **Functional**
  - [X] **Reproduced**

<!-- Authors can provide this content _either_ as a separate file in their artifact
_or_ as part of their existing documentation (e.g., `README.md`). In the latter
case, you should have the same section titles as in this template.

This template includes several placeholders. When filling in this template for
their artifact, the authors should:

1. Remove this note.
2. Delete the sections that are _not_ required for the badge(s) they are
   applying for.
3. Omit suffixes of the form "(required/encouraged for badge ...)" from the
   section titles.
4. Authors should not leave the placeholder descriptions initially provided with
   this file into the submitted version with their artifact.

While this template is provided for artifact review, you should write your
instructions for someone trying to reuse your artifact in the future (i.e., not
an artifact reviewer). -->

## Description
<!-- Replace this with the following:

1. List the paper that the artifact relates to (i.e., paper title, authors,
   year, or even a BibTex cite).
2. A short description of your artifact and how it is relevant to your paper. -->

- **Paper:** *Revisiting Assumptions for Membership Inference on Summary Statistics*
- **Authors:** P. Berrang, M. Ryan, K. Wooldridge
- **Year:** 2026

This artifact contains the datasets and code required for all plots in the
above paper. The two miRNA datasets used were downloaded from the Gene 
Expression Omnibus and the FitBit was downloaded from Kaggle. The code 
both computes the summary statistic results to mimic those used in published 
medical papers and the performance of an inference attack by an adversary on 
the same statistics. All participants in the datasets are used as potential 
targets. Where adversarial access to data is assumed to be noisy or missing, 
we manually add Gaussian noise and limit the dataset features to match.

### Security/Privacy Issues and Ethical Concerns

<!-- Replace this with a description of security or privacy risks that your artifact
may hold for the machine of the person trying to evaluate or reuse your
artifact. This is especially relevant for artifacts that _disable a security
mechanism_, such as a firewall, ASLR, etc., to demonstrate an attack, as well as
to artifacts that _run vulnerable code_, such as exploits, malware samples,
etc., to demonstrate a vulnerability.

User study artifacts that include anonymized transcripts or survey responses
should list the ethical review / IRB process followed to obtain participants'
consent to publishing this anonymized dataset. They may also list how
participants were compensated. -->

No security or privacy risks to the user.

As stated in the paper,
"We collected no additional data.
For the miRNA datasets, the local ethics committees approved the
original studies and patients gave informed consent before sample
collection. The Fitbit dataset is licensed CC0 and was collected from
users who consented to share their data."

## Basic Requirements

<!-- For both sections below, if you are giving reviewers remote access to special
hardware (e.g., Intel SGX v2.0) or proprietary software (e.g., Matlab R2025a)
for the purpose of the artifact evaluation, do not provide these instructions
here but rather in the corresponding submission field on HotCRP. -->

### Hardware Requirements

<!-- Replace this with the following:

1. A list of the _minimal hardware requirements_ to execute your artifact. If no
   specific hardware is needed, then state "Can run on a laptop (No special
   hardware requirements)". You may state how a researcher could gain access to
   that hardware, e.g., by buying, renting, or even emulating it.
2. When applying for the "Reproduced" badge, list _the specifications of the
   hardware_ on which the experiments reported in the paper were performed. This
   is especially relevant in cases were results might be influenced by the
   hardware used (e.g., latency, bandwidth, throughput experiments, etc.). -->

Can run on a laptop (no special hardware requirements).

Hardware specifications when performing experiments for the corresponding paper:

- MacBook Air (M1)
- 8-core GPU
- 8 GB memory
- 256 GB storage

### Software Requirements

<!-- Replace this with the software required to run your artifact and its versions,
as follows.

1. List the OS you used to run your artifact, along with its version (e.g.,
   Ubuntu 22.04). If your artifact can only run on a specific OS or a specific
   OS version, list it and explain why here. In general, your artifact reviewers
   will probably have access to a machine with a different OS or different OS
   version than yours; they should still be able to run appropriately packaged
   artifacts.
2. List the OS packages that your artifact requires, along with their versions.
3. Artifact packaging: If you use a container runtime (e.g., Docker) to run the
   artifact, list the container runtime and its version (e.g., Docker 23.0.3).
   If you use VMs, list the hypervisor (e.g., VirtualBox) to run the artifact.
4. List the programming language compiler or interpreter you used to run your
   artifact (e.g., Python 3.13.7). Your Docker image or VM image should have
   this version of the programming languages installed already. Your Dockerfile
   should start from a base image with this programming language version.
5. List packages that your artifact depends on, along with their versions. For
   example, Python-based privacy-preserving machine learning artifacts typically
   require `numpy`, `scipy`, etc. You may point to a file in your artifact with
   this list, such as a `requirements.txt` file. If you rely on proprietary
   software (e.g. Matlab R2025a), list this here and consider providing access
   to reviewers through HotCRP.
6. List any Machine Learning Models required to run your artifact, along with
   their versions. If your model is hosted on a different repository, such as on
   Zenodo, then your artifact should download it automatically (same for
   datasets). If a required ML model is _not_ in your artifact, provide a dummy
   model to demonstrate the functionality of the rest of your artifact.
7. List any datasets required to run your artifact. If any required dataset is
   not in your artifact, you should provide a synthetic dataset that showcases
   the expected data format. -->

OS used to run artifact: macOS Tahoe 26.3. The artifact is pure Python with no
OS-specific code and also runs on Linux and Windows.

OS packages required: none. The only prerequisite is
[`uv`](https://docs.astral.sh/uv/) (the Python package/version manager). `uv`
provisions the correct Python interpreter and every dependency from `uv.lock`,
so no system Python or manual `pip install` is needed.

Artifact packaging: a `Dockerfile` is provided for a fully self-contained,
reproducible environment (recommended for review — see "Set up the
environment"). It is built and run with Docker (tested with Docker 29.5); no
other container runtime is required. Reviewers who prefer a native install can
instead use `uv` directly.

Programming language interpreter: Python >=3.11, <3.14 (installed automatically
by `uv`, or baked into the Docker image, from the pin in `pyproject.toml`).

Shell (native runs only): the `run_*.sh` helper scripts use bash associative
arrays and require bash >= 4.0. macOS ships bash 3.2 by default, so either
install a newer bash (`brew install bash`) or run the underlying
`uv run python ...` commands directly. The Docker image (Debian) already
includes a recent bash, so Docker users are unaffected.

Packages depended on: numpy, scipy, matplotlib, pandas, scikit-learn, ipython.
Exact pinned versions are in `uv.lock`.

Machine Learning Models required: none.

Datasets required (all bundled under `Datasets/` — nothing to download):

- `GSE61741_series_matrix.csv` (cross-sectional miRNA dataset)
- `independent_90.csv` (our list of cross-sectional miRNAs with 90% independence)
- `GSE68951_series_matrix.txt` (longitudinal miRNA dataset)
- `dailyActivity_merged.csv` (FitBit dataset)
- `Diseased_miRNAs.txt` (our list of up/down regulated diseased miRNAs)

### Estimated Time and Storage Consumption

<!-- Replace the following with estimated values for:

- The overall human and compute times required to run the artifact.
- The overall disk space consumed by the artifact.

This helps reviewers schedule the evaluation in their time plan and others in
general to see if everything is running as intended. This should also be
specified at a finer granularity for each experiment (see below). -->

**Human time:** roughly 30-60 minutes to set up the environment and launch all
experiments (each is a single command).

**Compute time:** most figures complete in minutes on a recent laptop. The
exception is the random-pool figures, which average over 20 independent pool
draws and take on the order of one to a few hours each (Figure 2b, Figure 9b,
Figure 3b, and the pool-size / 19-disease sweeps). Reproducing *every* figure
end-to-end is therefore on the order of ~8-12 compute-hours, dominated by those
random-pool sweeps; any single figure can be reproduced much faster. All
timings below were measured on the reference MacBook Air (M1).

**Reducing runtime for a quick functional check:** the Monte-Carlo work scales
linearly with the number of random orderings and pool draws, both overridable
via environment variables. For example,
`NUM_ORDERS=200 NUM_POOLS=4 uv run python Ordered.py D3 0` finishes in well
under a minute and produces a qualitatively identical (noisier) figure; the
`experiment_nsweep.py` / `experiment_19disease.py` scripts expose the same knob
as `--iterations`. Use the full counts only for publication-quality curves.

**Storage:** negligible. The repository is ~15 MB (after the bundled datasets),
and generated figures/data add only a few MB to `results/`.

## Environment

<!-- In the following, describe how to access your artifact and all related and
necessary data and software components. Afterward, describe how to set up
everything and how to verify that everything is set up correctly. -->

### Accessibility

<!-- Replace the following by a description of how to access your artifact via
persistent sources. Valid hosting options are institutional and third-party
digital repositories (e.g., GitHub, Gitlab, BitBucket, Zenodo, Figshare, etc.).
Please do not use personal web pages or cloud storage services like Google
Drive, Dropbox, etc.

Note that once your artifact evaluation is finalized and a badge decision has
been made, artifact chairs will collect a stable and persistent reference to
your artifact to list on the website. For version-controlled repositories (e.g.,
Git repositories), this will be a specific commit-id or tag.

You _should not_ link to a specific commit here at submission time, as changes
will likely happen during the evaluation process to address the reviewers'
feedback, resulting in the link being out-of-date. Instead, you may link to the
latest commit in your branch (e.g. main) as follows:
https://github.com/PoPETS-AEC/example-docker-python-pip/tree/main -->

https://github.com/privacy-UoB/pmia-summary-stats/tree/main

There are two ways to set up the artifact. Both were tested from a clean state
on the reference machine. **Option A (Docker) is recommended for review**, as it
needs nothing on the host beyond Docker itself.

#### Option A — Docker (recommended)

With Docker installed, from the cloned repository:

```bash
git clone https://github.com/privacy-UoB/pmia-summary-stats.git
cd pmia-summary-stats
docker build -t pmia-summary-stats .          # ~2-3 min on first build
docker run --rm -it pmia-summary-stats        # drops you into a shell in /artifact
```

The image bundles the pinned Python, all dependencies (from `uv.lock`), and the
datasets, so it is self-contained and needs no network access at run time.
Inside the container, run any experiment with `uv run python <script>` (see
below). To recover generated figures onto the host, mount a directory, e.g.
`docker run --rm -it -v "$PWD/results:/artifact/results" pmia-summary-stats`.

#### Option B — native install with uv

Install `uv` (see https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/privacy-UoB/pmia-summary-stats.git
cd pmia-summary-stats
uv sync          # installs the pinned Python (3.11-3.13) and all dependencies
```

`uv sync` downloads the correct Python interpreter and every dependency listed
in `uv.lock`; it typically completes in well under a minute on a warm cache.

In either case, experiment scripts are invoked with `uv run python <script>`.
Figures and their underlying data are written to the `results/` directory,
which is created on demand.

### Testing the Environment

Two quick checks confirm the environment is working. Run them with `uv run` —
inside the Docker container (`docker run --rm -it pmia-summary-stats`) or
natively after `uv sync`; both behave identically.

1. Regenerate Figure 1 (deterministic, runs in ~1 second, no datasets):

   ```bash
   uv run python fig_min_error.py
   ```

   This writes `results/MinimumError_evolution_delta-n.pdf` and
   `results/MinimumError_evolution_A-n.pdf`.

2. Run the numerical regression test, which loads the cross-sectional miRNA
   dataset and checks the vectorised attack pipelines against the reference
   implementation:

   ```bash
   uv run python tests/test_fast_paths_regression.py
   ```

   Expected output ends with `ordered LLR: OK`, `ordered L1: OK`, and
   `noise dual: OK` (each comparison reports a small max absolute difference,
   well within tolerance). If both checks pass, all required software is
   functioning.

## Artifact Evaluation

<!-- This section should include all the steps required to evaluate your artifact's
functionality and validate your paper's key results and claims. Therefore,
highlight your paper's main results and claims in the first subsection. And
describe the experiments that support your claims in the subsection after that. -->

### Main Results and Claims

<!-- List all your paper's results and claims that are supported by your submitted
artifacts. -->

#### Main Result 1: Minimum Error Bounds

<!-- Describe the results in 1 to 3 sentences. Mention what the independent and
dependent variables are; independent variables are the ones on the x-axes of
your figures, whereas the dependent ones are on the y-axes. By varying the
independent variable (e.g., file size) in a given manner (e.g., linearly), we
expect to see trends in the dependent variable (e.g., runtime, communication
overhead) vary in another manner (e.g., exponentially). Refer to the related
sections, figures, and/or tables in your paper and reference the experiments
that support this result/claim. See example below. -->

Our paper gives a theoretical analysis that lower-bounds the minimum error of
successful inference (Theorem 1). The bound shows that inference becomes harder
— the minimum error rises towards 1/2 — as the pool size n grows, and becomes
easier — the minimum error falls towards 0 — as the per-feature shift delta or
the number of affected features |A| grows.
We plot these relationships in [Experiment 1](#experiment-1-minimum-error-plots).
We first range n through the natural numbers from 2 to 501 for five values of
delta (0, 0.1, 0.2, 0.3 and 0.5) at fixed |A| = 5; we then range n over the
same numbers for four values of |A| (0, 5, 10 and 20) at fixed delta = 0.5
(both with m = 500, sigma^2 = 1). Figure 1a shows the minimum error rising with
n and falling as delta increases; Figure 1b shows it rising with n and falling
as |A| increases. We report these results in "Figure 1a" and "Figure 1b" in our
paper.

#### Main Result 2: Increasing Adaptive Noise

<!-- Our paper claims that when varying the file size linearly, the runtime also
increases linearly. This claim is reproducible by executing our
[Experiment 2](#experiment-2-example-name). In this experiment, we change the
file size linearly, from 2KB to 24KB, at intervals of 2KB each, and we show that
the runtime also increases linearly, reaching at most 1ms. We report these
results in "Figure 1a" and "Table 3" (Column 3 or Row 2) of our paper. -->

Our paper demonstrates that both L1 and LLR attacks withstand substantial noise
per feature of the target data. The L1 distances attack is more robust than LLR.
As noise massively increases the attack eventually degrades to random-guessing.
We show this in [Experiment 2](#experiment-2-mia-under-increasing-gaussian-noise).
Here we multiply the standard deviation of each feature in the dataset by alpha,
where alpha varies logarithmically in powers of 10 from -2 to 2 over 50 intervals
for "Figure 2a", "Figure 2b" & "Figure 2d" and linearly from 0 to 20 with increments
of 0.1 for "Figure 2c". All results show the AUC and TPR @ 1% FPR decrease as the
noise amplifies.

#### Main Result 3: Decreasing Features

The paper investigated how L1 and LLR perform when the available data is partial 
or incomplete. The AUC and TPR @ 1% FPR are both fairly tolerant  under substantial 
feature reduction and rapdily decrease when the available features drop, this 
degredation begins at 25% of the maximum in our example.
We show this in [Experiment 3](#experiment-3-mia-with-incomplete-data). We decrease 
the miRNA count by intervals of 1 from the maximum down to 2 features. In doing so,
the AUC and TPR @ 1% FPR remain steady before sharply decreasing in performance.
These results can be seen in "Figure 3a" and "Figure 3b".

#### Main Result 4: Real World Longiduinal Data

We checked the MIA performance on real-world longitudinal data to compare against
previous work that used synthetic noise models. Our results showed much worse scores
than expected when using data from any other time than the original statistics.
These results can be reproduced in [Experiment 4](#experiment-4-mia-on-longitudinal-data).
To ensure even longitudinal coverage does not introduce dependencies, we clip the 
submissions after the first 8 consecutive results per person. Then we compare the
MIA scores using data from the later timestamps against the summary statistics of 
the initial submission. This is seen in "Figure 4".

#### Main Result 5: Pool Size

We claim that both the number of features in the study pool and the conflation of
disease signal impacts the MIA performance. We find that as the pool size decreases, 
the performance decreases also; further that the random pools consistently score 
worse than the diseased case pools.
These results can be reproduced in [Experiment 5](#experiment-5-pool-size-sweep).
We increase the pool size n by fixed amounts small increments between 2 and 14 for 
both the random and case pools for three diseases (Prostate Cancer, Wilms Tumor 
and Ovarian Cancer), as seen in the three plots in "Figure 5".

#### Main Result 6: Case v Random

Our paper states that the disease signal contributes to the summary statistics and 
hence improves inference accuracy. To do this we evaluated the MIA performance for 
random pools and case pools to determine that random pools are influenced only by 
size whereas diseased case pools perform consistently higher. We also show that 
different diseases perform better or worse than others, depending on how strong the 
miRNA disease signals influence individual miRNA expressions.
These results are reproduceable in [Experiment 6](#experiment-6-19-disease-validation).
We determine n by the counting the number of miRNAs relevant only to each of the 19 
diseased case pools, and sampling same-size miRNAs to use for the random pools. Then
we plot both scatter diagrams, "Figure 6A" comparing case vs random over increasing 
n on the x axis and "Figure 6B" comparing the random pool performance on the x axis 
against case pool performance on the y axis.

#### Main Result 7: Stratification

We claim that the performance of MIAs are conflated by disease-signals, and hence
non-members of the data are conflated with the shared conditions of those used in
the study.
We support this in [Experiment 7](#experiment-7-stratifying-diseased-features).
We limit the cross-sectional disease pools to only use the up/down-regulared
miRNAs listed in `Diseased_miRNAs.txt`, then limit the corresponding random pool
curves to sample the same number of miRNAs not relevant to the disease. Then we 
plot the MIA performance for decreasing miRNA count. We start with the maximum 
number of features and reduce this by intervals of 1 down to 2. We show these
results for Wilms Tumor in "Figure 7a", Renal Cancer in "Figure 7b" and Ovarian 
Cancer in "Figure 7c".

#### Main Result 8: Longitudinal Drift

We concluded that real longitudinal drift is difficult to replicate. We attempted 
to do this using three different synthetic models and demonstrated that none of 
them could reproduce the AUC degredation seen in the real world scenario using 
miRNA data collected from a later date.
This is backed up by [Experiment 8](#experiment-8-attempts-at-replicating-real-drift).
We compare the values at the 1st timepoint (as in the closest date to when the 
original data is stated in the summary statistics) and record these in "Table 3". 
The real drift is in row 1, the independent Gaussian model on row 2, the Leave 
One Out conditional mean on row 3 and the Nearest Neighbour's real drift on row 4.

### Experiments
<!-- List each experiment to execute to reproduce your results. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes to execute in human and compute times (approximately).
 - How much space it consumes on disk (approximately) (omit if <10GB).
 - Which claim and results does it support, and how. -->

<!-- #### Experiment 1: Minimum Error plots
- Time: replace with estimate in human-minutes/hours + compute-minutes/hours.
- Storage: replace with estimate for disk space used (omit if <10GB).

Provide a short explanation of the experiment and expected results. Describe
thoroughly the steps to perform the experiment and to collect and organize the
results as expected from your paper (see example below). Use code segments to
simplify the workflow, as follows.

```bash
python3 experiment_1.py
``` -->

#### Experiment 1: Minimum Error plots

- Time: 2 human-minutes + <1 compute-minute

This experiment reproduces [Main Result 1](#main-result-1-minimum-error-bounds).
The following plots the Theorem 1 bound directly from its closed form with the
parameters used in the paper (`m=500`, `sigma^2=1`; panel a fixes `|A|=5` and
sweeps `delta`, panel b fixes `delta=0.5` and sweeps `|A|`):

```bash
uv run python fig_min_error.py
```

This writes the two panels of Figure 1 to:
- `results/MinimumError_evolution_delta-n.pdf`  ("Figure 1a")
- `results/MinimumError_evolution_A-n.pdf`       ("Figure 1b")

It is deterministic (no datasets, no randomness) and runs in about a second.
Re-render from the saved data without recomputing via
`uv run python fig_min_error.py --plot-only`.

#### Experiment 2: MIA under Increasing Gaussian Noise

<!-- - Time: 10 human-minutes + 3 compute-hours
- Storage: 20GB

This example experiment reproduces
[Main Result 2: Example Name](#main-result-2-example-name), the following script
will run the simulation automatically with the different parameters specified in
the paper. (You may run the following command from the example Docker image.)

```bash
python3 main.py
```

Results from this example experiment will be aggregated over several iterations
by the script and output directly in raw format along with variances and
standard deviations in the `output-folder/` directory. You will also find there
the plots for "Figure 1a" in `.pdf` format and the table for "Table 3" in `.tex`
format. These can be directly compared to the results reported in the paper, and
should not quantitatively vary by more than 5% from expected results. -->


- Time: 5 human-minutes + ~4-6 compute-hours for the full set (each case-pool
  panel — Fig 2a, 2c, 2d, 9a — is ~15-20 compute-minutes; the random-pool
  panels Fig 2b and 9b average over 20 pools and are the long poles at
  ~1.5-5 compute-hours each). Reduce `NUM_ORDERS`/`NUM_POOLS` for a quick check.

This experiment reproduces [Main Result 2](#main-result-2-increasing-adaptive-noise).
The following gives the general command needed for all four plots in Figure 2:

```bash
uv run python Noise.py <dataset> <include_deviations> <disease> <pop_idx> <pool_idx> [random_sample_size] [output.pdf]
```

- For **Figure 2a**, `<disease>` is `D3`.
- For **Figure 2b**, `<pop_idx>` and `<pool_idx>` are `0`.
- For **Figure 2c**, `<dataset>` is `Timestamp`.
- For **Figure 2d**, `<dataset>` is `FitBit`.

The concrete invocations are wired up in `run_noise_experiments.sh`; for
example Figure 2a is `uv run python Noise.py miRNA true D3 1 1 _ fig2a_D3_case.pdf`.
The script averages over 2000 iterations and saves the pdf to `<output>`.

The noise- and crossover-threshold table (**Table 2**, "Summary of noise and
crossover thresholds") is generated directly — it does not need to be read off
the plots:

```bash
uv run python compute_thresholds.py
```

This prints the LaTeX table and writes `results/threshold_results.csv`, with
one row per configuration (D3 case/random, D17 case/random, Timestamp, FitBit;
the D17-random row corresponds to `random_sample_size = 20`).

#### Experiment 3: MIA with Incomplete Data

- Time: 5 human-minutes + ~2 compute-hours total (Fig 3a, case pool, ~6
  compute-minutes; Fig 3b, random pool with 20-pool averaging, ~1.5-2
  compute-hours). Reduce `NUM_ORDERS`/`NUM_POOLS` for a quick check.

This experiment corresponds to [Main Result 3](#main-result-3-decreasing-features).
The following is the general command needed for both plots in Figure 3:

```bash
uv run python Ordered.py <disease> <pool_idx> [random_sample_size] [output.pdf] [stratify]
```

- For **Figure 3a** (D3 case pool), `<pool_idx>` is `1`:
  `uv run python Ordered.py D3 1 _ fig3a_D3_case.pdf`.
- For **Figure 3b** (D3 random pool), `<pool_idx>` is `0`:
  `uv run python Ordered.py D3 0 _ fig3b_D3_random.pdf`.

This averages over 5000 random orderings (and, for the random pool, 20 pool
draws) and saves the pdf to `<output>`. Both invocations are also wired up in
`run_ordered_experiments.sh`.

#### Experiment 4: MIA on Longitudinal Data

- Time: 5 human-minutes + ~8-10 compute-minutes (single command, 5000 orderings).

This script matches [Main Result 4](#main-result-4-real-world-longiduinal-data).
The following will average over 5000 iterations and give the plot needed for 
"Figure 4":

```bash
uv run python Ordered_FitBit.py [output.pdf]
```

It is also available as `./run_misc_experiments.sh 4`.

#### Experiment 5: Pool Size Sweep

- Time: 5 human-minutes + ~1-3 compute-hours (approximate; 2000 iterations
  across all pool sizes for three diseases, case and random, L1 and LLR).
  Reduce with `--iterations` for a quick check.

This experiment matches [Main Result 5](#main-result-5-pool-size).
The following averages over 2000 iterations and automatically produces the 6 graphs 
for "Figure 5". This can then be directly saved in a `pdf` format.

```bash
uv run python experiment_nsweep.py [--iterations 2000] [--plot-only]
```

Outputs `results/nsweep_auc.pdf` and `results/nsweep_tpr.pdf` (3 panels each)
plus `results/nsweep_results.csv`. Pass `--plot-only` to re-render from the
saved CSV without recomputing.

#### Experiment 6: 19 Disease Validation

- Time: 5 human-minutes + ~1-2 compute-hours (approximate; 2000 random-pool
  iterations for each of the 19 diseases). Reduce with `--iterations` for a
  quick check.

This experiment corresponds to [Main Result 6](#main-result-6-case-v-random).
The following averages over 2000 iterations and automatically produces the 2
two-panel figures for "Figure 6". This can then be directly saved in a `pdf`
format.

```bash
uv run python experiment_19disease.py [--iterations N] [--plot-only]
```

Outputs `results/fig_19disease.pdf` (AUC) and `results/fig_19disease_tpr.pdf`
(TPR @ 1% FPR), each with panel A (pool size vs metric) and panel B (case vs
random scatter), plus `results/19disease_results.csv`. Pass `--plot-only` to
re-render from the saved CSV.

#### Experiment 7: Stratifying diseased features

- Time: 5 human-minutes + ~30-45 compute-minutes for all three panels (case
  pool, 5000 orderings, ~10-15 compute-minutes each).

This experiment matches [Main Result 7](#main-result-7-stratification).
The following is the general command line needed for all plots in Figure 7:

```bash
uv run python Ordered.py <disease> <pool_idx> [random_sample_size] [output.pdf] [stratify]
```

- For **Figure 7a**, `<disease>` is `D1`, `<pool_idx>` is `1`,
  `[random_sample_size]` is `124`, `[stratify]` is `stratified`.
- For **Figure 7b**, `<disease>` is `D17`, `<pool_idx>` is `1`,
  `[random_sample_size]` is `20`, `[stratify]` is `stratified`.
- For **Figure 7c**, `<disease>` is `D14`, `<pool_idx>` is `1`,
  `[random_sample_size]` is `24`, `[stratify]` is `stratified`.

This averages over 5000 iterations and saves the pdf to `<output>`. All three
invocations are wired up in `run_fig7_fig13_experiments.sh`.

#### Experiment 8: Attempts at replicating real drift

- Time: 5 human-minutes + ~5 compute-minutes (2000 iterations, parallelised
  across CPU cores).

This experiment corresponds to [Main Result 8](#main-result-8-longitudinal-drift)
and produces **Table 3** ("AUC and TPR @ 1% FPR under four drift models on the
longitudinal miRNA dataset"). It compares real biological drift against three
synthetic drift models — independent Gaussian (per-feature variance matched),
leave-one-out conditional mean (linear regression), and nearest-neighbour
(k=1) — to test which property of real drift degrades the attack:

```bash
uv run python fig_nn_drift.py
```

This averages over 2000 iterations and prints the four-model table to stdout,
comparing each model against real biological drift at t=1. The four rows of
Table 3 are:
- Row 1: Real drift (ground truth)
- Row 2: Independent Gaussian (per-feature variance matched)
- Row 3: LOO conditional mean (linear regression, held-out prediction)
- Row 4: NN k=1 (nearest neighbour's real drift, Euclidean)

Each row reports AUC and TPR @ 1% FPR for both the L1 and LLR tests. The table
values are also saved to `results/fig_nn_drift.csv`; the accompanying NN k-sweep
figure (`results/fig_nn_drift.pdf`) and its CSV
(`results/fig_nn_drift_appendix.csv`) correspond to the appendix figure.
Re-render without recomputing via `uv run python fig_nn_drift.py --plot-only`.

Supplementary drift-decomposition analyses can be regenerated with
`uv run python fig_linear_residual.py` and
`uv run python fig_correlation_preserving.py` (each also supports `--plot-only`).

## Limitations

<!-- Describe which steps, experiments, results, graphs, tables, etc. are _not
reproducible_ with the provided artifact. Explain why this is not
included/possible and argue why the artifact should _still_ be evaluated for the
respective badges. -->

All figures and tables in the paper are reproducible with the provided code and
bundled datasets. Two caveats, neither of which affects the reproduced
conclusions:

- Figure 1 plots the closed-form Theorem 1 bound; the regenerated PDFs match
  the paper's curves and values but their cosmetic styling (colours, fonts) is
  not pixel-identical to the originals.
- The random-pool experiments (e.g. Experiments 2-7) average over Monte-Carlo
  samples. Each script uses a fixed seed (default 42, overridable with
  `--seed`), so a given invocation is reproducible; absolute values may differ
  by a small Monte-Carlo margin from the paper if the seed or iteration count
  is changed.

## Notes on Reusability

<!-- First, this section might not apply to your artifacts. Describe how your
artifact can be used beyond your research paper, e.g., as a general framework.
The overall goal of artifact evaluation is not only to reproduce and verify your
research but also to help other researchers to re-use and extend your artifacts.
Discuss how your artifacts can be adapted to other settings, e.g., more input
dimensions, other datasets, and other behavior, through replacing individual
modules and functionality or running more iterations of a specific module. -->

The codebase is organised so the membership-inference attacks can be applied to
new settings:

- **Other datasets.** Dataset loading is centralised in `utils_datasets.py`
  (`load_dataset(...)` dispatches to per-dataset loaders). A new dataset only
  needs a loader that returns the population and pool feature matrices; the
  attack and plotting code is dataset-agnostic.
- **Other diseases / pools.** The 19 disease case pools are defined as feature
  sets (`D1`...`D19`) in `utils_datasets.py`; adding a condition is a matter of
  listing its associated features.
- **The attacks themselves.** `utils.py` implements both test statistics — the
  L1-distance difference (`L1_ttest`) and the likelihood-ratio test (`LLR`) —
  and provides `auc_scores`, `tpr_at_fpr`, and Gaussian-noise helpers, with the
  vectorised inner loops in `fast_paths.py`. All are reusable independently of
  the experiment drivers.
- **Reproducibility / replotting.** `experiment_io.py` provides `--seed`,
  `--replot`/`--plot-only`, and saves each figure's data as `<name>.npz` plus a
  `<name>.meta.json` recording the git commit and command line, so any figure
  can be regenerated or re-styled from saved data without rerunning the
  (expensive) computation.
- **Scaling effort.** The Monte-Carlo iteration counts are exposed (script
  constants or `--iterations` / the `NUM_ORDERS`, `NUM_POOLS` environment
  variables) to trade runtime for variance.
