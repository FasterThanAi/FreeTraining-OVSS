# FreeTraining-OVSS

Final-year research project, IIITDM Kurnool. Goal is a publishable paper.

**Training-free, annotation-free open-vocabulary semantic segmentation for remote sensing.**
Contribution: recover low-confidence pixels that presence-gated SAM 3 pipelines discard to
"background", using a semantic co-occurrence prior over region proposals.

Do **not** call this "unsupervised" — the vocabulary is given. See @ANALYSIS.md §3.6.

## Read these before proposing anything

- @ANALYSIS.md — problem framing, technical critique, measured PMI findings (§4). The
  empirical foundation. §4 numbers are load-bearing.
- @ROADMAP.md — 12-week plan, phase gates, reading list.
- @WEEK1_RESULTS.md — all measurements to date. Current state of the project.

## Baseline

SegEarth-OV3 (arXiv:2512.08730), our named baseline and closest competitor. Reproduced at
**47.38 mIoU** on LoveDA val vs their reported 47.4. This number is a gate — if any change
moves it, the change is wrong.

Second competitor: ConInfer (arXiv:2603.29271). Novelty claims must differentiate from it.

## Environment — do not "helpfully" upgrade anything

A three-way version deadlock. This is the **only** confirmed working combination; five other
approaches failed:

| | |
|---|---|
| Python | 3.11 (conda env `segov3`) |
| torch | 2.4.1+cu121 — SAM 3 needs ≥2.3, no mmcv wheels exist for 2.5 |
| mmcv | 2.2.0, prebuilt wheel from the torch2.4/cu121 index |
| mmsegmentation | 1.2.2 with `MMCV_MAX` patched to `'2.3.0'` in site-packages |

Never suggest `pip install -U`, mmcv-lite, or building mmcv from source (system nvcc is 13.3
against torch's 12.1 — it cannot work). Full rationale in @WEEK1_RESULTS.md §2.

Hardware: RTX 2000 Ada, 16 GB, power-capped at 70 W. A full LoveDA val pass is ~24 min and
that is the floor.

## Repo conventions

- `SegEarth-OV-3/` ships a **vendored `sam3/`** that shadows the editable `~/sam3` install
  when running from inside it. Edits to `~/sam3` have no effect there.
- Datasets, checkpoints and venvs are gitignored. Never commit them. Checkpoint is symlinked
  from the HF cache.
- The GitHub **profile** README sometimes overwrites the project README on Mac-side commits.
  If `README.md` looks wrong: `git checkout HEAD -- README.md`.
- Long runs: `nohup ... > log 2>&1 &`. The trailing `&` is not optional — without it the
  shell blocks and a serial loop silently runs one job at a time.

## Settled design decisions — do not relitigate

These were decided by measurement, not preference. Rationale in the cited sections.

- **Adjacency = shared boundary length**, not centroid distance. ANALYSIS §4.
- **Signed PMI, not raw counts.** Exclusion carries the largest magnitudes (building–water at
  0.02× chance). Additive-positive scoring cannot represent it. ANALYSIS §4.2.
- **Hierarchical M is required**: `M_eff = λ·M_global + (1−λ)·M_image`. Six of fifteen class
  pairs flip sign between urban and rural. ANALYSIS §4.4.
- **M is directed, not symmetric.** water→agricultural is 19.3M with no reverse in any top-8.
  WEEK1_RESULTS §8.1(b).
- **Presence gating is inherited from SegEarth-OV3 but is not free** — it vetoes whole tiles.
  WEEK1_RESULTS §9.2. (ANALYSIS §3.5 previously claimed otherwise; corrected.)

## Where the project stands

Weeks 1–2 complete. Baseline reproduced; premise confirmed and quantified:

- **29.68%** of real-class pixels discarded to background at τ=0.5 (323M pixels).
- τ-sweep: recovering ⅔ of that residual by lowering τ to 0.1 costs **5.54 mIoU**. Threshold
  relaxation buys 1 correct pixel per 1.73 wrong ones. This is the paper's motivation.
- Discard outnumbers real-class confusion **3:1** — the baseline's dominant error is silence,
  not error.

Next: instrument `measure_discard_rate.py` for per-class `S_pres` + an `.npz` cache, re-run
once at τ=0.5. Then Week 3, `M_global` construction.

## How to work on this

- **Measure before deciding.** Every open question in ROADMAP has a cheap experiment attached.
  Run it rather than arguing from intuition.
- **Validation gates are hard stops.** Instrumentation changes must still reproduce 47.37 mIoU
  and 29.68% discard. If they don't, the patch changed behaviour.
- **Cache SAM 3 outputs.** Downstream stages get re-run hundreds of times during ablations.
  Never pay encoder cost for arithmetic on a saved confidence map.
- **Seed everything, log every config.** Ablation tables assembled from unlabelled runs are how
  projects die in week 11.
- Keep a `LOGBOOK.md` entry per working day: what was tried, what broke, what the number was.
- Push back when the plan is wrong. A correction now is cheaper than a rewritten results
  section in week 11.