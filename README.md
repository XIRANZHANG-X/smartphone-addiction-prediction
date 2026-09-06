# Smartphone Addiction Prediction

Predicting whether a user is addicted to their smartphone, in R. Metric: **ROC-AUC**.
Kaggle in-class competition, live leaderboard.

---

## 1. The Task

Given 12 self-reported usage and lifestyle variables, predict the binary label
`addicted_label`.

| | |
|---|---|
| Training set | 691,369 rows (`id` 0 – 691368), labelled |
| Test set | 296,302 rows (`id` 691369 – 987670), unlabelled |
| Metric | ROC-AUC |
| Positive rate | 0.7094 (an all-constant submission at this value scores AUC = 0.5) |

### Features

Nine numeric, three categorical. **Every column has missing values.**

| Feature | Meaning | Missing rate |
|---|---|---|
| `age` | age in years | 4.18% |
| `daily_screen_time_hours` | daily screen time (hours) | 13.86% |
| `social_media_hours` | social media time | 19.38% |
| `gaming_hours` | gaming time | 18.34% |
| `work_study_hours` | work / study time | 7.45% |
| `sleep_hours` | sleep time | 6.43% |
| `notifications_per_day` | notifications received per day | 9.78% |
| `app_opens_per_day` | app opens per day | 11.67% |
| `weekend_screen_time` | weekend screen time | 16.21% |
| `gender` | Male / Female / Other | 4.20% |
| `stress_level` | High / Medium / Low | 7.98% |
| `academic_work_impact` | Yes / No | 6.40% |

Only **269,185 rows (38.94%)** are complete; the remaining **61%** are missing at
least one feature. The label is close to deterministic once a row is complete, so
in practice this is a missing-data problem rather than a modelling problem —
leaderboard separation happens in the third decimal place.

The data also carries a hard generator constraint, which several of the
imputation lines exploit:

```
daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours
```

It holds in 100% of the 421,427 rows where all four columns are observed.

**The data is not in this repository** (~70 MB, and competition data should not be
redistributed). Download it from the Kaggle competition page — see
[§3, step 2](#step-2-get-the-data).

---

## 2. Code Directory

```
├── R/                   all code
│   ├── 00_setup.R       package installation + environment check
│   ├── 01_load.R        read the raw CSVs, pin column types
│   ├── 02_eda.R         exploratory analysis
│   ├── 03_features.R    feature engineering (single source of truth)
│   ├── 04_folds.R       ★ the fold contract — frozen, must not be regenerated
│   ├── 05_impute_L*.R   the four imputation lines (see below)
│   ├── 06_framework.R   shared CV loop, tier selection, leakage guards, I/O
│   ├── lib_models.R     model factories: make_xgb/make_lgb/make_ranger/make_glmnet
│   ├── 06_model_*.R     ★ one thin config file per model × imputation line
│   ├── 07_ensemble.R    ensembling over all out-of-fold predictions
│   ├── 08_submit.R      write the Kaggle submission file
│   ├── 09_*.R … 41_*.R  focused experiments and EDA (ablation, calibration,
│   │                    repeated CV, resolution floor, adversarial controls,
│   │                    distributional EDA, joint missingness structure, …)
│   │                    Every script carries its own usage comment at the top.
│   ├── run_grid.R       the 4 × 4 comparison grid (200k subsample)
│   ├── run_grid_full.R  the same grid on the full training set
│   ├── run_tierb.R      full-data retraining of the selected models
│   ├── run_pipeline.R   one-shot driver for every experiment
│   └── deck_figures.R   English figures for the presentation deck
├── data/raw/            raw competition data (not in git)
├── output/              fold contract and experiment artifacts (mostly not in git)
├── paper/               full paper (Markdown + PDF) and its figures
├── reports/figures/     13 figures, 300 dpi
├── slides/              presentation deck and per-slide script
├── submissions/         submission log with leaderboard scores
└── docs/                detailed write-ups (Chinese)
```

### The four imputation lines

Each line is a different answer to "what do we do about the 61%":

| Line | Approach | Compatible models |
|---|---|---|
| **L1** | No imputation — feed `NA` straight to the model | xgboost, lightgbm only (they learn a default split direction per node) |
| **L2** | Median / mode fill | all |
| **L3** | Constraint-aware imputation using the generator identity above | all |
| **L4** | Predictive mean matching (chained random forests, `missRanger`) | all |

### Three-layer model code

Adding a model touches exactly one file.

| Layer | File | Responsibility |
|---|---|---|
| Framework | `R/06_framework.R` | data loading, tier selection, the leakage-safe CV loop, saving results. Shared — do not edit |
| Models | `R/lib_models.R` | factories `make_xgb()` / `make_lgb()` / `make_ranger()` / `make_glmnet()`; early stopping written once |
| Config | `R/06_model_*.R` | ~27 lines — three config variables and one `fit_predict <- make_xgb()` |

**Fold discipline (why the framework exists).** Both imputation and per-value
target encoding are fitted **inside each fold**. Fitting either on the full
training set and then applying it — even using the same folds — leaks validation
labels into the training rows and inflates CV. The framework already handles
this; `R/06_model_TEMPLATE.R` shows the correct loop structure.

---

## 3. Reproducing the Results

### Requirements

R **≥ 4.2** (developed on 4.6.1). Versions below 4.2 have no native UTF-8 support
on Windows.

Packages installed on top of base R — `R/00_setup.R` installs all of them and
verifies the library path is writable:

| Package | Used for |
|---|---|
| `data.table` | reading and manipulating 690k rows (`read.csv` is unusable at this size) |
| `xgboost` | gradient boosting, native `NaN` handling |
| `lightgbm` | gradient boosting, native `NaN` handling, usually faster |
| `ranger` | random forest |
| `glmnet` | regularised logistic regression, the interpretable baseline |
| `pROC` | AUC computation |
| `missRanger` | chained random-forest PMM imputation for line L4 |
| `renv` | dependency version locking (optional) |
| `ggplot2` | figures (optional) |
| `knitr`, `rmarkdown` | report rendering (optional) |

Exact versions are pinned in `renv.lock`.

### Step 1: Set up the environment

```r
# open smartphone-addiction-prediction.Rproj in RStudio, then:
source("R/00_setup.R")   # run once
```

### Step 2: Get the data

Download from the Kaggle competition page and place these three files in
`data/raw/`:

```
train.csv   test.csv   sample_submission.csv
```

### Step 3: Build the shared artifacts

```r
source("R/01_load.R")      # read and type-pin the CSVs
source("R/03_features.R")  # feature engineering
source("R/04_folds.R")     # the fold contract — takes seconds
```

`output/folds.rds` is committed to git and is the reason every score in this
project is comparable. **Regenerating it invalidates every previous result.**

### Step 4: Get the experiment artifacts

A full re-run of every experiment takes **7–8 hours**. You almost certainly want
the pre-computed artifacts instead.

`output/` is gitignored apart from three frozen files (`folds.rds`,
`subsample_200k.rds`, `results.md`), so a fresh clone starts empty.

> **[Releases](https://github.com/XIRANZHANG-X/smartphone-addiction-prediction/releases)
> → download the latest `artifacts-YYYY-MM-DD.zip`, unzip, and copy `output/`
> over the project root.**

The archive contains the full 4 × 4 grid (14 cells), the Tier A grid, all four
rungs of the sample-size ladder, repeated CV, and every focused experiment. It
does **not** contain the competition data itself (`raw_train.rds`,
`raw_test.rds`, `features_raw.rds`) — those are produced by step 3 in seconds.

Why not commit `output/` directly: it is 819 MB, and git keeps every version
forever, so each grid re-run would add another 75–100 MB that can never be
removed. Release attachments stay out of git history.

### Step 5: Re-run experiments (optional)

One command runs everything, in dependency order, cheapest-and-most-informative
first:

```bash
Rscript R/run_pipeline.R
```

Ten steps: comparison grid → feature ablation → xgboost hyperparameter search →
glmnet alpha sweep → probability calibration → repeated CV (n=15) → full-data
retraining → ensembling → submission file → results table. Every step skips
itself if its output already exists, so an interrupted run resumes by simply
re-running it. `Rscript R/run_pipeline.R 3` starts at step 3; `FORCE=1` re-runs
regardless of existing outputs.

Individual variants are driven by environment variables — no need to edit the 14
model config files:

```bash
TIER=B         Rscript R/06_model_L1_xgboost.R   # full-data retrain
REPEAT_ID=1    Rscript R/06_model_L1_xgboost.R   # repeated CV
USE_DERIVED=0  Rscript R/06_model_L1_xgboost.R   # disable derived features
USE_TE=0       Rscript R/06_model_L1_xgboost.R   # disable per-value target encoding
POOL_FILE=output/pools/pool_100k.rds \
               Rscript R/06_model_L1_xgboost.R   # explicit row pool (sample-size ladder)
IMPUTE_CACHE=1 TIER=B \
               Rscript R/06_model_L4_xgboost.R   # reuse cached imputations (L4 only)
```

`IMPUTE_CACHE=1` is essentially mandatory for L4: all four algorithms receive
bit-identical PMM imputations on a given fold, so caching cuts the four full-data
L4 cells from 15.6 hours to 4.2. The equivalence is verified in
`R/26_cache_check.R`.

### Expected outputs

Each model line writes two files with fixed shapes — useful for checking that a
re-run was correct:

| File | Length |
|---|---|
| `output/oof/oof_<name>.rds` | 691,369 (full-data out-of-fold predictions) |
| `output/test/test_<name>.rds` | 296,302 (test-set predictions) |
| `output/oof/oof_grid_<name>.rds` | 200,000 (comparison grid, 200k subsample) |

`R/07_ensemble.R` picks up every `oof_*.rds` in the directory automatically, so
adding a model requires no changes to existing code. `R/11_report.R` regenerates
`output/results.md`, the machine-written table of every experimental number.

### Reproducibility caveat

`lightgbm` is not bit-for-bit reproducible across runs on this setup; the
observed deviation is ~4×10⁻⁵ AUC, which is below the resolution floor of every
comparison reported in the paper. `xgboost`, `ranger` and `glmnet` reproduce
exactly given the frozen folds.
