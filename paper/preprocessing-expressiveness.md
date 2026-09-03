# The Value of a Preprocessing Step Is the Cost of Reconstructing It: Evidence from a Smartphone Addiction Prediction Task

[Author Name]  
[Student ID]  
[Affiliation]

---

## Abstract

Predicting smartphone addiction from behavioral and usage data is, on this dataset, less a modeling problem than a missing-data problem: 61.06% of rows are missing at least one feature, while the label is almost fully determined once the data are complete. This creates a concrete, practically motivated version of an old question — which missing-value treatment is best — that turns out to be underdetermined without specifying which model will consume the result. We argue, and test across four model families and three independent preprocessing decisions (missing-value imputation, a derived feature, and exact-value target encoding), that a preprocessing step is worth exactly as much as it is hard for the downstream model to reconstruct on its own. A direct consequence is that the same transform can carry opposite sign depending on the model's expressiveness rather than the data. We further show that two preprocessing steps can destroy each other when one's output leaves the domain the other requires, and, independently, that a model configuration selected cheaply on a subsample transfers to the full dataset with zero selection regret at every scale tested, though the rank inversions that do occur concentrate exactly where that destructive interaction is present. All comparisons use a single frozen cross-validation split and paired, not aggregate, statistics.

---

## 1. Introduction

Smartphone-addiction prediction, as posed here, is a binary classification task built on behavioral and usage variables — screen time, app opens, notifications, sleep, and similar measures. The dataset comes from a Kaggle-hosted competition run as part of coursework, scored by the area under the ROC curve, and it has two properties that shape everything that follows. First, as §2 documents in detail, the label is close to fully determined once a row's features are observed. Second, 61.06% of rows are missing at least one of twelve raw features. Practically, then, the central difficulty is not fitting a sufficiently powerful model; it is deciding what to do with the majority of rows where some of that determining information is absent.

The natural question is which missing-value treatment is best. Asked this way, the question is underdetermined: it has no answer independent of what will consume the imputed values. A treatment that helps one model can hurt another on the identical rows, folds, and features, because the treatment's benefit depends on what the model could already do without it. This paper is an attempt to make that dependency explicit and general, rather than treating each instance — an imputation choice, a derived feature, an encoding — as its own unrelated finding.

We propose, and test across four model families spanning gradient-boosted trees, a random forest, and a regularized linear model, and three independent preprocessing decisions, a single explanatory account:

> A preprocessing step is worth exactly as much as it is hard for the downstream model to reconstruct on its own.

Two consequences follow directly. First, because "how hard to reconstruct" is a property of a model's expressiveness rather than of the data, the same transform can carry opposite sign across model families — helping one and hurting another on identical inputs. Second, when a pipeline chains two preprocessing steps, the output of one can leave the domain the other requires, so the two steps destroy each other's contribution rather than compounding it; to our knowledge this second consequence has not previously been reported.

This paper makes three contributions.

1. We state and test a single account of when a preprocessing step is worth applying — its value equals the difficulty the downstream model would have reconstructing it unaided — across three independent instances (missing-value imputation, a derived feature, and exact-value target encoding) and four model families, and show that the same transform's benefit changes sign, not just magnitude, depending on which model receives it.

2. We identify and trace a failure mode that, as far as we are aware, has not previously been reported: two preprocessing steps in the same pipeline can destroy each other's value when the output of one leaves the domain the other requires. We trace this concretely to an interaction between regression-based imputation and exact-value encoding on this dataset's 0.01-valued lattice, and show the damage is invisible at moderate sample sizes and only visible at full scale.

3. Independently of the above, we test whether a model configuration chosen cheaply — on a 200,000-row stratified subsample rather than the full 691,369-row training set — is the configuration one would have chosen on the full data. Using a five-level sample-size ladder, we find zero selection regret at every level, but the rank inversions that do occur are not random: they concentrate specifically among the configurations subject to the destructive interaction identified in the previous contribution, and are absent elsewhere in the comparison grid.

The remainder of the paper is organized as follows. §2 describes the dataset and the task in more detail, including the generator's structural properties that later sections depend on. §3 sets out the experimental protocol — the frozen folds, the within-fold fitting discipline, and the statistical standard used to call a difference real — against which every subsequent result is measured. §4 reports the full grid of missing-value treatments crossed with model families; §5–§7 examine the three preprocessing instances behind the paper's central thesis in turn: imputation, a derived feature, and exact-value encoding. §8 shows how two of these preprocessing steps interact destructively. §9 asks whether conclusions reached on a subsample of the data transfer to the full dataset. §10 discusses the results together, and §11 concludes.

---

## 2. Data and Task

This section describes the dataset, the prediction task, and two structural properties of the data-generating process — a hard constraint and a value lattice — that later sections depend on.

### 2.1 Scale and Features

The training set contains 691,369 rows and the test set 296,302 rows, each row describing one user with twelve raw features: nine numeric (`age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, and `weekend_screen_time`) and three categorical (`gender`, `stress_level`, and `academic_work_impact`). The target, `addicted_label`, is binary; 70.94% of rows are labeled positive (addicted). The task is scored by the area under the ROC curve (AUC; Hanley & McNeil, 1982), and every model comparison in this paper reports AUC on held-out folds (§3).

**Table 1.** Raw features used as model inputs.

| Type | Features |
|---|---|
| Numeric (9) | `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time` |
| Categorical (3) | `gender`, `stress_level`, `academic_work_impact` |

A substantial fraction of rows are incomplete: 61.06% are missing at least one of the twelve features (`R/02_eda.R`).

### 2.2 Why Missingness Is the Central Difficulty

Two properties of the data, taken together, determine the shape of the entire experiment. First, the label is close to fully determined once the defining features are observed: binning `daily_screen_time_hours` into 1-hour-wide bins, the addiction rate rises monotonically from 24.2% to 99.96%. This is not an artifact of the binning procedure — repeating the identical binning after randomly shuffling the label collapses the range from 0.771 to 0.024 (`R/15_fig2_audit.R`), confirming that the monotone relationship reflects a real association between the feature and the label rather than a property of the binning itself.

Second, the missingness is uninformative. Across the twelve columns, the largest difference in addiction rate between the rows where a column is missing and the rows where it is observed is 0.0042 — small enough to be consistent with sampling noise rather than a systematic relationship between missingness and the outcome, i.e., missing completely at random (MCAR; Little & Rubin, *Statistical Analysis with Missing Data*). We refer to missingness under this pattern as MCAR throughout.

Together, these two facts reframe the prediction task. If the label were only weakly related to the features, a more powerful model would matter more than complete data. If missingness carried information about the label — for example, if users who did not report their screen time were disproportionately likely to be addicted — the missingness pattern itself would be a usable feature. Neither holds here: the signal is strong when the data are complete, and the incompleteness is uninformative about which rows are missing. What remains is a narrower, more concrete question: given that 61.06% of rows have some of that signal removed, how much of it can be recovered, and by what means?

### 2.3 The Generator's Structure: A Four-Term Constraint and a 0.01 Lattice

The training and test data were synthetically generated for the competition, inspired by a real smartphone-addiction survey instrument, and the generator leaves two structural fingerprints in the data that matter for later sections.

First, a hard constraint holds among four of the time-use features:

```
daily_screen_time_hours ≥ social_media_hours + gaming_hours + work_study_hours
```

Among the 421,427 rows where all four columns are observed, the inequality holds in 100.0000% of rows, and the minimum residual (the left side minus the right side) is exactly 0.000 — the constraint is occasionally tight. An earlier version of this check used only three terms, omitting `work_study_hours`; that version also held in 100% of rows, but its minimum residual was 0.100, not 0.000. Because the three-term inequality is implied by the four-term one — dropping a non-negative quantity from the right-hand side can only relax the inequality — the 0.100 minimum shows that the three-term version was never actually tight. It is a consequence of the true, four-term constraint rather than a boundary in its own right, and it is the four-term relation that we treat as the generator's real structural rule throughout this paper.

Second, feature values are generated on a 0.01 grid, and the second decimal digit of a value correlates with the row's positive rate. This lattice structure is a property of the generator, not of the underlying construct being measured; §8 shows that it interacts consequentially with how missing values are imputed.

### 2.4 Post-Competition Status

The competition this dataset was drawn from ended on August 31. Most of the numbers reported in this paper — including the corrected four-term constraint above and the exact-value encoding described in later sections — were computed after that date, and none of them have leaderboard verification: there is no live scoring system left to check them against beyond the frozen validation folds described in §3. This paper reports local, cross-validated results throughout; no leaderboard placement or ranking is claimed anywhere, and any value obtained by extrapolating beyond what was directly measured is flagged as an extrapolation at the point it is used.

---

## 3. Experimental Protocol

This section sets out the protocol shared by every comparison in this paper: how the data are split, how label-dependent transforms are fit, how a difference is judged real rather than noise, and what can and cannot be reproduced exactly.

### 3.1 Frozen Folds

All comparisons in this paper share a single 5-fold cross-validation split, generated once, stratified by label, with random seed 20260821, and stored rather than regenerated. `R/04_folds.R` — the script that produces it — refuses to overwrite an existing fold file unless explicitly told to; this is a deliberate guard, not an oversight. Regenerating the split would make every comparison collected before the regeneration incomparable with everything collected after, and the differences under study are frequently on the order of 0.001 AUC — smaller than the variation a fresh 5-fold split would itself introduce. Freezing the split trades the small statistical benefit of re-randomizing for the ability to compare results gathered at different times against each other at all.

### 3.2 Within-Fold Fitting

A second discipline concerns information leakage (Kaufman, Rosset & Perlich, 2012): any step that looks at the label while transforming the features — the missing-value imputers and the target encoder (Micci-Barreca, 2001), which replaces a value with a statistic of the label computed from other rows sharing that value — must be fit only on a fold's training portion and merely applied to that fold's validation portion. Fitting such a step on the validation portion, even incidentally, would inflate its validation-fold score in a way that has no counterpart in the true held-out test set.

This split is enforced by a single function, `prepare_fold()`, defined once in `R/03_features.R` as a `fit_*` / `apply_*` pair for every label-dependent transform. It replaces an earlier arrangement in which the same fold-preparation logic was duplicated across six scripts; when target encoding was added as a new feature, only the script built on the shared framework picked it up automatically, while the other three drifted onto a stale feature set without any run's output making that drift visible. `prepare_fold()` exists specifically to make that class of error structurally impossible rather than something to remember to avoid.

### 3.3 Paired Statistics and the Placebo Column

With only 5 folds, comparing two configurations by a p-value computed across those 5 paired AUC differences is not trustworthy — an ordinary test at n = 5 has little power to distinguish a real effect from noise, and a small p-value from so few points is easy to obtain by chance. We instead report three quantities for every comparison: the paired mean difference across folds; Cohen's d, the mean difference divided by the standard deviation of the per-fold differences (Cohen, 1988); and the number of folds, out of 5, on which the difference has the same sign. Five-out-of-five sign agreement together with Cohen's d greater than 2 is the standard actually used in this paper to call an effect real; a p-value alone is not treated as sufficient.

To calibrate what a null effect looks like under this standard, we added a placebo column — random noise, with a missingness pattern matched to a real column — as a negative control. Across the frozen folds, the placebo measures +0.00003 AUC, Cohen's d 0.11, with only 2 of 5 folds agreeing in sign: indistinguishable from adding nothing. Any feature or transform claimed to have a real effect in this paper is expected to clear this floor by a wide margin, not merely to be numerically positive.

### 3.4 The Resolution Floor Is a Property of the Pair

A further question is how large a paired difference needs to be before it is distinguishable from measurement noise at all — a resolution floor beneath the placebo check, which only characterizes what a null effect looks like on the folds already collected. `R/23_resolution.R` estimates this floor for a specific pair of configurations by bootstrap (Efron & Tibshirani, 1993): drawing 400 bootstrap resamples at the test set's size (296,302 rows), computing each configuration's AUC on every resample, and taking

```
SD(difference) = SD(single configuration's AUC) × √(2 × (1 − ρ))
```

where ρ is the correlation, across those resamples, between the two configurations' AUC estimates — not between their row-level predictions.

The critical property of this quantity is that it depends on which two configurations are being compared, not on the dataset alone. Two configurations that are near-twins — the same underlying model family, differing only in which of the missing-value treatments defined in §4 is applied upstream (`L1_xgboost` vs. `L2_xgboost`) — have highly correlated AUC estimates (ρ = 0.984) and a resolution floor of 0.000098. A pair drawn from different model families entirely (`L1_xgboost` vs. `L3_ranger`) is far less correlated (ρ = 0.745) and has a resolution floor of 0.000564 — on the same data, nearly six times larger.

**Table 2.** Resolution floor depends on which pair is compared, not on the dataset.

| Pair | Relationship | ρ | Resolution floor |
|---|---|---|---|
| `L1_xgboost` vs. `L2_xgboost` | Same model family, near-twin | 0.984 | 0.000098 |
| `L1_xgboost` vs. `L3_ranger` | Cross model family | 0.745 | 0.000564 |

There is accordingly no single answer to how big a difference needs to be to count as real: it depends on which pair is being asked about. Every comparison in this paper that claims a real difference is checked against the floor for that specific pair rather than against one fixed threshold.

### 3.5 Reproducibility Statement

Of the four model families used in this paper, three — xgboost (Chen & Guestrin, 2016), ranger (Wright & Ziegler, 2017), which implements the random forest algorithm (Breiman, 2001), and glmnet (Friedman, Hastie & Tibshirani, 2010) — are bit-reproducible: rerunning identical code on the identical frozen folds returns identical output to the last digit. lightgbm (Ke et al., 2017) is not. Its configuration sets `num_threads = detectCores()` without also setting `deterministic = TRUE`, and its multithreaded histogram construction does not guarantee a fixed floating-point summation order across runs, so results can vary in the last few digits depending on thread scheduling.

We measured this directly: three runs of the same lightgbm configuration returned 0.96039, 0.96043, and 0.96039 (the middle run competed for CPU with two other concurrent R processes), a maximum deviation of 4×10⁻⁵. This is below every resolution floor actually used in this paper — the smallest of which, the near-twin pair above, is 0.000098. No conclusion in this paper rests on a difference smaller than that. Where an individual lightgbm number is reported without averaging over repeated runs, it should be read as accurate to about ±4×10⁻⁵, not as an exact value in the way the other three families' numbers are.
