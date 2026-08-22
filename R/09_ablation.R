# =============================================================================
# 特征消融：回答两个问题
#   Q1 派生特征的冗余是否伤害 L3？（对比 L1 作为控制组）
#   Q2 已知的噪声特征该不该筛掉？
#
# 全部沿用共享折叠，与已有的 6 个格子严格可比。
# =============================================================================
suppressMessages({library(data.table); library(pROC)})
source("R/03_features.R")

SEED <- 20260821L

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")

train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

DERIVED <- c("other_screen", "weekend_ratio", "social_share",
             "gaming_share", "free_frac", "screen_social")

# EDA 判定的噪声：三个类别/计数特征 + 全部缺失指示（MCAR，delta <= 0.0042）
NOISE_RAW <- c("stress_level", "academic_work_impact",
               "notifications_per_day", "gender")
NA_IND <- grep("^is_na_", names(feat), value = TRUE)

drop_sets <- list(
  full     = character(0),
  noderiv  = DERIVED,
  nonoise  = c(NOISE_RAW, NA_IND, "n_missing"),
  lean     = c(NOISE_RAW, NA_IND, "n_missing", "free_frac")
)

fit_predict_xgb <- function(X_tr, y_tr, X_va, drop_extra) {
  drop_cols <- c("id", "addicted_label", "is_train", drop_extra)
  use_cols  <- setdiff(names(X_tr), drop_cols)
  to_m <- function(dt) {
    m <- dt[, ..use_cols]
    for (cc in names(m)) if (is.factor(m[[cc]])) set(m, j = cc, value = as.integer(m[[cc]]))
    as.matrix(m)
  }
  p <- list(objective = "binary:logistic", eval_metric = "auc",
            eta = 0.05, max_depth = 6, subsample = 0.8,
            colsample_bytree = 0.8, min_child_weight = 10,
            tree_method = "hist", nthread = parallel::detectCores())
  mdl <- xgboost::xgb.train(p, xgboost::xgb.DMatrix(to_m(X_tr), label = y_tr),
                            nrounds = 600, verbose = 0)
  list(pred = predict(mdl, xgboost::xgb.DMatrix(to_m(X_va))), n_feat = length(use_cols))
}

run <- function(line, set_name) {
  source(sprintf("R/05_impute_%s.R", line), local = FALSE)
  fit_i   <- get(paste0("fit_imputer_",   line))
  apply_i <- get(paste0("apply_imputer_", line))
  drop_extra <- drop_sets[[set_name]]

  aucs <- numeric(0); nf <- NA
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    imp  <- fit_i(X_pool[tr])
    X_tr <- derive_features(apply_i(imp, copy(X_pool[tr])))
    X_va <- derive_features(apply_i(imp, copy(X_pool[va])))
    r <- fit_predict_xgb(X_tr, y_pool[tr], X_va, drop_extra)
    nf <- r$n_feat
    aucs <- c(aucs, as.numeric(auc(roc(y_pool[va], r$pred, quiet = TRUE))))
  }
  list(auc = aucs, n_feat = nf)
}

res <- list()
for (line in c("L1", "L3")) {
  for (sn in names(drop_sets)) {
    key <- paste0(line, "_", sn)
    cat(sprintf("running %-12s ... ", key)); flush.console()
    r <- run(line, sn)
    res[[key]] <- r
    cat(sprintf("%d feats  AUC %.5f +- %.5f\n", r$n_feat, mean(r$auc), sd(r$auc)))
  }
}

cat("\n================= 汇总 =================\n")
cat(sprintf("%-14s %6s %10s %10s\n", "变体", "特征数", "AUC", "sd"))
for (k in names(res))
  cat(sprintf("%-14s %6d %10.5f %10.5f\n", k, res[[k]]$n_feat,
              mean(res[[k]]$auc), sd(res[[k]]$auc)))

cat("\n============ 配对检验（vs full）============\n")
for (line in c("L1", "L3")) {
  base <- res[[paste0(line, "_full")]]$auc
  for (sn in c("noderiv", "nonoise", "lean")) {
    d <- res[[paste0(line, "_", sn)]]$auc - base
    tt <- t.test(d)
    cat(sprintf("%-14s - %-9s  %+.5f  same_sign %-5s  t=%7.2f  p=%.4f\n",
                paste0(line, "_", sn), paste0(line, "_full"), mean(d),
                all(sign(d) == sign(d[1])), tt$statistic, tt$p.value))
  }
}
saveRDS(res, "output/ablation_res.rds")
