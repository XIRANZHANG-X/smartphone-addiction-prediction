# =============================================================================
# 消融二：拆开「17 个噪声特征」，定位 0.007 的损失究竟来自谁
# 只跑 L1（两条线表现几乎一致，L1 作代表足够）
# =============================================================================
suppressMessages({library(data.table); library(pROC)})
source("R/03_features.R")
source("R/05_impute_L1.R")

SEED <- 20260821L
feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

NA_IND <- grep("^is_na_", names(feat), value = TRUE)

drop_sets <- list(
  full          = character(0),
  no_naind      = c(NA_IND, "n_missing"),                       # 只删缺失相关
  no_raw4       = c("stress_level","academic_work_impact",
                    "notifications_per_day","gender"),          # 只删 4 个原始噪声
  no_notif      = "notifications_per_day",                      # 只删通知数
  no_cat3       = c("stress_level","academic_work_impact","gender"),
  no_opens      = "app_opens_per_day",                          # 对照：另一个疑似噪声
  no_age        = "age"
)

fitpred <- function(X_tr, y_tr, X_va, drop_extra) {
  use_cols <- setdiff(names(X_tr), c("id","addicted_label","is_train", drop_extra))
  to_m <- function(dt) {
    m <- dt[, ..use_cols]
    for (cc in names(m)) if (is.factor(m[[cc]])) set(m, j = cc, value = as.integer(m[[cc]]))
    as.matrix(m)
  }
  p <- list(objective="binary:logistic", eval_metric="auc", eta=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, tree_method="hist",
            nthread=parallel::detectCores())
  mdl <- xgboost::xgb.train(p, xgboost::xgb.DMatrix(to_m(X_tr), label=y_tr),
                            nrounds=600, verbose=0)
  list(pred=predict(mdl, xgboost::xgb.DMatrix(to_m(X_va))), n=length(use_cols))
}

res <- list()
for (sn in names(drop_sets)) {
  cat(sprintf("running %-10s ... ", sn)); flush.console()
  a <- numeric(0); nf <- NA
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    imp  <- fit_imputer_L1(X_pool[tr])
    X_tr <- derive_features(apply_imputer_L1(imp, copy(X_pool[tr])))
    X_va <- derive_features(apply_imputer_L1(imp, copy(X_pool[va])))
    r <- fitpred(X_tr, y_pool[tr], X_va, drop_sets[[sn]]); nf <- r$n
    a <- c(a, as.numeric(auc(roc(y_pool[va], r$pred, quiet=TRUE))))
  }
  res[[sn]] <- a
  cat(sprintf("%2d feats  AUC %.5f +- %.5f\n", nf, mean(a), sd(a)))
}

cat("\n============ 配对检验（vs full）============\n")
base <- res$full
for (sn in setdiff(names(res), "full")) {
  d <- res[[sn]] - base; tt <- t.test(d)
  cat(sprintf("%-10s  %+.5f  same_sign %-5s  t=%8.2f  p=%.4f\n",
              sn, mean(d), all(sign(d)==sign(d[1])), tt$statistic, tt$p.value))
}
saveRDS(res, "output/ablation2_res.rds")
