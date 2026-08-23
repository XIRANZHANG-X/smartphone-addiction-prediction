# =============================================================================
# 06_framework.R —— 建模框架（所有模型线共用，不要修改）
#
# 用法：在你自己的 R/06_model_<名字>.R 里设好配置和 fit_predict，然后
#         source("R/06_framework.R")
#
# 必须在 source 之前定义好：
#   MODEL_NAME   字符串，输出文件名的一部分
#   TIER         "A"（20 万子样本，对比实验）或 "B"（全量，集成候选）
#   IMPUTE_LINE  "L1" / "L2" / "L3" / "L4"
#   fit_predict  函数 (X_tr, y_tr, X_va) -> numeric，用 lib_models.R 里的工厂造
#
# 可选：
#   SEED         默认 20260821
#   USE_DERIVED  默认 TRUE。设 FALSE 则不计算 6 个派生特征
#                （消融已证明它们无效，见项目说明 6.6）
#   REPEAT_ID    默认 0。非 0 时用不同的折划分做重复 CV，
#                供统计稳健性检验使用（见 09_repeated_cv.R）
#   QUIET        默认 FALSE
#
# -----------------------------------------------------------------------------
# 为什么这个文件不能改
# -----------------------------------------------------------------------------
# 它保证了所有人的分数可比。两条纪律写死在这里：
#
#   1. 插补器只在训练折上拟合，绝不碰验证折（防泄漏）
#   2. 派生特征在插补之后才算（否则比值型特征全是 NA）
#
# 早停所需的内部验证集由 fit_predict 自己从训练折里切（见 lib_models.R），
# 外层验证折从头到尾不参与任何模型选择。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})

# ---- 配置检查 ---------------------------------------------------------------
for (v in c("MODEL_NAME", "TIER", "IMPUTE_LINE", "fit_predict")) {
  if (!exists(v)) stop("source 框架之前必须先定义 ", v)
}
if (!exists("SEED"))        SEED        <- 20260821L
if (!exists("USE_DERIVED")) USE_DERIVED <- TRUE
if (!exists("REPEAT_ID"))   REPEAT_ID   <- 0L
if (!exists("QUIET"))       QUIET       <- FALSE

# 环境变量覆盖，避免为了跑 Tier B 或重复 CV 去改 14 个模型文件：
#   TIER=B        Rscript R/06_model_L1_xgboost.R
#   REPEAT_ID=1   Rscript R/06_model_L1_xgboost.R
#   USE_DERIVED=0 Rscript R/06_model_L1_xgboost.R
if (nzchar(Sys.getenv("TIER")))      TIER        <- Sys.getenv("TIER")
if (nzchar(Sys.getenv("REPEAT_ID"))) REPEAT_ID   <- as.integer(Sys.getenv("REPEAT_ID"))
if (nzchar(Sys.getenv("USE_DERIVED")))
  USE_DERIVED <- !(Sys.getenv("USE_DERIVED") %in% c("0", "FALSE", "false"))

say <- function(...) if (!QUIET) cat(...)

# ---- 依赖 -------------------------------------------------------------------
source("R/03_features.R")
source(sprintf("R/05_impute_%s.R", IMPUTE_LINE))
.fit_imputer   <- get(paste0("fit_imputer_",   IMPUTE_LINE))
.apply_imputer <- get(paste0("apply_imputer_", IMPUTE_LINE))

# ---- 共享产物 ---------------------------------------------------------------
.feat  <- readRDS("output/features_raw.rds")
.folds <- readRDS("output/folds.rds")

.train_all <- .feat[is_train == 1L]
.test_all  <- .feat[is_train == 0L]
.y_all     <- .train_all$addicted_label

stopifnot("features_raw 训练行数与 folds 长度不一致" =
            nrow(.train_all) == length(.folds))

# ---- 选行 -------------------------------------------------------------------
if (TIER == "A") {
  .row_idx <- readRDS("output/subsample_200k.rds")
} else if (TIER == "B") {
  .row_idx <- seq_len(nrow(.train_all))
} else {
  stop('TIER 只能是 "A" 或 "B"')
}

X_pool <- .train_all[.row_idx]
y_pool <- .y_all[.row_idx]
f_pool <- .folds[.row_idx]

# ---- 重复 CV（统计稳健性用，默认关闭）---------------------------------------
# REPEAT_ID = 0 时使用冻结的 folds.rds，这是所有正式结果的依据。
# REPEAT_ID > 0 时重新分层划分，用于把配对样本量从 n=5 提到 n=15，
# 解决 n=5 配对 t 检验不稳健的问题（审查意见 1.4）。
# 这些结果是**补充分析**，不进网格对比表。
if (REPEAT_ID > 0L) {
  set.seed(SEED + 7919L * REPEAT_ID)
  f_pool <- integer(length(y_pool))
  for (cls in c(0L, 1L)) {
    ii <- which(y_pool == cls)
    f_pool[sample(ii)] <- rep_len(1:5, length(ii))
  }
}

.derive <- if (USE_DERIVED) derive_features else function(dt) dt

say(sprintf("模型 %s | Tier %s | %s 行 | 插补 %s | 派生特征 %s%s\n",
            MODEL_NAME, TIER, format(nrow(X_pool), big.mark = ","),
            IMPUTE_LINE, if (USE_DERIVED) "开" else "关",
            if (REPEAT_ID > 0L) sprintf(" | 重复 #%d", REPEAT_ID) else ""))

# ---- 交叉验证主循环 ---------------------------------------------------------
oof       <- rep(NA_real_, nrow(X_pool))
fold_auc  <- numeric(0)
best_iter <- integer(0)
t0 <- Sys.time()

for (k in sort(unique(f_pool))) {
  say(sprintf("  第 %d 折 ... ", k))
  set.seed(SEED + k)

  tr <- which(f_pool != k)
  va <- which(f_pool == k)

  # 1. 插补器只看训练折
  imp <- .fit_imputer(X_pool[tr])

  # 2. 变换两边
  X_tr <- .apply_imputer(imp, copy(X_pool[tr]))
  X_va <- .apply_imputer(imp, copy(X_pool[va]))

  # 3. 插补之后才算派生特征
  X_tr <- .derive(X_tr)
  X_va <- .derive(X_va)

  # 4. 训练 + 预测（早停在 fit_predict 内部完成）
  pred <- fit_predict(X_tr, y_pool[tr], X_va)
  bi <- attr(pred, "best_iteration")
  if (!is.null(bi)) best_iter <- c(best_iter, bi)

  oof[va] <- as.numeric(pred)
  a <- as.numeric(pROC::auc(pROC::roc(y_pool[va], as.numeric(pred), quiet = TRUE)))
  fold_auc <- c(fold_auc, a)
  say(sprintf("AUC %.5f%s\n", a,
              if (!is.null(bi)) sprintf("  (%d 轮)", bi) else ""))
}

cv_mean <- mean(fold_auc)
cv_sd   <- sd(fold_auc)
oof_auc <- as.numeric(pROC::auc(pROC::roc(y_pool, oof, quiet = TRUE)))
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

say(sprintf("\n跨折 AUC  %.5f ± %.5f\n", cv_mean, cv_sd))
say(sprintf("整体 OOF  %.5f\n", oof_auc))
if (length(best_iter)) {
  say(sprintf("早停轮数  %s（均值 %.0f）\n",
              paste(best_iter, collapse = " / "), mean(best_iter)))
}
say(sprintf("耗时      %.1f 分钟\n", elapsed))

# ---- 存盘 -------------------------------------------------------------------
dir.create("output/oof",  showWarnings = FALSE, recursive = TRUE)
dir.create("output/test", showWarnings = FALSE, recursive = TRUE)

.meta <- list(model = MODEL_NAME, tier = TIER, impute = IMPUTE_LINE,
              use_derived = USE_DERIVED, repeat_id = REPEAT_ID,
              fold_auc = fold_auc, cv_mean = cv_mean, cv_sd = cv_sd,
              oof_auc = oof_auc, best_iter = best_iter, minutes = elapsed)

if (REPEAT_ID > 0L) {
  dir.create("output/repeat", showWarnings = FALSE, recursive = TRUE)
  saveRDS(.meta, sprintf("output/repeat/rep%d_%s.rds", REPEAT_ID, MODEL_NAME))
  say(sprintf("\n已保存 output/repeat/rep%d_%s.rds\n", REPEAT_ID, MODEL_NAME))

} else if (TIER == "A") {
  saveRDS(oof,   sprintf("output/oof/oof_grid_%s.rds", MODEL_NAME))
  saveRDS(.meta, sprintf("output/oof/meta_grid_%s.rds", MODEL_NAME))
  say(sprintf("\n已保存 output/oof/oof_grid_%s.rds（长度 %s）\n",
              MODEL_NAME, format(length(oof), big.mark = ",")))

} else {
  saveRDS(oof,   sprintf("output/oof/oof_%s.rds", MODEL_NAME))
  saveRDS(.meta, sprintf("output/oof/meta_%s.rds", MODEL_NAME))

  say("\n生成 test 预测 ... ")
  set.seed(SEED)
  imp    <- .fit_imputer(X_pool)
  X_full <- .derive(.apply_imputer(imp, copy(X_pool)))
  X_test <- .derive(.apply_imputer(imp, copy(.test_all)))
  pred_test <- as.numeric(fit_predict(X_full, y_pool, X_test))

  saveRDS(pred_test, sprintf("output/test/test_%s.rds", MODEL_NAME))
  say("完成\n")
  say(sprintf("已保存 oof_%s.rds（%s）和 test_%s.rds（%s）\n",
              MODEL_NAME, format(length(oof), big.mark = ","),
              MODEL_NAME, format(length(pred_test), big.mark = ",")))
}
