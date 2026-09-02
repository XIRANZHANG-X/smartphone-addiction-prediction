# =============================================================================
# 09_ablation.R —— 特征消融实验
#
# 用法：Rscript R/09_ablation.R
# 产出：output/ablation.rds
#
# 回答三个问题：
#   Q1 派生特征有用吗？插补制造的冗余变量会不会拖累 L3？
#   Q2 EDA 判定的「噪声特征」该不该筛掉？
#   Q3 如果该筛，损失/收益具体来自哪一个特征？
#
# -----------------------------------------------------------------------------
# 关于代码复用（审查意见 3.3）
# -----------------------------------------------------------------------------
# 本脚本早期版本自己重新实现了一遍 xgboost 的训练逻辑。这意味着主模型
# 一旦改参数（比如加早停），消融脚本不会跟着改，两边结果就不可比了 ——
# 而消融的全部意义就在于「除了被消融的特征，其他一切都相同」。
#
# 现在统一从 R/lib_models.R 取模型工厂，与 06_model_*.R 用的是同一份代码。
#
# -----------------------------------------------------------------------------
# 一个必须声明的方法学缺陷（审查意见 1.5）
# -----------------------------------------------------------------------------
# 「哪些特征是噪声」这个判断来自 EDA，而 EDA 用了全部训练数据**和标签**。
# 然后我们又在同一套折叠上评估删掉它们的效果 —— 严格说应当把特征选择
# 也放进折内。
#
# 我们没有这么做，但影响是可控的，原因有二：
#   1. 判定依据极干净（69 万行上边际差值 <= 0.004）
#   2. **最终模型不做特征选择** —— 因为消融的结论就是「筛了也不涨」。
#      所以这个偏差只影响消融结果的解读，不影响交付模型的分数。
# =============================================================================

# -----------------------------------------------------------------------------
# ⚠ 口径声明（2026-09-03 更新）
# -----------------------------------------------------------------------------
# 本脚本自己写 CV 循环，不走 06_framework.R，但**调的是同一个 prepare_fold()**
# （见下方第 90 行附近），所以插补、派生、逐取值编码三步与主网格完全一致。
#
# 此前这里写着「因此它不做 target encoding」——那句话在 prepare_fold() 抽出来
# 之前是对的，之后就不对了。2026-09-03 连同本次重跑一并更正。
#
# 需要注意的是**消融的解释会因此改变**：加入编码后，原始列的重要性被编码列
# 大量吸收（R/13_importance.R：每日屏幕时间的置换重要性从 0.1332 掉到 0.0061）。
# 所以「删掉某原始列掉多少分」在有编码的配置下会明显小于编码之前——
# 这不是那一列变得不重要了，而是**它的信息还有第二条通路**。
#
# 数据量：Tier A 20 万行，全部配置约需 3 小时。

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
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
             "gaming_share", "free_frac")

# 缺失指示列已从默认特征集移除（见 R/03_features.R 的说明），因此在新的
# features_raw.rds 上 NA_IND 为空，相关变体会自动跳过。
# 要复现「缺失指示无效」那一组消融，先重建带指示列的特征矩阵：
#   WITH_NA_INDICATORS <- TRUE; FORCE_REBUILD <- TRUE; source("R/03_features.R")
NA_IND     <- grep("^is_na_", names(feat), value = TRUE)
HAS_NA_IND <- length(NA_IND) > 0L
NA_BLOCK   <- if (HAS_NA_IND) c(NA_IND, "n_missing") else character(0)
if (!HAS_NA_IND) cat("提示：features_raw.rds 不含缺失指示列，相关变体跳过。\n")
CAT3    <- c("stress_level", "academic_work_impact", "gender")

#' 跑一个消融变体
#' @param line 插补线
#' @param drop 要排除的列
run_variant <- function(line, drop = character(0)) {
  source(sprintf("R/05_impute_%s.R", line), local = FALSE)
  fit_i   <- get(paste0("fit_imputer_",   line))
  apply_i <- get(paste0("apply_imputer_", line))

  # 与 06_model_*_xgboost.R 用同一个工厂、同一套默认参数、同样的早停
  fp <- make_xgb(drop_extra = drop)

  aucs <- numeric(0); nf <- NA_integer_; iters <- integer(0)
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    # 插补 -> 派生 -> 逐取值编码，三条纪律固化在 prepare_fold() 里
    fold <- prepare_fold(X_pool[tr], y_pool[tr], X_pool[va], fit_i, apply_i)
    X_tr <- fold$tr; X_va <- fold$va
    if (is.na(nf)) nf <- length(setdiff(names(X_tr),
                       c("id", "addicted_label", "is_train", drop)))
    p <- fp(X_tr, y_pool[tr], X_va)
    bi <- attr(p, "best_iteration"); if (!is.null(bi)) iters <- c(iters, bi)
    aucs <- c(aucs, as.numeric(auc(roc(y_pool[va], as.numeric(p), quiet = TRUE))))
  }
  list(auc = aucs, n_feat = nf, iters = iters)
}

# -----------------------------------------------------------------------------
# 阶段一：派生特征与整组噪声（L1 / L3 对照）
# -----------------------------------------------------------------------------
sets1 <- list(
  full    = character(0),
  noderiv = DERIVED,
  nonoise = c(CAT3, "notifications_per_day", NA_BLOCK),
  lean    = c(CAT3, "notifications_per_day", NA_BLOCK, "free_frac")
)

res <- list()
cat("======================================================\n")
cat("  阶段一：派生特征 与 整组噪声（L1 作控制组）\n")
cat("======================================================\n")
for (line in c("L1", "L3")) {
  for (sn in names(sets1)) {
    key <- paste0(line, "_", sn)
    cat(sprintf("  %-14s ... ", key)); flush.console()
    r <- run_variant(line, sets1[[sn]])
    res[[key]] <- r
    cat(sprintf("%2d 特征  AUC %.5f ± %.5f\n", r$n_feat, mean(r$auc), sd(r$auc)))
  }
}

# -----------------------------------------------------------------------------
# 阶段二：拆开噪声组，定位具体是哪个特征（只用 L1）
# -----------------------------------------------------------------------------
sets2 <- list(
  no_cat3  = CAT3,
  no_notif = "notifications_per_day",
  no_opens = "app_opens_per_day",
  no_age   = "age"
)
if (HAS_NA_IND) sets2 <- c(list(no_naind = NA_BLOCK), sets2)

cat("\n======================================================\n")
cat("  阶段二：逐个拆解（L1）\n")
cat("======================================================\n")
for (sn in names(sets2)) {
  key <- paste0("L1_", sn)
  cat(sprintf("  %-14s ... ", key)); flush.console()
  r <- run_variant("L1", sets2[[sn]])
  res[[key]] <- r
  cat(sprintf("%2d 特征  AUC %.5f ± %.5f\n", r$n_feat, mean(r$auc), sd(r$auc)))
}

# -----------------------------------------------------------------------------
# 报表
# -----------------------------------------------------------------------------
cat("\n======================================================\n")
cat("  汇总\n")
cat("======================================================\n")
cat(sprintf("%-16s %7s %10s %10s\n", "变体", "特征数", "AUC", "sd"))
for (k in names(res))
  cat(sprintf("%-16s %7d %10.5f %10.5f\n",
              k, res[[k]]$n_feat, mean(res[[k]]$auc), sd(res[[k]]$auc)))

cat("\n============ 配对检验（vs 同线 full）============\n")
cat(sprintf("%-16s %+10s %8s %7s %9s\n",
            "变体", "均值差", "Cohen d", "同号", "p(t)"))
for (base_line in c("L1", "L3")) {
  b <- res[[paste0(base_line, "_full")]]
  if (is.null(b)) next
  for (k in names(res)) {
    if (!startsWith(k, paste0(base_line, "_")) || endsWith(k, "_full")) next
    d <- res[[k]]$auc - b$auc
    tt <- t.test(d)
    coh <- if (sd(d) > 0) mean(d) / sd(d) else NA_real_
    cat(sprintf("%-16s %+10.5f %8.2f %3d/%d %9.4f\n",
                k, mean(d), coh, sum(sign(d) == sign(d[1])), length(d),
                tt$p.value))
  }
}

cat("\n注：n=5 的配对 t 检验不够稳健，Cohen's d 与符号一致性是更可靠的判据。\n")
cat("    关键对比的 n=15 版本见 R/09_repeated_cv.R。\n")

saveRDS(res, "output/ablation.rds")
cat("\n已保存 output/ablation.rds\n")
