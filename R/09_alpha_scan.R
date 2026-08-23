# =============================================================================
# 09_alpha_scan.R —— glmnet 的 alpha 敏感性分析（审查意见 2.5）
#
# 用法：Rscript R/09_alpha_scan.R
# 产出：output/alpha_scan.rds
#
# -----------------------------------------------------------------------------
# 问题
# -----------------------------------------------------------------------------
# 项目此前的两个 glmnet 格子都写死 alpha = 0.5（弹性网，L1 与 L2 正则化
# 各占一半），但从没解释过为什么是 0.5。审查意见指出这缺乏依据 —— 对的。
#
# alpha 的含义：
#   alpha = 0    纯 ridge，把相关特征的系数一起收缩，不做特征选择
#   alpha = 1    纯 lasso，会把一部分系数压到恰好 0，隐式做特征选择
#   0 < alpha < 1  两者混合
#
# 本题的特征结构对这个选择有明确指向：
#   - 特征只有 31 个，不需要大规模特征选择
#   - 但共线性很强（other_screen 和 screen_social 是原始列的**精确线性
#     组合**，设计矩阵秩亏；screen ↔ weekend 相关 0.80）
#   - 理论上 ridge 更适合处理共线性，lasso 在共线特征里会随机挑一个
#
# 所以先验预期是 alpha 偏小更好。跑一遍看实际情况。
# 用 L3 插补线（glmnet 上表现最好的那条）作为载体。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")
source("R/05_impute_L3.R")

SEED   <- 20260821L
N_FOLD <- 5L
ALPHAS <- c(0, 0.25, 0.5, 0.75, 1)

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

cat("======================================================\n")
cat("  glmnet alpha 敏感性分析（载体：L3 插补线）\n")
cat("  alpha = 0 纯 ridge  →  alpha = 1 纯 lasso\n")
cat("======================================================\n\n")

res <- list()
for (a in ALPHAS) {
  cat(sprintf("  alpha = %.2f ... ", a)); flush.console()
  fp <- make_glmnet(alpha = a)
  t0 <- Sys.time()

  aucs <- numeric(0)
  for (k in seq_len(N_FOLD)) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    imp  <- fit_imputer_L3(X_pool[tr])
    X_tr <- derive_features(apply_imputer_L3(imp, copy(X_pool[tr])))
    X_va <- derive_features(apply_imputer_L3(imp, copy(X_pool[va])))
    p <- fp(X_tr, y_pool[tr], X_va)
    aucs <- c(aucs, as.numeric(auc(roc(y_pool[va], p, quiet = TRUE))))
  }
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  res[[as.character(a)]] <- aucs
  cat(sprintf("AUC %.5f ± %.5f  (%.1f 分钟)\n", mean(aucs), sd(aucs), mins))
}

# ---- 报表 -------------------------------------------------------------------
means <- vapply(res, mean, 0)
best_a <- names(means)[which.max(means)]

cat("\n======================================================\n")
cat(sprintf("%8s %10s %10s %12s\n", "alpha", "AUC", "sd", "vs alpha=0.5"))
for (a in names(res)) {
  cat(sprintf("%8s %10.5f %10.5f %+12.5f%s\n",
              a, mean(res[[a]]), sd(res[[a]]),
              mean(res[[a]]) - means[["0.5"]],
              if (a == best_a) "   <- 最优" else ""))
}

# 与原先固定的 0.5 做配对检验
d <- res[[best_a]] - res[["0.5"]]
if (best_a != "0.5" && sd(d) > 0) {
  tt <- t.test(d)
  cat(sprintf("\n最优 alpha=%s 相对 alpha=0.5：%+.5f，%d/%d 折同号，p=%.4f\n",
              best_a, mean(d), sum(sign(d) == sign(d[1])), length(d), tt$p.value))
} else {
  cat("\nalpha=0.5 已是最优，或差异为零。\n")
}

cat(sprintf("\n极差 %.5f —— %s\n", diff(range(means)),
            if (diff(range(means)) < 0.001)
              "alpha 的选择对本题几乎没有影响，原先固定 0.5 无害"
            else
              "alpha 的选择有实质影响，报告中必须说明依据"))

saveRDS(res, "output/alpha_scan.rds")
cat("已保存 output/alpha_scan.rds\n")
