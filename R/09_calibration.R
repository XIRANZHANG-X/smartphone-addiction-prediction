# =============================================================================
# 09_calibration.R —— 概率校准评估（审查意见 2.6）
#
# 用法：Rscript R/09_calibration.R
# 产出：output/calibration.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 为什么要看校准，以及它和 AUC 的关系
# -----------------------------------------------------------------------------
# 比赛评价指标是 AUC，而 AUC **只看排序**：把所有预测值做任意单调变换，
# AUC 一点不变。所以从「拿分」角度，校准与我们无关。
#
# 但有三个理由必须看：
#
#   1. 我们交的是「概率」。说一个人有 0.83 的概率成瘾，就该有 83% 的
#      这类人真的成瘾。如果没有，这个数字就是在误导读者 ——
#      而这份数据的主题是行为健康，误导的代价不只是分数。
#
#   2. 集成时不同模型的概率尺度不一致会出问题。这正是我们的
#      07_ensemble.R 里 rank 平均和爬山法都在秩空间做的原因。
#
#   3. 如果评分标准换成 log-loss，校准立刻决定成败。
#
# -----------------------------------------------------------------------------
# 一个必须区分清楚的点
# -----------------------------------------------------------------------------
# 只有**单模型的输出**才是概率。集成里的 rank 平均和爬山法输出的是
# **秩**，被线性压缩到 (0,1) 之后长得像概率，但它不是 —— 谈它的校准
# 没有意义。因此本脚本只评估单模型，并在报告中明确这一点。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})

folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
y     <- readRDS("output/raw_train.rds")$addicted_label[sub]

fs <- list.files("output/oof", pattern = "^oof_grid_", full.names = FALSE)
fs <- fs[endsWith(fs, ".rds")]
if (!length(fs)) stop("output/oof/ 下没有 Tier A 结果，请先跑 R/run_grid.R")

models <- sub("^oof_grid_", "", sub(".rds", "", fs, fixed = TRUE))

# -----------------------------------------------------------------------------
# 指标
# -----------------------------------------------------------------------------

#' Brier score：预测概率与 0/1 标签的均方误差。越小越好。
#' 全预测基准率（0.7094）时 Brier = 0.7094 × (1-0.7094) = 0.2062，
#' 这是「什么都不知道」的参照点。
brier <- function(p, y) mean((p - y)^2)

#' Expected Calibration Error：按预测值分箱，
#' 每箱里 |平均预测值 − 实际正例率| 的加权平均。0 表示完美校准。
ece <- function(p, y, n_bin = 20L) {
  b <- cut(p, breaks = seq(0, 1, length.out = n_bin + 1L),
           include.lowest = TRUE)
  d <- data.table(p = p, y = y, b = b)[
    , .(n = .N, mp = mean(p), my = mean(y)), by = b][n > 0]
  sum(d$n * abs(d$mp - d$my)) / sum(d$n)
}

#' 可靠性曲线（reliability diagram）的数据
reliability <- function(p, y, n_bin = 10L) {
  b <- cut(p, breaks = quantile(p, seq(0, 1, length.out = n_bin + 1L)),
           include.lowest = TRUE, labels = FALSE)
  data.table(p = p, y = y, b = b)[
    , .(n = .N, pred_mean = mean(p), actual_rate = mean(y)), by = b][order(b)]
}

# -----------------------------------------------------------------------------
# 评估
# -----------------------------------------------------------------------------
base_rate <- mean(y)
cat("======================================================\n")
cat("  概率校准评估（Tier A OOF 预测）\n")
cat("======================================================\n")
cat(sprintf("基准率 %.4f，全预测基准率时 Brier = %.4f\n\n",
            base_rate, base_rate * (1 - base_rate)))

cat(sprintf("%-16s %8s %9s %8s %s\n",
            "模型", "AUC", "Brier", "ECE", "校准判读"))

res <- list()
for (i in seq_along(models)) {
  p <- readRDS(file.path("output/oof", fs[i]))
  if (length(p) != length(y) || anyNA(p)) next

  # glmnet/xgboost/lightgbm/ranger 都直接输出 (0,1) 概率，不需要变换
  a  <- as.numeric(auc(roc(y, p, quiet = TRUE)))
  bs <- brier(p, y)
  ec <- ece(p, y)
  verdict <- if (ec < 0.01) "很好" else if (ec < 0.03) "尚可" else "偏差明显"

  cat(sprintf("%-16s %8.5f %9.5f %8.5f %s\n", models[i], a, bs, ec, verdict))
  res[[models[i]]] <- list(auc = a, brier = bs, ece = ec,
                           reliability = reliability(p, y))
}

# -----------------------------------------------------------------------------
# 最好那个模型的可靠性曲线
# -----------------------------------------------------------------------------
if (length(res)) {
  best <- names(res)[which.max(vapply(res, function(z) z$auc, 0))]
  cat(sprintf("\n---- %s 的可靠性曲线（按预测值十分位）----\n", best))
  cat(sprintf("%6s %9s %12s %12s %10s\n",
              "分位", "样本数", "平均预测值", "实际正例率", "偏差"))
  rel <- res[[best]]$reliability
  for (i in seq_len(nrow(rel))) {
    r <- rel[i]
    cat(sprintf("%6d %9s %12.4f %12.4f %+10.4f\n",
                r$b, format(r$n, big.mark = ","),
                r$pred_mean, r$actual_rate, r$actual_rate - r$pred_mean))
  }
  cat("\n偏差列全部接近 0 = 模型说 80% 就真的有 80% 会成瘾。\n")
}

cat("\n------------------------------------------------------\n")
cat("注意：以上只评估**单模型**输出。\n")
cat("07_ensemble.R 里的 rank 平均和爬山法输出的是**秩**，\n")
cat("线性压缩到 (0,1) 后长得像概率但并不是，谈它的校准没有意义。\n")
cat("AUC 只看排序，因此校准好坏不影响我们的比赛成绩 ——\n")
cat("但我们交的是概率，这个数字应当能被读者当真。\n")

saveRDS(res, "output/calibration.rds")
cat("\n已保存 output/calibration.rds\n")
