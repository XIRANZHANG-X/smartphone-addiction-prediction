# =============================================================================
# 09_stats.R —— 模型对比的统计工具
#
# 用法：source("R/09_stats.R") 后调用 grid_table() / compare()
#
# -----------------------------------------------------------------------------
# 为什么需要专门写一个统计模块（审查意见 1.4）
# -----------------------------------------------------------------------------
# 原先所有方法对比都用「5 折配对 t 检验」，报 p = 0.0231 这样的数字。
# 审查意见指出 n=5 时 t 检验的正态性假设无法验证，这是对的。
#
# 但它建议改用 Wilcoxon 符号秩检验 —— 这个药方会让情况更糟：
#
#   n=5 时符号秩统计量最多只有 2^5 = 32 种排列，
#   双侧最小可能 p 值是 2/32 = 0.0625。
#   也就是说即使 5/5 折全部同号、差异再大，也永远达不到 p < 0.05。
#
# 换过去等于主动放弃全部统计功效。真正的解法是**增加配对样本量**，
# 而不是换一个在小样本下更保守的检验。
#
# 因此本模块的做法是：
#   1. 主口径改用**重复交叉验证**（3 个不同 fold seed × 5 折 = n=15）
#      见 R/09_repeated_cv.R
#   2. 无论 n 多少，一律同时报 **effect size（Cohen's d）** 和
#      **符号一致性**，不让 p 值单独承担结论
#   3. Wilcoxon 仍然计算并报出，但注明它在 n=5 时的下限
#   4. 明确标注每个 p 值背后的 n
# =============================================================================

suppressMessages({library(data.table); library(pROC)})

# -----------------------------------------------------------------------------
# 读取网格结果
# -----------------------------------------------------------------------------

#' 载入所有 Tier A 网格格子的元数据
#' @return data.table，一行一个格子
grid_table <- function(dir = "output/oof") {
  fs <- list.files(dir, pattern = "^meta_grid_", full.names = TRUE)
  if (!length(fs)) return(data.table())

  rows <- lapply(fs, function(f) {
    m <- readRDS(f)
    data.table(
      model      = m$model,
      impute     = m$impute,
      algo       = sub("^L[0-9]+_", "", m$model),
      cv_mean    = m$cv_mean,
      cv_sd      = m$cv_sd,
      oof_auc    = m$oof_auc,
      best_iter  = if (length(m$best_iter)) mean(m$best_iter) else NA_real_,
      minutes    = m$minutes
    )
  })
  rbindlist(rows)[order(-cv_mean)]
}

#' 取某个格子的逐折 AUC
fold_auc_of <- function(model, dir = "output/oof") {
  f <- file.path(dir, sprintf("meta_grid_%s.rds", model))
  if (!file.exists(f)) return(NULL)
  readRDS(f)$fold_auc
}

#' 取某个格子在重复 CV 下的全部逐折 AUC（n = 5 × 重复次数）
fold_auc_repeated <- function(model, dir = "output/repeat") {
  fs <- list.files(dir, pattern = sprintf("^rep[0-9]+_%s\\.rds$", model),
                   full.names = TRUE)
  if (!length(fs)) return(NULL)
  unlist(lapply(fs, function(f) readRDS(f)$fold_auc))
}

# -----------------------------------------------------------------------------
# 配对比较
# -----------------------------------------------------------------------------

#' 两个模型的配对比较
#'
#' @param a,b 模型名
#' @param use_repeated TRUE 则优先使用重复 CV 的样本（n=15），
#'   没有则回退到单次 CV（n=5）并在输出中标注
#' @return 单行 data.table
compare <- function(a, b, use_repeated = TRUE) {
  da <- if (use_repeated) fold_auc_repeated(a) else NULL
  db <- if (use_repeated) fold_auc_repeated(b) else NULL
  src <- "repeated"

  if (is.null(da) || is.null(db) || length(da) != length(db)) {
    da <- fold_auc_of(a); db <- fold_auc_of(b); src <- "single"
  }
  if (is.null(da) || is.null(db)) return(NULL)

  d <- da - db
  n <- length(d)

  # Cohen's d（配对）：差值均值除以差值标准差。
  # 这个数不随样本量变化，是比 p 值更稳定的「差异有多大」的度量。
  coh <- if (stats::sd(d) > 0) mean(d) / stats::sd(d) else NA_real_

  tt <- tryCatch(stats::t.test(d), error = function(e) NULL)

  # Wilcoxon：n=5 时双侧下限是 0.0625，达不到 0.05。一并算出来供参考。
  wt <- tryCatch(suppressWarnings(stats::wilcox.test(d)),
                 error = function(e) NULL)

  # 符号检验：k 个正号里有多少，二项检验。对分布形状不作任何假设。
  k <- sum(d > 0)
  bt <- stats::binom.test(k, n, p = 0.5)

  data.table(
    a = a, b = b, source = src, n = n,
    mean_diff = mean(d),
    cohens_d  = coh,
    same_sign = sprintf("%d/%d", max(k, n - k), n),
    p_t       = if (!is.null(tt)) tt$p.value else NA_real_,
    p_wilcox  = if (!is.null(wt)) wt$p.value else NA_real_,
    p_sign    = bt$p.value
  )
}

#' 批量比较并打印
compare_many <- function(pairs, use_repeated = TRUE) {
  out <- rbindlist(lapply(pairs, function(p) compare(p[1], p[2], use_repeated)),
                   fill = TRUE)
  if (!nrow(out)) { cat("没有可比较的结果\n"); return(invisible(NULL)) }

  cat(sprintf("%-16s %-16s %4s %3s %10s %8s %6s %9s %9s %9s\n",
              "A", "B", "来源", "n", "均值差", "Cohen d", "同号",
              "p(t)", "p(符号秩)", "p(符号)"))
  for (i in seq_len(nrow(out))) {
    r <- out[i]
    cat(sprintf("%-16s %-16s %4s %3d %+10.5f %8.2f %6s %9.4f %9.4f %9.4f\n",
                r$a, r$b, if (r$source == "repeated") "重复" else "单次",
                r$n, r$mean_diff, r$cohens_d, r$same_sign,
                r$p_t, r$p_wilcox, r$p_sign))
  }
  if (any(out$n <= 5)) {
    cat("\n注意：n<=5 的行，符号秩检验的双侧 p 值下限是 0.0625，",
        "永远达不到 0.05 —— 这是检验本身的分辨率极限，不是证据不足。\n", sep = "")
  }
  invisible(out)
}

#' Cohen's d 的常规解读
interpret_d <- function(d) {
  ad <- abs(d)
  if (is.na(ad)) return("NA")
  if (ad < 0.2) "可忽略" else if (ad < 0.5) "小" else
    if (ad < 0.8) "中" else "大"
}
