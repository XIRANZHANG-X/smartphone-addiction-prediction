# =============================================================================
# 09_repeated_cv.R —— 重复交叉验证（审查意见 1.4 的真正解法）
#
# 用法：Rscript R/09_repeated_cv.R [模型名 ...]
#   不带参数 = 跑默认的关键对比集合
#
# -----------------------------------------------------------------------------
# 问题
# -----------------------------------------------------------------------------
# 项目此前的全部方法对比都基于 5 折配对 t 检验，n=5。审查意见指出：
# n=5 时正态性假设无法验证，t 检验对偏离正态很敏感，报出的 p 值不够稳健。
# 这个批评是对的。
#
# 但它建议改用 Wilcoxon 符号秩检验 —— 那会让情况更糟：
# n=5 时符号秩统计量只有 2^5 = 32 种排列，双侧最小可能 p 值是 2/32 = 0.0625。
# 即使 5 折全部同号、差异再大，也永远达不到 p < 0.05。
# 换过去等于主动放弃全部统计功效。
#
# -----------------------------------------------------------------------------
# 解法
# -----------------------------------------------------------------------------
# 换更保守的检验解决不了样本量问题，**增加配对样本量**才行。
#
# 用 3 个不同的 fold seed 各做一次完整 5 折 CV，得到 15 个配对差值。
# n=15 时 t 检验的稳健性显著改善，符号秩检验也有了实际分辨率
# （双侧下限降到 2^-15，不再是瓶颈）。
#
# 重要：这些重复只用于**统计检验**，不进网格对比表。
# 网格表里的数字仍然来自冻结的 folds.rds（REPEAT_ID = 0），
# 那是全组共享的唯一口径。重复 CV 是补充分析，产物存在 output/repeat/。
# =============================================================================

N_REPEAT <- 3L

# 默认只对「有结论要下」的格子做重复 —— 重复 CV 成本是单次的 3 倍，
# 没必要对全部 14 格都做。
DEFAULT_CELLS <- c(
  "L1_xgboost", "L2_xgboost", "L3_xgboost", "L4_xgboost",  # 核心阶梯
  "L2_glmnet",  "L3_glmnet"                                 # 反向预测那一对
)

args  <- commandArgs(trailingOnly = TRUE)
cells <- if (length(args)) args else DEFAULT_CELLS

rscript <- file.path(R.home("bin"), "Rscript")

cat("======================================================\n")
cat(sprintf("  重复交叉验证：%d 个格子 × %d 次重复 = %d 次运行\n",
            length(cells), N_REPEAT, length(cells) * N_REPEAT))
cat(sprintf("  目标：把配对样本量从 n=5 提到 n=%d\n", 5L * N_REPEAT))
cat("======================================================\n\n")

t_all <- Sys.time()
ok <- 0L; failed <- character(0)

for (cell in cells) {
  f <- sprintf("R/06_model_%s.R", cell)
  if (!file.exists(f)) { cat(sprintf("[跳过] %s 文件不存在\n", cell)); next }

  for (r in seq_len(N_REPEAT)) {
    out <- sprintf("output/repeat/rep%d_%s.rds", r, cell)
    if (file.exists(out) && !nzchar(Sys.getenv("FORCE"))) {
      cat(sprintf("  %-14s 重复 %d  已存在，跳过\n", cell, r)); ok <- ok + 1L; next
    }

    cat(sprintf("  %-14s 重复 %d ... ", cell, r)); flush.console()
    t0 <- Sys.time()
    # 每次跑独立子进程，REPEAT_ID 通过环境变量传给框架
    status <- system2(rscript, f, stdout = NULL, stderr = NULL,
                      env = sprintf("REPEAT_ID=%d", r))
    mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

    if (status == 0 && file.exists(out)) {
      m <- readRDS(out)
      cat(sprintf("AUC %.5f ± %.5f  (%.1f 分钟)\n", m$cv_mean, m$cv_sd, mins))
      ok <- ok + 1L
    } else {
      cat(sprintf("失败（退出码 %s）\n", status))
      failed <- c(failed, sprintf("%s#%d", cell, r))
    }
  }
}

cat(sprintf("\n完成 %d 次，失败 %d 次，总耗时 %.1f 分钟\n",
            ok, length(failed),
            as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
if (length(failed)) cat("失败：", paste(failed, collapse = ", "), "\n")
cat("\n下一步：source(\"R/09_stats.R\") 后用 compare_many() 查看 n=15 的检验结果\n")
