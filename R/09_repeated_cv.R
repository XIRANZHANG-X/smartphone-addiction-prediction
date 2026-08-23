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
# 不含 L4：它每格 38 分钟（missRanger），三次重复要 114 分钟，
# 而 L3 vs L4 的差异已经很大（Cohen's d = 8.49，5/5 折同号），
# 不需要 n=15 来加固。真正需要更大样本量的是 L1 vs L2 这种边缘对比
# （原 n=5 时 p=0.023，4/5 同号）。
DEFAULT_CELLS <- c(
  "L1_xgboost", "L2_xgboost", "L3_xgboost",   # 核心阶梯（去掉最贵的 L4）
  "L2_glmnet",  "L3_glmnet"                    # 反向预测那一对
)

args  <- commandArgs(trailingOnly = TRUE)
cells <- if (length(args)) args else DEFAULT_CELLS

rscript <- file.path(R.home("bin"), "Rscript")

# -----------------------------------------------------------------------------
# 带环境变量地跑一个子进程
# -----------------------------------------------------------------------------
# 不能用 system2(env = ...) —— 它在 Windows 上是坏的：
# R 的实现是给命令加 Unix 的 `env` 前缀，而 Windows 没有这个命令，
# 结果整条命令秒退（退出码非 0，耗时 0.0 分钟），而且错得很安静。
#
# 正确做法是在父进程里 Sys.setenv()，子进程自然继承。
run_with_env <- function(rscript, args, vars = character(0)) {
  old <- vapply(names(vars), function(k) Sys.getenv(k, unset = NA_character_),
                character(1))
  do.call(Sys.setenv, as.list(vars))
  on.exit({
    for (k in names(vars)) {
      if (is.na(old[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv, setNames(list(old[[k]]), k))
    }
  }, add = TRUE)
  system2(rscript, args, stdout = "", stderr = "")
}

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
    status <- run_with_env(rscript, f, c(REPEAT_ID = as.character(r)))
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
