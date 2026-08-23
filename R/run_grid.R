# =============================================================================
# run_grid.R —— 依次跑完整个 4×4 实验网格
#
# 用法：Rscript R/run_grid.R  [格子名 ...]
#   不带参数 = 跑全部 14 个格子（跳过已有结果的）
#   带参数   = 只跑指定的，例如 Rscript R/run_grid.R L1_xgboost L2_xgboost
#   环境变量 FORCE=1 则连已有结果的也重跑
#
# 顺序按「快的先跑」排，这样最有价值的对比结果最早拿到。
#
# -----------------------------------------------------------------------------
# 为什么每个格子要开独立子进程
# -----------------------------------------------------------------------------
# 早期版本用 sys.source(f, envir = new.env()) 做隔离，结果 14 个格子全部
# 秒退报「source 框架之前必须先定义 MODEL_NAME」。
#
# 原因：模型文件里的 source("R/06_framework.R") 用的是默认 source()，
# 它在**全局环境**求值，看不到我们塞进 new.env() 里的 MODEL_NAME。
#
# 与其去改 14 个文件的 source 写法，不如让每个格子跑在自己的 R 进程里 ——
# 这本来就是组员实际使用的方式（Rscript R/06_model_X.R），
# 隔离彻底，而且一个格子崩了不影响其余。
# =============================================================================

ORDER <- c(
  # 梯度提升树：核心对比，最快
  "L1_xgboost",  "L2_xgboost",  "L3_xgboost",  "L4_xgboost",
  "L1_lightgbm", "L2_lightgbm", "L3_lightgbm", "L4_lightgbm",
  # 线性模型：必须插补，是「插补价值」那条结论的关键
  "L2_glmnet",   "L3_glmnet",   "L4_glmnet",
  # 随机森林：最慢，放最后
  "L2_ranger",   "L3_ranger",   "L4_ranger"
)

args  <- commandArgs(trailingOnly = TRUE)
todo  <- if (length(args)) args else ORDER
FORCE <- nzchar(Sys.getenv("FORCE"))

# 找到当前这个 R 的 Rscript，保证子进程用同一个版本
rscript <- file.path(R.home("bin"), "Rscript")

cat("======================================================\n")
cat(sprintf("  实验网格：%d 个格子%s\n", length(todo),
            if (FORCE) "（强制重跑）" else "（跳过已完成的）"))
cat(sprintf("  开始时间：%s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat("======================================================\n\n")

t_all <- Sys.time()
ok <- character(0); failed <- character(0); skipped <- character(0)

for (i in seq_along(todo)) {
  cell <- todo[i]
  f <- sprintf("R/06_model_%s.R", cell)

  if (!file.exists(f)) {
    cat(sprintf("[%d/%d] %s —— 文件不存在，跳过\n\n", i, length(todo), cell))
    next
  }

  # 有 meta 文件才算真正完成 —— 只有 oof 没有 meta 的是旧框架的遗留物
  done <- file.exists(sprintf("output/oof/meta_grid_%s.rds", cell))
  if (done && !FORCE) {
    m <- readRDS(sprintf("output/oof/meta_grid_%s.rds", cell))
    cat(sprintf("[%d/%d] %-14s 已完成（AUC %.5f），跳过\n\n",
                i, length(todo), cell, m$cv_mean))
    skipped <- c(skipped, cell)
    next
  }

  cat(sprintf("---- [%d/%d] %s ----\n", i, length(todo), cell))
  t0 <- Sys.time()

  status <- system2(rscript, f, stdout = "", stderr = "")

  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  if (status == 0 && file.exists(sprintf("output/oof/meta_grid_%s.rds", cell))) {
    ok <- c(ok, cell)
    cat(sprintf("  ✓ 完成，耗时 %.1f 分钟\n\n", mins))
  } else {
    failed <- c(failed, cell)
    cat(sprintf("  ✗ 失败（退出码 %s），耗时 %.1f 分钟\n\n", status, mins))
  }
}

cat("======================================================\n")
cat(sprintf("  全部结束，总耗时 %.1f 分钟\n",
            as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
cat(sprintf("  成功 %d  跳过 %d  失败 %d\n",
            length(ok), length(skipped), length(failed)))
if (length(ok))      cat(sprintf("  成功：%s\n", paste(ok, collapse = ", ")))
if (length(failed))  cat(sprintf("  失败：%s\n", paste(failed, collapse = ", ")))
cat("======================================================\n")
