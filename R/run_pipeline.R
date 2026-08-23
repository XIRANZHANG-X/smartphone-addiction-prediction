# =============================================================================
# run_pipeline.R —— 一键跑完全部实验与交付物
#
# 用法：Rscript R/run_pipeline.R [起始步骤]
#   例如 Rscript R/run_pipeline.R 3   从第 3 步开始
#
# 每一步都会跳过已有产物，所以中断之后直接重跑即可续上。
#
# 步骤顺序按「依赖 + 性价比」排：
#   先把便宜且能改结论的做完（调参、消融），再做贵的（重复 CV、Tier B）。
# =============================================================================

rscript <- file.path(R.home("bin"), "Rscript")

STEPS <- list(
  list(name = "实验网格（14 格）",       cmd = "R/run_grid.R",       args = character(0),
       check = function() length(list.files("output/oof", "^meta_grid_")) >= 14),

  list(name = "特征消融",                cmd = "R/09_ablation.R",    args = character(0),
       check = function() file.exists("output/ablation.rds")),

  list(name = "超参数搜索 xgboost",      cmd = "R/10_tune.R",        args = "xgboost",
       check = function() file.exists("output/tune_xgboost.rds")),

  list(name = "glmnet alpha 敏感性",     cmd = "R/09_alpha_scan.R",  args = character(0),
       check = function() file.exists("output/alpha_scan.rds")),

  list(name = "概率校准",                cmd = "R/09_calibration.R", args = character(0),
       check = function() file.exists("output/calibration.rds")),

  list(name = "重复交叉验证（n=15）",     cmd = "R/09_repeated_cv.R", args = character(0),
       check = function() length(list.files("output/repeat", "^rep")) >= 18),

  list(name = "Tier B 全量重训",         cmd = "R/run_tierb.R",      args = character(0),
       check = function() length(list.files("output/test", "^test_")) >= 5),

  list(name = "集成",                    cmd = "R/07_ensemble.R",    args = character(0),
       check = function() file.exists("output/ensemble_best.rds")),

  list(name = "生成提交文件",            cmd = "R/08_submit.R",      args = character(0),
       check = function() length(list.files("submissions", "\\.csv$")) > 1),

  list(name = "汇总结果表",              cmd = "R/11_report.R",      args = character(0),
       check = function() FALSE)   # 总是重跑，它很快且要刷新
)

args  <- commandArgs(trailingOnly = TRUE)
start <- if (length(args)) as.integer(args[1]) else 1L
FORCE <- nzchar(Sys.getenv("FORCE"))

cat("======================================================\n")
cat(sprintf("  完整流水线：%d 步，从第 %d 步开始\n", length(STEPS), start))
cat(sprintf("  开始时间：%s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat("======================================================\n\n")

t_all <- Sys.time()
done <- character(0); failed <- character(0)

for (i in seq_along(STEPS)) {
  if (i < start) next
  s <- STEPS[[i]]

  if (!FORCE && isTRUE(tryCatch(s$check(), error = function(e) FALSE))) {
    cat(sprintf("[%d/%d] %-24s 已完成，跳过\n", i, length(STEPS), s$name))
    done <- c(done, s$name); next
  }

  cat(sprintf("\n##### [%d/%d] %s #####\n", i, length(STEPS), s$name))
  t0 <- Sys.time()
  status <- system2(rscript, c(s$cmd, s$args), stdout = "", stderr = "")
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (status == 0) {
    done <- c(done, s$name)
    cat(sprintf("##### ✓ %s 完成，耗时 %.1f 分钟 #####\n", s$name, mins))
  } else {
    failed <- c(failed, s$name)
    cat(sprintf("##### ✗ %s 失败（退出码 %s），耗时 %.1f 分钟 #####\n",
                s$name, status, mins))
    cat("      后续步骤可能依赖它，但流水线继续 —— 独立的步骤仍然能出结果。\n")
  }
}

cat("\n======================================================\n")
cat(sprintf("  流水线结束，总耗时 %.1f 分钟\n",
            as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
cat(sprintf("  完成 %d 步，失败 %d 步\n", length(done), length(failed)))
if (length(failed)) cat("  失败：", paste(failed, collapse = ", "), "\n")
cat("======================================================\n")
