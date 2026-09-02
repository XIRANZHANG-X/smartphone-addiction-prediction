# =============================================================================
# run_grid_full.R —— 在**全量 69 万行**上跑完整个 4×4 实验网格
#
# 用法：
#   Rscript R/run_grid_full.R              跑全部 14 格
#   Rscript R/run_grid_full.R --no-l4      跳过四个 L4 格（它们占了 80% 的时间）
#   Rscript R/run_grid_full.R L1_xgboost   只跑指定的格子
#   FORCE=1 Rscript R/run_grid_full.R      连已有结果的也重跑
#
# 产出：output/oof/oof_<名字>.rds（691,369）
#       output/oof/meta_<名字>.rds
#       output/test/test_<名字>.rds（296,302）
#
# -----------------------------------------------------------------------------
# 与 run_grid.R 的关系
# -----------------------------------------------------------------------------
# run_grid.R 跑的是 Tier A（20 万子样本），产物叫 *_grid_*，用于**方法对比**。
# 本脚本跑 Tier B（全量），产物不带 grid 前缀，用于**交付**。
#
# 两者并存是有意的，不要合并：
#   - Tier A 便宜，是所有对比实验的载体，其内部可比性由冻结折叠保证；
#   - Tier B 昂贵，但它是真正要交付的数字。
#
# ⚠ 已知的一个代价（2026-09-01 才发现）：**Tier A 会系统性低估吃样本量的方法**。
#   精确取值 one-hot 在 16 万训练行上是 0.95583，在 55 万行上是 0.95929 ——
#   差 0.0035 纯粹来自样本量。所以 Tier A 上的"这个方法不行"要小心，
#   尤其当该方法的参数量随取值数增长时。见 docs/讨论区核查.md 第十三节。
#
# -----------------------------------------------------------------------------
# 为什么 L4 可以单独跳过
# -----------------------------------------------------------------------------
# L4（PMM 随机插补，missRanger）在 Tier A 上单格就要 34~39 分钟，
# 四格合计 149 分钟，占整个网格的 81%。按实测的 3.53 倍缩放，
# 全量四格约需 8.8 小时，而其余十格加起来约 2 小时。
#
# L4 那条线的结论（"越复杂的插补在 GBDT 上越差"）在 Tier A 上已经很稳，
# 所以如果时间紧，`--no-l4` 先拿到 84% 的信息量。
# =============================================================================

suppressMessages(library(data.table))

ALL <- c(
  # 便宜的先跑，最有价值的对比最早拿到
  "L1_xgboost",  "L2_xgboost",  "L3_xgboost",
  "L1_lightgbm", "L2_lightgbm", "L3_lightgbm",
  "L2_glmnet",   "L3_glmnet",
  "L2_ranger",   "L3_ranger",
  # L4 最贵，放最后 —— 中断也不影响前面十格
  "L4_xgboost",  "L4_lightgbm", "L4_glmnet", "L4_ranger"
)

args   <- commandArgs(trailingOnly = TRUE)
no_l4  <- "--no-l4" %in% args
picked <- setdiff(args, "--no-l4")
cells  <- if (length(picked)) picked else ALL
if (no_l4) cells <- grep("^L4_", cells, value = TRUE, invert = TRUE)

FORCE   <- nzchar(Sys.getenv("FORCE"))
rscript <- file.path(R.home("bin"), "Rscript")

#' Windows 上 system2(env=) 是坏的（R 会前置 Unix 的 env 命令），
#' 所以用 Sys.setenv + 继承。与 run_tierb.R 里的同名函数一致。
run_with_env <- function(rscript, args, vars = character(0)) {
  old <- vapply(names(vars), function(k) Sys.getenv(k, unset = NA_character_),
                character(1))
  do.call(Sys.setenv, as.list(vars))
  on.exit({
    for (k in names(vars))
      if (is.na(old[[k]])) Sys.unsetenv(k)
      else do.call(Sys.setenv, setNames(list(old[[k]]), k))
  }, add = TRUE)
  system2(rscript, args, stdout = "", stderr = "")
}

cat(strrep("=", 70), "\n")
cat(sprintf("  全量网格：%d 格%s\n", length(cells), if (no_l4) "（已跳过 L4）" else ""))
cat(sprintf("  开始：%s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(strrep("=", 70), "\n\n")

t_all <- Sys.time(); ok <- character(0); failed <- character(0); skipped <- character(0)
log <- list()

for (i in seq_along(cells)) {
  cell <- cells[i]
  f    <- sprintf("R/06_model_%s.R", cell)
  meta <- sprintf("output/oof/meta_%s.rds", cell)

  if (!file.exists(f)) { cat(sprintf("[%d/%d] %s —— 找不到 %s，跳过\n\n",
                                     i, length(cells), cell, f))
                         failed <- c(failed, cell); next }
  if (file.exists(meta) && !FORCE) {
    m <- readRDS(meta)
    cat(sprintf("[%d/%d] %s —— 已有结果（AUC %.5f），跳过。FORCE=1 可重跑\n\n",
                i, length(cells), cell, m$cv_mean))
    skipped <- c(skipped, cell); next
  }

  cat(sprintf("[%d/%d] %s ...\n", i, length(cells), cell))
  t0 <- Sys.time()
  status <- run_with_env(rscript, f, c(TIER = "B"))
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (status == 0 && file.exists(meta)) {
    m <- readRDS(meta)
    cat(sprintf("  ✓ AUC %.5f ± %.5f，%.1f 分钟\n\n", m$cv_mean, m$cv_sd, mins))
    ok <- c(ok, cell); log[[cell]] <- list(auc = m$cv_mean, sd = m$cv_sd, minutes = mins)
  } else {
    cat(sprintf("  ✗ 失败（退出码 %s），%.1f 分钟\n\n", status, mins))
    failed <- c(failed, cell)
  }
}

cat(strrep("=", 70), "\n")
cat(sprintf("  完成 %d，跳过 %d，失败 %d，总耗时 %.1f 分钟\n",
            length(ok), length(skipped), length(failed),
            as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
if (length(failed)) cat(sprintf("  失败的格子：%s\n", paste(failed, collapse = ", ")))
cat(strrep("=", 70), "\n")

# 汇总现有的全量结果
metas <- list.files("output/oof", "^meta_[^g]", full.names = TRUE)
if (length(metas)) {
  tb <- rbindlist(lapply(metas, function(f) {
    m <- readRDS(f)
    data.table(model = m$model, auc = m$cv_mean, sd = m$cv_sd, minutes = m$minutes)
  }))[order(-auc)]
  cat("\n全量结果（当前全部）：\n")
  for (i in seq_len(nrow(tb)))
    cat(sprintf("  %-22s %.5f ± %.5f  (%.1f 分钟)\n",
                tb$model[i], tb$auc[i], tb$sd[i], tb$minutes[i]))
}
saveRDS(log, "output/grid_full_log.rds")
