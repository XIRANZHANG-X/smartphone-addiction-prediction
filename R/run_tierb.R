# =============================================================================
# run_tierb.R —— 把 Tier A 里排名靠前的候选在全量数据上重训
#
# 用法：Rscript R/run_tierb.R [TOP_N]
#   默认取 Tier A 里 AUC 最高的 5 个格子
#
# 产出：每个候选的 output/oof/oof_<名字>.rds（69 万）
#                  output/test/test_<名字>.rds（29.6 万）
#
# -----------------------------------------------------------------------------
# 为什么不是全部 14 个格子都跑全量
# -----------------------------------------------------------------------------
# 全量是 20 万子样本的 3.5 倍数据，ranger 和 L4 的 missRanger 在全量上
# 会慢到不可接受。而集成只需要**好且互补**的几个成员 —— 把明显更差的
# 格子塞进去只会拖累二层模型。
#
# 选择标准是 Tier A 的 CV AUC，这个选择用的是训练数据的交叉验证结果，
# 不涉及测试集，不构成泄漏。
# -----------------------------------------------------------------------------
# 关于「20 万上的结论在 69 万上还成立吗」
# -----------------------------------------------------------------------------
# 这是审查里提出的一个好问题。Tier B 跑完之后，把同一批格子在两个规模上
# 的 AUC 排序对比一下就能回答 —— 脚本最后会打印这个对照表。
# =============================================================================

suppressMessages(library(data.table))

args  <- commandArgs(trailingOnly = TRUE)
TOP_N <- if (length(args)) as.integer(args[1]) else 5L

# ---- 按 Tier A 成绩排序 -----------------------------------------------------
metas <- list.files("output/oof", pattern = "^meta_grid_", full.names = TRUE)
if (!length(metas)) stop("没有 Tier A 结果，请先跑 Rscript R/run_grid.R")

tbl <- rbindlist(lapply(metas, function(f) {
  m <- readRDS(f)
  data.table(model = m$model, tierA = m$cv_mean, sd = m$cv_sd)
}))[order(-tierA)]

cat("======================================================\n")
cat("  Tier A 排名\n")
cat("======================================================\n")
for (i in seq_len(nrow(tbl))) {
  cat(sprintf("  %2d. %-16s %.5f ± %.5f%s\n", i, tbl$model[i],
              tbl$tierA[i], tbl$sd[i], if (i <= TOP_N) "   <- 入选" else ""))
}

cand <- tbl$model[seq_len(min(TOP_N, nrow(tbl)))]

cat(sprintf("\n将在全量 691,369 行上重训 %d 个候选\n", length(cand)))
cat("======================================================\n\n")

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
t_all <- Sys.time()
ok <- character(0); failed <- character(0)

for (i in seq_along(cand)) {
  cell <- cand[i]
  f <- sprintf("R/06_model_%s.R", cell)

  if (file.exists(sprintf("output/test/test_%s.rds", cell)) &&
      !nzchar(Sys.getenv("FORCE"))) {
    cat(sprintf("[%d/%d] %-16s 已有 Tier B 产物，跳过\n", i, length(cand), cell))
    ok <- c(ok, cell); next
  }

  cat(sprintf("---- [%d/%d] %s（全量）----\n", i, length(cand), cell))
  t0 <- Sys.time()
  status <- run_with_env(rscript, f, c(TIER = "B"))
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (status == 0 && file.exists(sprintf("output/test/test_%s.rds", cell))) {
    ok <- c(ok, cell)
    cat(sprintf("  ✓ 完成，耗时 %.1f 分钟\n\n", mins))
  } else {
    failed <- c(failed, cell)
    cat(sprintf("  ✗ 失败（退出码 %s），耗时 %.1f 分钟\n\n", status, mins))
  }
}

# ---- Tier A vs Tier B 对照 --------------------------------------------------
cat("======================================================\n")
cat("  规模一致性检查：20 万 vs 69 万\n")
cat("======================================================\n")
cat(sprintf("%-16s %10s %10s %10s\n", "模型", "Tier A", "Tier B", "差值"))

rows <- list()
for (cell in ok) {
  fb <- sprintf("output/oof/meta_%s.rds", cell)
  if (!file.exists(fb)) next
  b <- readRDS(fb)$cv_mean
  a <- tbl[model == cell, tierA]
  rows[[cell]] <- data.table(model = cell, tierA = a, tierB = b, diff = b - a)
  cat(sprintf("%-16s %10.5f %10.5f %+10.5f\n", cell, a, b, b - a))
}

if (length(rows) > 1) {
  d <- rbindlist(rows)
  ra <- rank(-d$tierA); rb <- rank(-d$tierB)
  cat(sprintf("\n两个规模下的排序一致性（Spearman）：%.3f\n",
              suppressWarnings(cor(ra, rb, method = "spearman"))))
  cat(if (identical(order(ra), order(rb)))
        "排序完全一致 —— 20 万子样本上的结论可以外推到全量。\n"
      else
        "排序有变化 —— 报告中必须说明「对比实验在子样本上进行」这一限制。\n")
}

cat(sprintf("\n总耗时 %.1f 分钟，成功 %d，失败 %d\n",
            as.numeric(difftime(Sys.time(), t_all, units = "mins")),
            length(ok), length(failed)))
if (length(failed)) cat("失败：", paste(failed, collapse = ", "), "\n")
cat("\n下一步：Rscript R/07_ensemble.R\n")
