# =============================================================================
# 28_ladder_pairs.R —— 阶梯里发生名次交换的那几对，本来分得开吗
#
# 用法：Rscript R/28_ladder_pairs.R
# 依赖：R/23_resolution.R（分辨率下限工具）、全量 OOF
# 产出：output/ladder_pairs.rds
#
# -----------------------------------------------------------------------------
# 为什么要单独算这个
# -----------------------------------------------------------------------------
# R/25_size_ladder.R 发现小样本与全量的排名不是 100% 一致（Spearman 0.90~0.96）。
# 但"排名不一致"有两种截然不同的含义：
#
#   (a) 小样本把一对**本来分得开**的模型排错了 —— 这是子样本策略的缺陷；
#   (b) 交换发生在一对**本来就分不开**的模型上 —— 这不是缺陷，
#       是这一对在任何样本量下都没有确定的先后。
#
# 区分这两者需要每一对自己的分辨率下限，而**下限是那一对的性质**：
# 实测同库近乎孪生的一对与跨模型族的一对，下限相差九倍
# （见 docs/讨论区核查.md 第十二节）。所以不能拿别的对的下限来套。
#
# 本脚本对阶梯中发生过名次交换的每一对，实测它们的分辨率下限，
# 再与观测到的全量差值比较。
# =============================================================================

suppressMessages({library(data.table)})
source("R/23_resolution.R")   # 提供 resolvable() 与 fast_auc()

y  <- readRDS("output/raw_train.rds")$addicted_label

lad <- readRDS("output/size_ladder.rds")
w   <- lad$wide
rk  <- lad$ranks
cols <- intersect(c("50k", "100k", "200k", "400k", "full"), names(w))
sub_cols <- setdiff(cols, "full")

# -----------------------------------------------------------------------------
# 找出所有「在某一级的相对顺序与全量相反」的对
# -----------------------------------------------------------------------------
models <- w$model
full_r <- setNames(rk[["full"]], rk$model)

swaps <- list()
for (a in seq_along(models)) for (b in seq_along(models)) {
  if (a >= b) next
  ma <- models[a]; mb <- models[b]
  # 全量上谁在前
  ahead_full <- full_r[[ma]] < full_r[[mb]]
  flipped <- sub_cols[vapply(sub_cols, function(cc) {
    ra <- rk[[cc]][rk$model == ma]; rb <- rk[[cc]][rk$model == mb]
    (ra < rb) != ahead_full
  }, logical(1))]
  if (length(flipped))
    swaps[[length(swaps) + 1L]] <- list(a = ma, b = mb, rungs = flipped)
}

if (!length(swaps)) {
  cat("没有任何一对在任何一级发生名次交换。\n"); quit(save = "no")
}

cat(sprintf("共有 %d 对在至少一级上与全量顺序相反。\n\n", length(swaps)))

# -----------------------------------------------------------------------------
# 逐对实测分辨率下限
# -----------------------------------------------------------------------------
get_oof <- local({
  cache <- list()
  function(m) {
    if (is.null(cache[[m]])) cache[[m]] <<- readRDS(sprintf("output/oof/oof_%s.rds", m))
    cache[[m]]
  }
})

rows <- list()
for (s in swaps) {
  if (!all(file.exists(sprintf("output/oof/oof_%s.rds", c(s$a, s$b))))) {
    cat(sprintf("  %s vs %s —— 缺全量 OOF，跳过\n", s$a, s$b)); next
  }
  r <- resolvable(get_oof(s$a), get_oof(s$b), y)
  rows[[length(rows) + 1L]] <- data.table(
    a = s$a, b = s$b, rungs = paste(s$rungs, collapse = ","),
    gap_full = abs(r$observed_gap), floor95 = r$resolvable95,
    rho = r$rho,
    verdict = if (abs(r$observed_gap) < r$resolvable95) "本来就分不开" else "本来分得开")
  cat(sprintf("  %-14s vs %-14s  交换于 %-14s  全量差 %.5f  下限 %.5f  → %s\n",
              s$a, s$b, paste(s$rungs, collapse = ","),
              abs(r$observed_gap), r$resolvable95, rows[[length(rows)]]$verdict))
}

res <- rbindlist(rows)
cat("\n汇总：\n")
n_unres <- sum(res$verdict == "本来就分不开")
cat(sprintf("  %d 对中有 %d 对的全量差值低于自身的分辨率下限 —— 这些交换不是小样本的错，\n",
            nrow(res), n_unres))
cat("  是这几对在我们拥有的任何样本量下都没有确定的先后。\n")
if (n_unres < nrow(res)) {
  cat("\n  以下几对是**真的排错了**（全量差值高于下限）：\n")
  for (i in which(res$verdict != "本来就分不开"))
    cat(sprintf("    %s vs %s（交换于 %s，差 %.5f > 下限 %.5f）\n",
                res$a[i], res$b[i], res$rungs[i], res$gap_full[i], res$floor95[i]))
}

saveRDS(res, "output/ladder_pairs.rds")
cat("\n已保存 output/ladder_pairs.rds\n")
