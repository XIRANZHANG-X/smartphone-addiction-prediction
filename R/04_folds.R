# =============================================================================
# 04_folds.R —— 折叠契约 ★
#
# 用法：source("R/04_folds.R")
# 产出：output/folds.rds            长度 691369 的折号向量（1..5）
#       output/subsample_200k.rds   Tier A 对比实验用的 20 万行索引
#
# =============================================================================
#  这是全项目最重要的一个文件。
#
#  它决定了每一行属于哪一折。所有人的 AUC 之所以可以互相比较，
#  唯一的原因就是大家用的是同一份折叠。
#
#  ★ 生成之后任何人不得重新生成。★
#
#  重跑一次会发生什么：所有人之前跑出来的 OOF 预测都会与新折叠错位，
#  分数不再可比，集成的二层模型会学到错误的模式，
#  全组几天的实验结果全部作废，而且不会报错——你只会看到分数莫名其妙地变差。
#
#  为了防止手滑，脚本检测到产物已存在时会直接拒绝执行。
#  真的需要重建（比如项目初期还没人开始跑），
#  必须显式设置 I_KNOW_WHAT_IM_DOING <- TRUE。
# =============================================================================

library(data.table)

SEED   <- 20260821L   # 项目起始日期，无特殊含义，固定即可
N_FOLD <- 5L
N_SUB  <- 200000L     # Tier A 子样本规模，见 docs/项目说明.md 常见问题

dir_out  <- "output"
f_folds  <- file.path(dir_out, "folds.rds")
f_sub    <- file.path(dir_out, "subsample_200k.rds")

# ---- 冻结保护 ---------------------------------------------------------------
if (!exists("I_KNOW_WHAT_IM_DOING")) I_KNOW_WHAT_IM_DOING <- FALSE

if (file.exists(f_folds) && !I_KNOW_WHAT_IM_DOING) {
  cat("========================================\n")
  cat("  folds.rds 已存在，拒绝重新生成\n")
  cat("========================================\n")
  folds <- readRDS(f_folds)
  cat(sprintf("现有折叠：%s 行，%d 折\n",
              format(length(folds), big.mark = ","), length(unique(folds))))
  cat("\n这个文件是全组分数可比的唯一基础，重新生成会让所有人的实验结果作废。\n")
  cat("直接用现有的即可：folds <- readRDS(\"output/folds.rds\")\n")
} else {

  # ---- 读入 -----------------------------------------------------------------
  f_train <- file.path(dir_out, "raw_train.rds")
  if (!file.exists(f_train)) {
    stop("找不到 output/raw_train.rds，请先运行 source(\"R/01_load.R\")")
  }
  train <- readRDS(f_train)
  y <- train$addicted_label
  n <- length(y)

  stopifnot("训练集行数不是 691369" = n == 691369L)

  # ---- 分层 5 折 ------------------------------------------------------------
  # 分层的目的：让每一折的正例率都接近全局的 0.7094。
  # 如果不分层，随机划分在 69 万行上其实也够稳，但分层是零成本的保险，
  # 而且能让跨折标准差更小，方法之间的差异更容易看出来。
  #
  # RNGkind 显式指定：R 3.6.0 改过 sample() 的默认算法，
  # 写死它才能保证不同机器、不同 R 版本上生成完全一样的折叠。
  RNGkind(sample.kind = "Rejection")
  set.seed(SEED)

  folds <- integer(n)
  for (cls in c(0L, 1L)) {
    idx <- which(y == cls)
    idx <- sample(idx)                      # 打乱
    # 轮流发牌：保证每折拿到的该类样本数最多相差 1
    folds[idx] <- rep_len(seq_len(N_FOLD), length(idx))
  }

  stopifnot(
    "有行没有分配到折" = all(folds %in% seq_len(N_FOLD)),
    "折号长度不对"     = length(folds) == n
  )

  # ---- Tier A 子样本 --------------------------------------------------------
  # 关键：子样本继承原折标签，不重新划分。
  #
  # 如果这里重新划分，Tier A（20 万行对比实验）和 Tier B（全量集成候选）
  # 的分数就不可比了，跨层级的任何结论都会失效。
  # 从每个 (折, 类别) 单元里按比例抽，保证子样本的折分布和类别分布
  # 都和全量一致。
  set.seed(SEED + 1L)

  sub_idx <- integer(0)
  for (k in seq_len(N_FOLD)) {
    for (cls in c(0L, 1L)) {
      cell <- which(folds == k & y == cls)
      # as.numeric 不能省：N_SUB 和 length(cell) 都是整数，
      # 200000L * 98000L 会超出 32 位上限直接变成 NA。
      n_take <- round(as.numeric(N_SUB) * length(cell) / n)
      sub_idx <- c(sub_idx, sample(cell, n_take))
    }
  }
  sub_idx <- sort(sub_idx)

  # ---- 存盘 -----------------------------------------------------------------
  saveRDS(folds,   f_folds)
  saveRDS(sub_idx, f_sub)

  # ---- 验证报告 -------------------------------------------------------------
  cat("========================================\n")
  cat("  折叠契约已生成并冻结\n")
  cat("========================================\n\n")

  cat("---- 全量 5 折 ----\n")
  cat(sprintf("总行数 %s，seed = %d\n\n", format(n, big.mark = ","), SEED))
  cat(sprintf("  %-6s %10s %10s\n", "折", "行数", "正例率"))
  for (k in seq_len(N_FOLD)) {
    m <- folds == k
    cat(sprintf("  %-6d %10s %10.4f\n",
                k, format(sum(m), big.mark = ","), mean(y[m])))
  }
  cat(sprintf("  %-6s %10s %10.4f\n", "全体", format(n, big.mark = ","), mean(y)))

  spread <- diff(range(vapply(seq_len(N_FOLD), function(k) mean(y[folds == k]), 0)))
  cat(sprintf("\n跨折正例率极差 %.5f", spread))
  cat(if (spread < 0.002) "  （分层生效）\n" else "  （偏大，请检查）\n")

  cat("\n---- Tier A 子样本 ----\n")
  cat(sprintf("行数 %s（目标 %s）\n",
              format(length(sub_idx), big.mark = ","),
              format(N_SUB, big.mark = ",")))
  cat(sprintf("正例率 %.4f（全量 %.4f）\n", mean(y[sub_idx]), mean(y)))
  cat("\n  折分布：\n")
  for (k in seq_len(N_FOLD)) {
    cat(sprintf("    第 %d 折  %8s 行\n",
                k, format(sum(folds[sub_idx] == k), big.mark = ",")))
  }

  cat("\n----------------------------------------\n")
  cat("这两个文件已进 git。请不要再运行本脚本。\n")
  cat("用法：\n")
  cat('  folds   <- readRDS("output/folds.rds")\n')
  cat('  sub_idx <- readRDS("output/subsample_200k.rds")\n')
}
