# =============================================================================
# 23_resolution.R —— 分辨率下限：一个差值要多大才算真的
#
# 用法：
#   Rscript R/23_resolution.R                        # 对几对现有提交算一遍
#   source("R/23_resolution.R"); resolvable(p1, p2, y)   # 对你自己的一对算
#
# -----------------------------------------------------------------------------
# 为什么需要它
# -----------------------------------------------------------------------------
# 我们一路上反复遇到「这个 +0.0002 算不算真的」这个问题，而此前只能靠
# 五折同号数和 Cohen's d 来回答。讨论区第 41 帖给了一个更直接的答案，
# 而且指出了大多数人算错的地方：
#
#   **没有人是拿自己的分数去和真值比的，所有人都在同一批行上被评分。**
#   共同的扰动在**差值**里会抵消，在各自的分数里不会。
#   所以用单个 AUC 的标准误（Hanley–McNeil）去判断两个模型的差异，
#   会悲观好几倍——第 22 帖测出配对 sigma 比边际 sigma 小六到七倍。
#
# 公式（第 41 帖 Georgy Mamarin 与 Dariush Afshar 共同确认，
#       后者在一对极端相似的提交上验证到 0.04% 以内）：
#
#     sd(gap) = sd(move) * sqrt(2 * (1 - rho))
#     95% 可分辨差异 = 1.96 * sd(gap)
#
# 其中 rho 是**两个 AUC 估计**的相关性，**不是两个预测向量**的相关性。
# 后者会让答案偏乐观——在他们的数据上，预测相关 0.994 对应 AUC 估计相关 0.973，
# 代入后的误差不小且方向是乐观的。
#
# -----------------------------------------------------------------------------
# 这个工具真正的用处：它没有单一答案
# -----------------------------------------------------------------------------
# 「多大的差异算显著」取决于**你在比较哪两个东西**。在同一个 74 成员库里，
# 这个阈值从 0.000005（最好单模型 vs 第二好）到 0.00043（最好 vs 第 20 好），
# 相差七十倍。
#
#   => 所以「我们那个 +0.00019 不显著」这句话，要先问清楚是**对谁**不显著。
#      两个只差十四列的配置属于近乎孪生，其分辨率下限远低于我们与陌生人之间的。
# =============================================================================

suppressMessages(library(data.table))

fast_auc <- function(y, p) {
  ok <- !is.na(p); y <- y[ok]; p <- p[ok]
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

#' 一对预测之间，多大的 AUC 差值才是可分辨的
#'
#' @param p1,p2 两个预测向量（同长度、同顺序）
#' @param y     标签
#' @param n_eval 评分时的行数。默认用测试集规模 296,302；
#'   传 59260 得到公榜规模（20%）、237042 得到私榜规模（80%）。
#' @param B     自助次数
#' @return list(sd_move, rho, sd_gap, resolvable95, observed_gap, verdict)
#'
#' 做法：自助重采样 B 次，每次同时给两个预测打分，得到 B 对 AUC。
#' 由这 B 对算出各自的 sd 与**它们之间的相关性**，再代入公式。
resolvable <- function(p1, p2, y, n_eval = 296302L, B = 400L, seed = 20260821L) {
  stopifnot(length(p1) == length(p2), length(p1) == length(y))
  set.seed(seed)
  n <- length(y)
  A <- matrix(NA_real_, B, 2L)
  for (b in seq_len(B)) {
    i <- sample.int(n, n_eval, replace = TRUE)
    A[b, 1] <- fast_auc(y[i], p1[i])
    A[b, 2] <- fast_auc(y[i], p2[i])
  }
  sd_move <- mean(apply(A, 2, sd))
  rho     <- stats::cor(A[, 1], A[, 2])
  sd_gap_formula  <- sd_move * sqrt(2 * (1 - rho))
  sd_gap_measured <- sd(A[, 1] - A[, 2])
  res95   <- 1.96 * sd_gap_measured
  gap     <- fast_auc(y, p1) - fast_auc(y, p2)
  list(sd_move = sd_move, rho = rho,
       sd_gap_formula = sd_gap_formula, sd_gap_measured = sd_gap_measured,
       resolvable95 = res95, observed_gap = gap,
       verdict = if (abs(gap) > res95) "可分辨" else "低于下限，无法分辨")
}

report <- function(lab, r) {
  cat(sprintf("%-34s 观测差 %+.6f   下限 %.6f   %s\n",
              lab, r$observed_gap, r$resolvable95, r$verdict))
  cat(sprintf("%-34s   （rho = %.5f，公式 %.3e 对实测 %.3e，误差 %+.1f%%）\n", "",
              r$rho, r$sd_gap_formula, r$sd_gap_measured,
              100 * (r$sd_gap_formula / r$sd_gap_measured - 1)))
}

# -----------------------------------------------------------------------------
# 直接运行时：对手边几对预测算一遍，展示「下限随配对而变」
# -----------------------------------------------------------------------------
if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  if (!file.exists("output/oof")) {
    cat("找不到 output/oof/，跳过示例。\n")
  } else {
    y  <- readRDS("output/raw_train.rds")$addicted_label
    fs <- list.files("output/oof", pattern = "^oof_[^g].*\\.rds$", full.names = TRUE)
    nm <- sub("^oof_", "", sub("\\.rds$", "", basename(fs)))
    P  <- lapply(fs, readRDS); names(P) <- nm
    aucs <- sort(sapply(P, function(p) fast_auc(y, p)), decreasing = TRUE)

    cat("单模型 AUC（Tier B，加入编码之前的口径）：\n")
    for (i in seq_along(aucs)) cat(sprintf("  %-20s %.5f\n", names(aucs)[i], aucs[i]))

    cat("\n分辨率下限（按测试集规模 296,302 行，400 次自助）：\n\n")
    pairs <- list(
      c(names(aucs)[1], names(aucs)[2]),   # 最好 vs 第二好：近乎孪生
      c(names(aucs)[1], names(aucs)[4]),
      c(names(aucs)[1], names(aucs)[length(aucs)])  # 最好 vs 最差：完全不同的族
    )
    for (pr in pairs) report(sprintf("%s vs %s", pr[1], pr[2]),
                             resolvable(P[[pr[1]]], P[[pr[2]]], y))

    cat("\n=> 同一份数据、同一个库，下限相差一个数量级以上。\n")
    cat("   「多大算显著」没有单一答案，它是那**一对**的性质。\n")
  }
}
