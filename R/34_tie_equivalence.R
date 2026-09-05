# =============================================================================
# 34_tie_equivalence.R —— 表 11「平局」判决的正式等价性论证
#
# 用法：Rscript R/34_tie_equivalence.R
# 依赖：R/23_resolution.R（fast_auc()、resolvable()，本脚本不修改该文件）、全量 OOF
# 产出：output/tie_equivalence.rds + 屏幕报告
#
# -----------------------------------------------------------------------------
# 背景（审稿意见，第 4 项任务；依赖第 2 项任务里统一好的分辨率下限定义）
# -----------------------------------------------------------------------------
# R/28_ladder_pairs.R 对 L2_xgboost vs L1_lightgbm 这一对给出的判决是
# 「本来就分不开」：全量差值 0.00009 低于该对自己的分辨率下限 0.00011。
# 审稿人指出：「不能拒绝差别的存在」不等于「没有差别」——单纯的
# 「观测差 < 下限」比较，本身不构成一个正式的等价性论证。
#
# 本脚本补一个标准等价性检验里最简单的形式：CI-inclusion 法。
#
# `resolvable()`（R/23_resolution.R）内部本来就会对每次自助重采样算出一对
# AUC，攒成一个 B×2 矩阵 A，只是算完 sd(A[,1] - A[,2]) 就把 A 丢掉了、
# 只返回这个标量。A[,1] - A[,2] 这 400 个数，就是「全量配对差值」的经验
# 自助分布——百分位置信区间要的正是这个。
#
# 这里选择**不修改** resolvable()：它同时喂 Table 2 与 Table 11 其余
# 6 对的下限，不想为了这一对而牵动其他数字。改为在本文件里用完全相同的
# seed / B / n_eval，配合 source() 来的 fast_auc()，独立重建同一个 B×2
# 矩阵（约十行，不重新发明任何机制）。
#
# 步骤：
#   (1) 原样调用一次 resolvable()，核对 Table 11 现有的 0.00009 / 0.00011
#       两个数字能否复现；
#   (2) 本地重建同一个自助循环（同 seed=20260821L, B=400L,
#       n_eval=296302L），核对本地算出的观测差、下限与 (1) 严格相等——
#       确认这是「同一批」400 次重采样，不是另一次只是近似可比的重采样；
#   (3) 对本地重建的差值向量取 2.5% / 97.5% 分位数，得到全量差值的
#       95% 百分位自助置信区间；
#   (4) 检查这个区间是否整个落在等价界 [-floor95, +floor95] 以内
#       （floor95 即 Task 2 统一定义后的「分辨率下限」，
#       1.96 × 自助实测 SD(差值)）；顺带报告 400 次重采样里差值本身落在
#       等价界以内的比例，作为辅助描述。
# =============================================================================

suppressMessages({library(data.table)})
source("R/23_resolution.R")   # 提供 fast_auc()、resolvable()；不修改此文件

y  <- readRDS("output/raw_train.rds")$addicted_label
p1 <- readRDS("output/oof/oof_L2_xgboost.rds")    # a
p2 <- readRDS("output/oof/oof_L1_lightgbm.rds")   # b

stopifnot(length(p1) == length(y), length(p2) == length(y))

# -----------------------------------------------------------------------------
# (1) 原样调用 resolvable()，核对 Table 11 现有数字能否复现
# -----------------------------------------------------------------------------
r <- resolvable(p1, p2, y)   # 全部用默认值：n_eval = 296302L, B = 400L, seed = 20260821L

cat(sprintf("[核对 1] resolvable() 输出：观测差 %.8f   下限(1.96*实测SD) %.8f\n",
            r$observed_gap, r$resolvable95))
cat("[核对 1] Table 11 现有数字：观测差 0.00009   下限 0.00011\n")
stopifnot(abs(round(r$observed_gap, 5) - 0.00009) < 1e-5)
stopifnot(abs(round(r$resolvable95, 5) - 0.00011) < 1e-5)
cat("[核对 1 通过] 0.00009 / 0.00011 复现。\n\n")

# -----------------------------------------------------------------------------
# (2) 本地独立重建同一个 B×2 自助矩阵，拿到 resolvable() 算完即丢弃的
#     那 400 对 AUC，供百分位置信区间使用
# -----------------------------------------------------------------------------
n_eval <- 296302L; B <- 400L; seed <- 20260821L
set.seed(seed)
n <- length(y)
A <- matrix(NA_real_, B, 2L)
for (b in seq_len(B)) {
  i <- sample.int(n, n_eval, replace = TRUE)
  A[b, 1] <- fast_auc(y[i], p1[i])
  A[b, 2] <- fast_auc(y[i], p2[i])
}
diff <- A[, 1] - A[, 2]   # 400 个自助配对差值：L2_xgboost − L1_lightgbm

local_gap   <- fast_auc(y, p1) - fast_auc(y, p2)
local_floor <- 1.96 * sd(diff)
cat(sprintf("[核对 2] 本地重建：观测差 %.8f   下限 %.8f\n", local_gap, local_floor))
stopifnot(isTRUE(all.equal(local_gap, r$observed_gap)))
stopifnot(isTRUE(all.equal(local_floor, r$resolvable95)))
cat("[核对 2 通过] 本地重建与 resolvable() 内部计算严格一致（同一批 400 次重采样）。\n\n")

# -----------------------------------------------------------------------------
# (3)(4) 正式的等价性论证：CI-inclusion 法
# -----------------------------------------------------------------------------
floor95 <- r$resolvable95
ci <- quantile(diff, c(0.025, 0.975))
inside <- ci[[1]] >= -floor95 && ci[[2]] <= floor95
frac_in_margin <- mean(diff >= -floor95 & diff <= floor95)

cat(sprintf("等价界（该对自己的分辨率下限）：[%.6f, %.6f]\n", -floor95, floor95))
cat(sprintf("全量差值的 95%% 百分位自助置信区间：[%.6f, %.6f]\n", ci[[1]], ci[[2]]))
cat(sprintf("该区间是否整个落在等价界以内：%s\n", if (inside) "是" else "否"))
cat(sprintf("400 次自助重采样里，差值本身落在等价界以内的比例：%.1f%%\n",
            100 * frac_in_margin))

verdict <- if (inside) {
  "在该对自己的等价界以内不可分辨（正式等价性论证成立）"
} else {
  "置信区间未完全落在等价界以内——不能把「平局」当等价性论证，需人工复核措辞"
}
cat(sprintf("\n结论：%s\n", verdict))

out <- list(
  pair = c(a = "L2_xgboost", b = "L1_lightgbm"),
  n_eval = n_eval, B = B, seed = seed,
  observed_gap = r$observed_gap, floor95 = floor95,
  ci_lower = unname(ci[[1]]), ci_upper = unname(ci[[2]]),
  inside_margin = inside, frac_resamples_in_margin = frac_in_margin,
  diff = diff, verdict = verdict
)
saveRDS(out, "output/tie_equivalence.rds")
cat("\n已保存 output/tie_equivalence.rds\n")
