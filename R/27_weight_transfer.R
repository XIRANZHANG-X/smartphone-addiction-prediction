# =============================================================================
# 27_weight_transfer.R —— 集成权重能不能从 20 万迁移到全量
#
# 用法：Rscript R/27_weight_transfer.R
# 依赖：先跑完 R/25_size_ladder.R（要 200k 那一级）和 R/run_grid_full.R
# 产出：output/weight_transfer.rds
#
# -----------------------------------------------------------------------------
# 这是样本量迁移实验的第二问
# -----------------------------------------------------------------------------
# R/25_size_ladder.R 问的是「小样本选出的**模型**是不是全量的最优模型」。
# 但交付用的不是单模型而是集成，集成还有一组要学的权重。所以还要问：
#
#   **用 20 万上学到的集成权重，去融合全量的成员预测，比直接在全量上
#     学权重差多少？**
#
# 如果差得可以忽略，那么「小样本筛选 + 全量交付」这条流水线就是完整成立的：
# 不只是选模型可以在小样本上做，配权重也可以。
#
# -----------------------------------------------------------------------------
# 口径与已知偏差
# -----------------------------------------------------------------------------
# 两组权重都用同一个融合器（秩空间二层 logistic，见 07_ensemble.R 方法一之二）：
#
#   w_A  在 20 万那一级的 OOF 上拟合（成员模型也是在 20 万上训练的）
#   w_B  在全量 OOF 上拟合（成员模型在全量上训练）
#
# 评估时**两者都作用在同一份全量成员预测上**，比较 AUC。
#
# ⚠ 这个比较对 w_B 有利：它是在同一批行上拟合又在同一批行上评估的，
#   而 w_A 对这批行完全没见过。所以报出来的差距是**代价的上界**，
#   真实代价只会更小。补一个对照：只在 20 万池之外的 49 万行上评估，
#   那里 w_A 是彻底的样本外，w_B 仍是样本内——方向不变，仍是上界。
#   这样解释起来不需要额外假设：如果连上界都可以忽略，结论就成立。
# =============================================================================

suppressMessages({library(data.table)})

CELLS <- c("L1_xgboost", "L1_lightgbm",
           "L2_xgboost", "L2_lightgbm", "L2_ranger", "L2_glmnet",
           "L3_xgboost", "L3_lightgbm", "L3_ranger", "L3_glmnet")

#' 秩变换：把每一列换成它自己的秩再归一到 (0,1]
to_rank <- function(M) apply(M, 2, function(v) rank(v, ties.method = "average") / length(v))

#' 快速 AUC（Mann-Whitney）。n1*n0 会超 32 位，必须先转 numeric。
fast_auc <- function(y, p) {
  r  <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# -----------------------------------------------------------------------------
# 载入
# -----------------------------------------------------------------------------
y_all <- readRDS("output/raw_train.rds")$addicted_label
pool  <- readRDS("output/pools/pool_200k.rds")

have_full <- CELLS[file.exists(sprintf("output/oof/oof_%s.rds", CELLS))]
have_sub  <- CELLS[file.exists(sprintf("output/ladder/oof_pool_200k_%s.rds", CELLS))]
use <- intersect(have_full, have_sub)
if (length(use) < 3L)
  stop("两边共有的成员不足 3 个：全量 ", length(have_full), " 个，20 万 ", length(have_sub), " 个")

cat(sprintf("共有成员 %d 个：%s\n\n", length(use), paste(use, collapse = ", ")))

M_full <- do.call(cbind, lapply(use, function(m) readRDS(sprintf("output/oof/oof_%s.rds", m))))
M_sub  <- do.call(cbind, lapply(use, function(m) readRDS(sprintf("output/ladder/oof_pool_200k_%s.rds", m))))
colnames(M_full) <- colnames(M_sub) <- use

stopifnot("全量 OOF 行数不对" = nrow(M_full) == length(y_all),
          "20 万 OOF 行数不对" = nrow(M_sub)  == length(pool))

# -----------------------------------------------------------------------------
# 拟合两组权重
# -----------------------------------------------------------------------------
fit_w <- function(M, y) {
  df <- as.data.frame(to_rank(M)); df$.y <- y
  coef(stats::glm(.y ~ ., data = df, family = stats::binomial()))
}

cat("拟合权重 ... "); flush.console()
w_A <- fit_w(M_sub,  y_all[pool])     # 20 万
w_B <- fit_w(M_full, y_all)           # 全量
cat("完成\n\n")

# -----------------------------------------------------------------------------
# 都作用在同一份全量成员预测上
# -----------------------------------------------------------------------------
R_full <- to_rank(M_full)
Xd <- cbind(1, R_full)                                 # 加截距列
colnames(Xd) <- c("(Intercept)", use)
score <- function(w) as.numeric(Xd %*% w[colnames(Xd)])

p_A <- score(w_A); p_B <- score(w_B)
out_idx <- setdiff(seq_len(nrow(M_full)), pool)        # 20 万池之外的 49 万行

res <- data.table(
  评估集    = c("全量 691,369 行", sprintf("池外 %s 行", format(length(out_idx), big.mark = ","))),
  n         = c(nrow(M_full), length(out_idx)),
  AUC_20万权重 = c(fast_auc(y_all, p_A),          fast_auc(y_all[out_idx], p_A[out_idx])),
  AUC_全量权重 = c(fast_auc(y_all, p_B),          fast_auc(y_all[out_idx], p_B[out_idx]))
)
res[, 迁移代价上界 := AUC_全量权重 - AUC_20万权重]

cat("集成 AUC（同一份全量成员预测，只换权重）：\n")
cat(strrep("-", 76), "\n")
for (i in seq_len(nrow(res)))
  cat(sprintf("  %-22s  20万权重 %.5f   全量权重 %.5f   代价上界 %+.5f\n",
              res$评估集[i], res$AUC_20万权重[i], res$AUC_全量权重[i], res$迁移代价上界[i]))

# -----------------------------------------------------------------------------
# 权重本身像不像
# -----------------------------------------------------------------------------
wt <- data.table(成员 = names(w_A), w_20万 = as.numeric(w_A), w_全量 = as.numeric(w_B))
wt[, 差 := w_全量 - w_20万]
rho <- cor(wt$w_20万[-1], wt$w_全量[-1])              # 去掉截距
sp  <- cor(wt$w_20万[-1], wt$w_全量[-1], method = "spearman")

cat(sprintf("\n权重向量（去掉截距后）Pearson %.3f，Spearman %.3f\n\n", rho, sp))
cat(sprintf("  %-16s %10s %10s %10s\n", "成员", "20万", "全量", "差"))
for (i in seq_len(nrow(wt)))
  cat(sprintf("  %-16s %10.3f %10.3f %+10.3f\n",
              wt$成员[i], wt$w_20万[i], wt$w_全量[i], wt$差[i]))

# 符号一致性：哪些成员在两边都是正贡献 / 都是负贡献
same_sign <- sum(sign(wt$w_20万[-1]) == sign(wt$w_全量[-1]))
cat(sprintf("\n符号一致的成员：%d / %d\n", same_sign, nrow(wt) - 1L))

saveRDS(list(res = res, weights = wt, pearson = rho, spearman = sp,
             same_sign = same_sign, members = use),
        "output/weight_transfer.rds")
cat("\n已保存 output/weight_transfer.rds\n")
