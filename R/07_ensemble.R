# =============================================================================
# 07_ensemble.R —— 集成
#
# 用法：Rscript R/07_ensemble.R
# 产出：output/ensemble_best.rds
#
# -----------------------------------------------------------------------------
# 这个脚本不需要知道有哪些模型。
#
# 它扫描 output/oof/，把所有 Tier B 的 oof_*.rds 当作二层特征。
# 加模型不需要改这里一个字 —— 组员跑完 push 上来，重跑一次就自动纳入。
#
# 只吃 Tier B（全量 69 万行）。Tier A 的 oof_grid_*.rds 会被跳过：
# 用 20 万行的 OOF 训二层模型再套到全量上，尺度和分布都对不上。
# -----------------------------------------------------------------------------
# 横向对比三种集成方式（这本身是报告里的一个对比点）：
#   1. 二层 logistic 回归 —— 让模型自己学权重，可以是负的
#   2. rank 平均         —— 只用排序信息，对尺度差异免疫
#   3. 爬山法搜权重       —— 贪心地一个个加模型，允许重复选同一个来加权
# =============================================================================

suppressMessages({library(data.table); library(pROC)})

dir_oof  <- "output/oof"
dir_test <- "output/test"

# -----------------------------------------------------------------------------
# 快速 AUC
# -----------------------------------------------------------------------------
# 爬山法要在 55 万行上反复算 AUC（30 步 × N 个模型 × 5 折）。
# pROC::roc 每次约 1 秒，那样要跑十几分钟。
# 用秩公式直接算：AUC = (正例秩和 - n1(n1+1)/2) / (n1 * n0)，
# 只需一次排序，快两个数量级，结果与 pROC 完全一致。
fast_auc <- function(y, p) {
  r <- rank(p, ties.method = "average")
  # as.numeric 不能省：sum(y == 1) 返回**整型**，而全量上 n1 = 490,474、
  # n0 = 200,895，n1 * n0 ≈ 9.85e10 远超 32 位整型上限，
  # 整型乘法会直接产生 NA（只给一条 warning，很容易被淹没在日志里）。
  # 这和 04_folds.R 里踩过的是同一类坑。
  n1 <- as.numeric(sum(y == 1))
  n0 <- as.numeric(length(y)) - n1
  if (n1 == 0 || n0 == 0) return(NA_real_)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# -----------------------------------------------------------------------------
# 收集 Tier B 产物
# -----------------------------------------------------------------------------
oof_files <- list.files(dir_oof, pattern = "^oof_", full.names = FALSE)
oof_files <- oof_files[endsWith(oof_files, ".rds") &
                       !startsWith(oof_files, "oof_grid_")]

if (length(oof_files) == 0) {
  stop("output/oof/ 下没有找到任何 Tier B 的 oof_*.rds。\n",
       "请先跑 Tier B：TIER=B Rscript R/06_model_<名字>.R\n",
       "或者用 Rscript R/run_tierb.R 批量跑。")
}

model_names <- sub("^oof_", "", sub(".rds", "", oof_files, fixed = TRUE))

folds <- readRDS("output/folds.rds")
y     <- readRDS("output/raw_train.rds")$addicted_label
n     <- length(y)

cat(sprintf("找到 %d 个 Tier B 候选：\n", length(model_names)))

oof_list <- list(); test_list <- list()

for (i in seq_along(model_names)) {
  nm <- model_names[i]
  o  <- readRDS(file.path(dir_oof, oof_files[i]))
  tf <- file.path(dir_test, sprintf("test_%s.rds", nm))

  # 逐个校验形状。形状不对的直接跳过并报警，不让它污染集成。
  if (length(o) != n) {
    cat(sprintf("  [跳过] %-22s OOF 长度 %s，应为 %s\n",
                nm, format(length(o), big.mark = ","),
                format(n, big.mark = ","))); next
  }
  if (!file.exists(tf)) {
    cat(sprintf("  [跳过] %-22s 缺少 test_%s.rds\n", nm, nm)); next
  }
  if (anyNA(o)) {
    cat(sprintf("  [跳过] %-22s OOF 里有 NA\n", nm)); next
  }
  tp <- readRDS(tf)
  if (anyNA(tp)) {
    cat(sprintf("  [跳过] %-22s test 预测里有 NA\n", nm)); next
  }

  cat(sprintf("  [采用] %-22s OOF AUC %.5f\n", nm, fast_auc(y, o)))
  oof_list[[nm]] <- o
  test_list[[nm]] <- tp
}

model_names <- names(oof_list)
if (length(model_names) == 0) stop("没有任何模型通过校验。")

OOF  <- do.call(cbind, oof_list)
TEST <- do.call(cbind, test_list)
colnames(OOF) <- colnames(TEST) <- model_names

cat(sprintf("\n集成矩阵：%s × %d\n", format(nrow(OOF), big.mark = ","), ncol(OOF)))

# 模型之间的相关性 —— 相关性越低，集成的收益越大
if (ncol(OOF) > 1) {
  cr <- cor(apply(OOF, 2, rank))
  cat(sprintf("模型间秩相关：最低 %.4f，中位 %.4f，最高 %.4f\n",
              min(cr[upper.tri(cr)]), median(cr[upper.tri(cr)]),
              max(cr[upper.tri(cr)])))
}

# -----------------------------------------------------------------------------
# 二层模型也必须交叉验证
# -----------------------------------------------------------------------------
# 否则报出来的分数是在训练它自己的数据上算的，必然虚高。
# 用的是同一套 folds，和一层保持一致。
cv_blend <- function(blend_fn) {
  pred <- rep(NA_real_, n)
  for (k in sort(unique(folds))) {
    tr <- folds != k; va <- folds == k
    pred[va] <- blend_fn(OOF[tr, , drop = FALSE], y[tr],
                         OOF[va, , drop = FALSE])
  }
  pred
}

results <- list()

# ---- 方法一：二层 logistic ---------------------------------------------------
blend_logistic <- function(X_tr, y_tr, X_va) {
  df <- as.data.frame(X_tr); df$.y <- y_tr
  m  <- stats::glm(.y ~ ., data = df, family = stats::binomial())
  stats::predict(m, newdata = as.data.frame(X_va), type = "response")
}
cat("\n计算二层 logistic ... "); flush.console()
results$logistic <- cv_blend(blend_logistic)
cat(sprintf("CV AUC %.5f\n", fast_auc(y, results$logistic)))

# ---- 方法二：rank 平均 -------------------------------------------------------
# 无需训练，没有可拟合的参数，因此不存在泄漏，直接在全量上算即可。
rank_avg <- function(M) rowMeans(apply(M, 2, rank))
cat("计算 rank 平均 ... "); flush.console()
results$rank_avg <- rank_avg(OOF)
cat(sprintf("CV AUC %.5f\n", fast_auc(y, results$rank_avg)))

# ---- 方法三：爬山法 ---------------------------------------------------------
# 从空集合开始，每一步在所有候选里挑一个加进来（可以重复挑同一个，
# 等效于给它更大的权重），使当前平均预测的 AUC 提升最多。
blend_hillclimb <- function(X_tr, y_tr, X_va, n_step = 30L, tol = 1e-6) {
  R_tr <- apply(X_tr, 2, rank)   # 在秩空间做，免疫各模型输出尺度差异
  R_va <- apply(X_va, 2, rank)

  weights <- rep(0, ncol(X_tr))
  cur_sum <- rep(0, nrow(X_tr))
  best <- -Inf

  for (step in seq_len(n_step)) {
    gains <- vapply(seq_len(ncol(X_tr)), function(j) {
      fast_auc(y_tr, (cur_sum + R_tr[, j]) / step)
    }, numeric(1))
    j <- which.max(gains)
    if (gains[j] <= best + tol) break
    best <- gains[j]
    weights[j] <- weights[j] + 1
    cur_sum <- cur_sum + R_tr[, j]
  }

  if (sum(weights) == 0) return(rowMeans(R_va))
  attr_out <- as.numeric(R_va %*% weights) / sum(weights)
  attr(attr_out, "weights") <- weights
  attr_out
}
cat("计算爬山法 ... "); flush.console()
results$hillclimb <- cv_blend(blend_hillclimb)
cat(sprintf("CV AUC %.5f\n", fast_auc(y, results$hillclimb)))

# -----------------------------------------------------------------------------
# 选优
# -----------------------------------------------------------------------------
scores <- vapply(results, function(p) fast_auc(y, p), numeric(1))
best_name <- names(scores)[which.max(scores)]

single <- vapply(model_names, function(nm) fast_auc(y, OOF[, nm]), numeric(1))
single_best <- max(single)

cat("\n======================================================\n")
cat("  集成方式对比\n")
cat("======================================================\n")
for (nm in names(scores)) {
  cat(sprintf("  %-14s %.5f%s\n", nm, scores[[nm]],
              if (nm == best_name) "   <- 最优" else ""))
}
cat(sprintf("\n  最好的单模型   %.5f  (%s)\n",
            single_best, names(single)[which.max(single)]))
cat(sprintf("  集成带来的提升 %+.5f\n", max(scores) - single_best))

# -----------------------------------------------------------------------------
# 生成最终测试集预测
# -----------------------------------------------------------------------------
cat("\n生成 test 预测 ... "); flush.console()
final_test <- switch(
  best_name,
  logistic  = blend_logistic(OOF, y, TEST),
  rank_avg  = rank_avg(TEST),
  hillclimb = blend_hillclimb(OOF, y, TEST)
)
cat("完成\n")

saveRDS(list(
  method       = best_name,
  cv_auc       = max(scores),
  all_scores   = scores,
  single_aucs  = single,
  models       = model_names,
  test_pred    = as.numeric(final_test)
), "output/ensemble_best.rds")

cat(sprintf("\n已保存 output/ensemble_best.rds（方式 %s，CV %.5f）\n",
            best_name, max(scores)))
if (best_name != "logistic") {
  cat("注意：rank 平均和爬山法输出的是**秩**，不是概率。\n")
  cat("      08_submit.R 会把它线性压到 (0,1)，形式上像概率但不具备概率含义。\n")
  cat("      AUC 只看排序，这不影响成绩；但报告中谈校准时必须区分。\n")
}
cat("下一步：Rscript R/08_submit.R\n")
