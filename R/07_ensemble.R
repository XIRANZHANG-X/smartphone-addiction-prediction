# =============================================================================
# 07_ensemble.R —— 集成
#
# 用法：source("R/07_ensemble.R")
# 产出：output/ensemble_best.rds   最终的测试集预测
#
# -----------------------------------------------------------------------------
# 这个脚本不需要知道有哪些模型。
#
# 它扫描 output/oof/ 目录，把所有 oof_*.rds 当作二层特征。
# 加模型不需要改这里一个字 —— 组员 push 上来，重跑一次就自动纳入。
#
# 只吃 Tier B（全量 69 万行）的产物。Tier A 的 oof_grid_*.rds 会被跳过：
# 用 20 万行的 OOF 去训二层模型再套到全量上，尺度和分布都对不上。
# -----------------------------------------------------------------------------
# 横向对比三种集成方式（这本身是报告里的一个对比点）：
#   1. 二层 logistic 回归 —— 让模型自己学权重，可以是负的
#   2. rank 平均         —— 只用排序信息，对尺度差异免疫，AUC 场景下常出奇效
#   3. 爬山法搜权重       —— 贪心地一个个加模型，允许重复选同一个来加权
# =============================================================================

library(data.table)
library(pROC)

dir_oof  <- "output/oof"
dir_test <- "output/test"

# ---- 收集 Tier B 产物 -------------------------------------------------------
oof_files <- list.files(dir_oof, pattern = "^oof_.*\\.rds$", full.names = FALSE)
oof_files <- oof_files[!grepl("^oof_grid_", oof_files)]   # 排除 Tier A

if (length(oof_files) == 0) {
  stop("output/oof/ 下没有找到任何 Tier B 的 oof_*.rds。\n",
       "请先让各条线用 TIER <- \"B\" 跑出全量结果。")
}

model_names <- sub("^oof_", "", sub("\\.rds$", "", oof_files))

folds <- readRDS("output/folds.rds")
train <- readRDS("output/raw_train.rds")
y     <- train$addicted_label
n     <- length(y)

cat(sprintf("找到 %d 个 Tier B 模型：\n", length(model_names)))

# ---- 组装矩阵 ---------------------------------------------------------------
keep <- logical(length(model_names))
oof_list  <- list()
test_list <- list()

for (i in seq_along(model_names)) {
  nm <- model_names[i]
  o  <- readRDS(file.path(dir_oof, oof_files[i]))
  tf <- file.path(dir_test, sprintf("test_%s.rds", nm))

  # 逐个校验形状。形状不对的直接跳过并报警，不要让它污染集成。
  if (length(o) != n) {
    cat(sprintf("  [跳过] %-24s OOF 长度 %s，应为 %s\n",
                nm, format(length(o), big.mark = ","), format(n, big.mark = ",")))
    next
  }
  if (!file.exists(tf)) {
    cat(sprintf("  [跳过] %-24s 缺少 test_%s.rds\n", nm, nm))
    next
  }
  if (anyNA(o)) {
    cat(sprintf("  [跳过] %-24s OOF 里有 NA\n", nm))
    next
  }

  a <- as.numeric(pROC::auc(pROC::roc(y, o, quiet = TRUE)))
  cat(sprintf("  [采用] %-24s OOF AUC %.5f\n", nm, a))

  keep[i]        <- TRUE
  oof_list[[nm]]  <- o
  test_list[[nm]] <- readRDS(tf)
}

model_names <- names(oof_list)
if (length(model_names) == 0) stop("没有任何模型通过校验。")

OOF  <- do.call(cbind, oof_list)
TEST <- do.call(cbind, test_list)
colnames(OOF) <- colnames(TEST) <- model_names

cat(sprintf("\n集成矩阵：%s × %d\n", format(nrow(OOF), big.mark = ","), ncol(OOF)))

# ---- 评分工具 ---------------------------------------------------------------
auc_of <- function(pred) as.numeric(pROC::auc(pROC::roc(y, pred, quiet = TRUE)))

# 二层模型也必须交叉验证，否则报出来的分数是在训练数据上算的，虚高。
# 用的是同一套 folds，和一层保持一致。
cv_blend <- function(blend_fn) {
  pred <- rep(NA_real_, n)
  for (k in sort(unique(folds))) {
    tr <- folds != k
    va <- folds == k
    pred[va] <- blend_fn(OOF[tr, , drop = FALSE], y[tr], OOF[va, , drop = FALSE])
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
results$logistic <- cv_blend(blend_logistic)
cat(sprintf("\n二层 logistic   CV AUC %.5f\n", auc_of(results$logistic)))

# ---- 方法二：rank 平均 -------------------------------------------------------
# 不需要训练，直接对每列取秩再平均。
# AUC 只看排序，所以这个方法天然契合，而且对各模型输出尺度的差异完全免疫。
rank_avg <- function(M) rowMeans(apply(M, 2, rank))
results$rank_avg <- rank_avg(OOF)
cat(sprintf("rank 平均       CV AUC %.5f\n", auc_of(results$rank_avg)))

# ---- 方法三：爬山法 ---------------------------------------------------------
# 从空集合开始，每一步在所有候选模型里挑一个加进来（可以重复挑同一个，
# 等效于给它更大的权重），使当前平均预测的 AUC 提升最多。
# 提升不到阈值就停。
blend_hillclimb <- function(X_tr, y_tr, X_va, n_step = 30L, tol = 1e-6) {
  auc_tr <- function(p) as.numeric(pROC::auc(pROC::roc(y_tr, p, quiet = TRUE)))

  # 在秩空间里做，避免各模型输出尺度不一致
  R_tr <- apply(X_tr, 2, rank)
  R_va <- apply(X_va, 2, rank)

  weights <- rep(0, ncol(X_tr))
  cur_sum <- rep(0, nrow(X_tr))
  best    <- -Inf

  for (step in seq_len(n_step)) {
    gains <- vapply(seq_len(ncol(X_tr)), function(j) {
      auc_tr((cur_sum + R_tr[, j]) / step)
    }, numeric(1))
    j <- which.max(gains)
    if (gains[j] <= best + tol) break
    best <- gains[j]
    weights[j] <- weights[j] + 1
    cur_sum <- cur_sum + R_tr[, j]
  }

  if (sum(weights) == 0) return(rowMeans(R_va))
  as.numeric(R_va %*% weights) / sum(weights)
}
results$hillclimb <- cv_blend(blend_hillclimb)
cat(sprintf("爬山法          CV AUC %.5f\n", auc_of(results$hillclimb)))

# ---- 选优 -------------------------------------------------------------------
scores <- vapply(results, auc_of, numeric(1))
best_name <- names(scores)[which.max(scores)]

cat("\n----------------------------------------\n")
cat("集成方式对比：\n")
for (nm in names(scores)) {
  cat(sprintf("  %-14s %.5f%s\n", nm, scores[[nm]],
              if (nm == best_name) "   <- 最优" else ""))
}

single_best <- max(vapply(model_names, function(nm) auc_of(OOF[, nm]), numeric(1)))
cat(sprintf("\n最好的单模型    %.5f\n", single_best))
cat(sprintf("集成带来的提升  %+.5f\n", max(scores) - single_best))

# ---- 生成最终测试集预测 -----------------------------------------------------
# 用全部 OOF 训一次二层模型，套到 TEST 上。
final_test <- switch(
  best_name,
  logistic  = blend_logistic(OOF, y, TEST),
  rank_avg  = rank_avg(TEST),
  hillclimb = blend_hillclimb(OOF, y, TEST)
)

saveRDS(list(
  method      = best_name,
  cv_auc      = max(scores),
  all_scores  = scores,
  models      = model_names,
  test_pred   = final_test
), "output/ensemble_best.rds")

cat(sprintf("\n已保存 output/ensemble_best.rds（方式 %s，CV %.5f）\n",
            best_name, max(scores)))
cat('下一步：source("R/08_submit.R")\n')
