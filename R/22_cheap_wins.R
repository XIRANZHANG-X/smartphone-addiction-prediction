# =============================================================================
# 22_cheap_wins.R —— 讨论区里三项便宜但没测过的做法
#
# 用法：Rscript R/22_cheap_wins.R
# 产出：output/cheap_wins.rds
#
# -----------------------------------------------------------------------------
# 三项分别对应讨论区哪一帖，以及为什么值得测
# -----------------------------------------------------------------------------
# A. 秩变换 + 带正则的负权重 stacking（第 13 帖 protects-lab）
#    我们已有交叉拟合的 logistic 元模型（允许负权重），但它输给爬山
#    （0.96458 对 0.96487）。第 13 帖那个变体差在两点：**先把各成员做秩变换**，
#    再上**带正则**的元模型。他实测 0.969557 -> 0.969636（+0.00008），
#    且十个连续数据区块全为正。我们从没试过这个组合。
#
#    ⚠ 口径：用的是 output/oof/ 里的 Tier B 预测，那是**加入编码之前**的候选池。
#    所以这测的是「这个集成方法本身好不好」，不是「我们现在的流程能到多少」。
#
# B. 种子平均（第 24 帖 YKuma）
#    他实测 GBDT 上四个种子值 +0.0002（NN 上 +0.0013）。我们的 make_xgb
#    从不设 xgboost 自己的种子，所以一直跑在默认的 random_state = 0 上，
#    从来没做过种子平均。
#
# C. 更低的学习率出正式结果（第 46 帖 Tilii）
#    「搜参用 0.02~0.05，正式跑用 0.01 或 0.005。」我们全程固定 0.05。
#    这是纯粹的基础设置，与我们「基础功课比方法创新值钱」的主线一致。
#
# 口径与 R/06_framework.R 一致：同一份冻结折叠、同一个工厂、同样的早停。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

SEED <- 20260821L
R <- list()
hr <- function(x) cat("\n", strrep("=", 74), "\n ", x, "\n", strrep("=", 74), "\n", sep = "")

fast_auc <- function(y, p) {
  ok <- !is.na(p); y <- y[ok]; p <- p[ok]
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# =============================================================================
hr("A. 秩变换 + 带正则的负权重 stacking（第 13 帖）")
# =============================================================================
fs <- list.files("output/oof", pattern = "^oof_[^g].*\\.rds$", full.names = TRUE)
nms <- sub("^oof_", "", sub("\\.rds$", "", basename(fs)))
OOF <- do.call(cbind, lapply(fs, readRDS))
colnames(OOF) <- nms
y   <- readRDS("output/raw_train.rds")$addicted_label
folds <- readRDS("output/folds.rds")
cat(sprintf("候选 %d 个，%s 行（Tier B，**加入编码之前**的口径）\n",
            ncol(OOF), format(nrow(OOF), big.mark = ",")))

# 交叉拟合的元模型：在元折之外拟合，评在元折之内
cv_meta <- function(M, fitfun) {
  out <- rep(NA_real_, nrow(M))
  for (k in sort(unique(folds))) {
    tr <- which(folds != k); va <- which(folds == k)
    out[va] <- fitfun(M[tr, , drop = FALSE], y[tr], M[va, , drop = FALSE])
  }
  out
}

# 秩变换：把每一列换成它自己的秩（缩放到 0~1），中和各模型的尺度与校准差异
to_rank <- function(M) apply(M, 2, function(v) rank(v, ties.method = "average") / length(v))

fit_plain_glm <- function(Xtr, ytr, Xva) {
  d <- data.frame(.y = ytr, Xtr)
  m <- suppressWarnings(stats::glm(.y ~ ., data = d, family = stats::binomial()))
  as.numeric(predict(m, newdata = data.frame(Xva), type = "response"))
}
fit_ridge <- function(Xtr, ytr, Xva) {
  cvm <- glmnet::cv.glmnet(as.matrix(Xtr), ytr, family = "binomial",
                           alpha = 0, nfolds = 5)
  as.numeric(predict(cvm, as.matrix(Xva), s = "lambda.min", type = "response"))
}

RANK <- to_rank(OOF)
res_A <- c(
  `最好单模型`               = max(apply(OOF, 2, function(v) fast_auc(y, v))),
  `等权平均（概率）`         = fast_auc(y, rowMeans(OOF)),
  `等权平均（秩）`           = fast_auc(y, rowMeans(RANK)),
  `logistic（概率，现用）`   = fast_auc(y, cv_meta(OOF,  fit_plain_glm)),
  `logistic（秩）`           = fast_auc(y, cv_meta(RANK, fit_plain_glm)),
  `ridge 正则（概率）`       = fast_auc(y, cv_meta(OOF,  fit_ridge)),
  `ridge 正则（秩）★第13帖`  = fast_auc(y, cv_meta(RANK, fit_ridge))
)
for (i in seq_along(res_A))
  cat(sprintf("  %-26s %.5f\n", names(res_A)[i], res_A[i]))
cat(sprintf("\n我们现用的爬山法（存档值）  0.96487\n"))
cat("=> 若「ridge 正则（秩）」超过 0.96487，第 13 帖的做法在我们这里也成立。\n")
R$stacking <- res_A

# =============================================================================
hr("B. 种子平均（第 24 帖）")
# =============================================================================
feat  <- readRDS("output/features_raw.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]; y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

run_one <- function(params = list(), tag = "") {
  fp <- make_xgb(params)
  oof <- rep(NA_real_, nrow(X_pool)); it <- integer(0)
  t0 <- Sys.time()
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    X_tr <- derive_features(copy(X_pool[tr])); X_va <- derive_features(copy(X_pool[va]))
    e <- fit_target_encoder(X_tr, y_pool[tr])
    X_tr <- apply_target_encoder(e, X_tr); X_va <- apply_target_encoder(e, X_va)
    p <- fp(X_tr, y_pool[tr], X_va)
    b <- attr(p, "best_iteration"); if (!is.null(b)) it <- c(it, b)
    oof[va] <- as.numeric(p)
  }
  a <- sapply(sort(unique(f_pool)), function(k) fast_auc(y_pool[f_pool == k], oof[f_pool == k]))
  cat(sprintf("  %-22s AUC %.5f ± %.5f  (%.1f 分钟, %.0f 轮)\n", tag, mean(a), sd(a),
              as.numeric(difftime(Sys.time(), t0, units = "mins")), mean(it)))
  list(oof = oof, auc = a)
}

seeds <- c(0L, 20260821L, 34540L)
S <- lapply(seq_along(seeds), function(i)
  run_one(list(seed = seeds[i]), sprintf("种子 %d", seeds[i])))
# 对预测取秩平均（AUC 只看秩，各模型尺度可能不同）
avg <- rowMeans(sapply(S, function(s) rank(s$oof, ties.method = "average")))
a_avg <- sapply(sort(unique(f_pool)), function(k) fast_auc(y_pool[f_pool == k], avg[f_pool == k]))
cat(sprintf("  %-22s AUC %.5f ± %.5f\n", "三种子秩平均", mean(a_avg), sd(a_avg)))
d <- a_avg - S[[1]]$auc
cat(sprintf("\n  对单种子（seed=0，即我们此前一直在跑的）：%+.5f，%d/5 折同号\n",
            mean(d), sum(d > 0)))
R$seed_avg <- list(single = lapply(S, function(s) s$auc), averaged = a_avg)

# =============================================================================
hr("C. 更低的学习率（第 46 帖）")
# =============================================================================
cat("Tilii：搜参用 0.02~0.05，正式跑用 0.01 或 0.005。我们全程固定 0.05。\n\n")
L <- list()
for (eta in c(0.05, 0.02, 0.01)) {
  L[[as.character(eta)]] <- run_one(list(eta = eta), sprintf("eta = %.3f", eta))
}
base <- L[["0.05"]]$auc
cat("\n配对检验（对 eta = 0.05）：\n")
for (nm in names(L)) {
  if (nm == "0.05") next
  d <- L[[nm]]$auc - base
  cat(sprintf("  eta = %-6s %+.5f   Cohen d %5.2f   %d/5 同号\n",
              nm, mean(d), mean(d) / sd(d), sum(d > 0)))
}
R$eta <- lapply(L, function(z) z$auc)

saveRDS(R, "output/cheap_wins.rds")
cat("\n已保存 output/cheap_wins.rds\n")
