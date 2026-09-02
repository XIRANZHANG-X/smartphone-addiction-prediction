# =============================================================================
# 24_onehot_lr.R —— 精确取值 one-hot + 逻辑回归，能不能到 0.96？
#
# 用法：Rscript R/24_onehot_lr.R
# 产出：output/onehot_lr.rds
#
# -----------------------------------------------------------------------------
# 这一项可能推翻我们自己的一段结论
# -----------------------------------------------------------------------------
# 讨论区第 26 帖 broccoli beef 贴了十行代码，声称**纯逻辑回归能到 0.96005**：
#   把所有数值列 astype(str) 后 one-hot，加一个 daily&social 的两列组合，
#   10 折 StratifiedKFold，sklearn 的 LogisticRegression 默认参数。
#
# 而我们的 glmnet 线是这样的：
#   原始 17 特征            0.91452
#   加逐取值 target encoding 0.94805
#
# 如果 0.96 成立，我们文档里那句「线性模型完全无法表示非单调查找」就说过头了。
# 正确的说法应该是：**它无法在原始尺度上做，但给它足够的自由度就能做。**
#
# 机制上的预期：
#   target encoding 把每个取值压成**一个数**（该取值的目标均值），
#   线性模型只能对这个数做线性缩放 —— 它拿到的是查找表的一个单调投影。
#   one-hot 给每个取值**一个自由参数**，线性模型可以学出任意的取值→logit 映射。
#   后者严格更强。所以预期 one-hot > TE，问题只是差多少。
#
# -----------------------------------------------------------------------------
# 一个必须拆开的混淆
# -----------------------------------------------------------------------------
# 原始配方里 pandas 的 astype(str) 会把 NaN 变成字符串 "nan"，于是
# **缺失自动获得了自己的一个哑变量** —— 这等于偷偷塞进了缺失指示。
# 而我们的发现 6 说缺失指示对目标无信息。两者要分开，所以这里测三个配置：
#   C  one-hot，NA 作为自己的水平（原配方）
#   E  先做 L2 中位数填补再 one-hot（没有 NA 水平）—— 隔离 NA 的贡献
#   D  C + daily&social 交互（原配方的完整版）
#
# 口径：同一份冻结折叠、Tier A 20 万行，可与上面两个 glmnet 数字直接比。
# 模型用 ridge（alpha = 0），对应 sklearn LogisticRegression 的默认 L2。
# =============================================================================

suppressMessages({library(data.table); library(Matrix); library(glmnet); library(pROC)})

SEED <- 20260821L
NUM  <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
          "work_study_hours", "sleep_hours", "notifications_per_day",
          "app_opens_per_day", "weekend_screen_time")

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X <- train_all[sub]; y <- train_all$addicted_label[sub]; f <- folds[sub]
cat(sprintf("Tier A：%s 行，冻结 5 折\n", format(nrow(X), big.mark = ",")))

fast_auc <- function(y, p) {
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

#' 把若干列按**精确取值**展开成稀疏 one-hot
#'
#' NA 是否单独成一个水平由调用方决定（传进来之前处理好）。
#' 只出现一次的取值不做过滤：训练折里它拟合一行、验证折里它未出现，
#' 两种情况都无害，只是浪费一列。原配方也没过滤。
build_onehot <- function(dt, cols, min_count = 1L) {
  blocks <- lapply(cols, function(cc) {
    v <- as.character(dt[[cc]]); v[is.na(v)] <- "__NA__"
    lv <- names(which(table(v) >= min_count))
    j  <- match(v, lv)
    keep <- !is.na(j)
    sparseMatrix(i = which(keep), j = j[keep], x = 1,
                 dims = c(length(v), length(lv)),
                 dimnames = list(NULL, paste0(cc, "=", lv)))
  })
  do.call(cbind, blocks)
}

run <- function(tag, M) {
  cat(sprintf("  %-30s %s 列 ... ", tag, format(ncol(M), big.mark = ",")))
  flush.console()
  aucs <- numeric(0); t0 <- Sys.time()
  for (k in sort(unique(f))) {
    set.seed(SEED + k)
    tr <- which(f != k); va <- which(f == k)
    # ridge（alpha = 0），对应 sklearn LogisticRegression 的默认 L2
    cvm <- cv.glmnet(M[tr, , drop = FALSE], y[tr], family = "binomial",
                     alpha = 0, nfolds = 3, nlambda = 20)
    p <- as.numeric(predict(cvm, M[va, , drop = FALSE],
                            s = "lambda.min", type = "response"))
    aucs <- c(aucs, fast_auc(y[va], p))
  }
  cat(sprintf("AUC %.5f ± %.5f  (%.1f 分钟)\n", mean(aucs), sd(aucs),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  aucs
}

R <- list()

# --- C：原配方核心，NA 自成一个水平 -----------------------------------------
Mc <- build_onehot(X, NUM)
R$C_onehot <- run("C  one-hot（NA 成水平）", Mc)

# --- E：先中位数填补再 one-hot，隔离 NA 水平的贡献 ---------------------------
Xi <- copy(X)
for (cc in NUM) set(Xi, j = cc,
                    value = fifelse(is.na(Xi[[cc]]),
                                    median(Xi[[cc]], na.rm = TRUE), Xi[[cc]]))
Me <- build_onehot(Xi, NUM)
R$E_onehot_imputed <- run("E  one-hot（先填补，无 NA 水平）", Me)

# --- D：加 daily & social 的交互，原配方的完整版 -----------------------------
Xp <- copy(X)
Xp[, pair := paste(daily_screen_time_hours, social_media_hours, sep = "&")]
Md <- cbind(Mc, build_onehot(Xp, "pair"))
R$D_onehot_pair <- run("D  C + daily&social 交互", Md)

# -----------------------------------------------------------------------------
cat("\n", strrep("=", 74), "\n glmnet 线的全部配置（同一套冻结折叠，Tier A）\n",
    strrep("=", 74), "\n", sep = "")
ref <- c(`原始 17 特征（L2 填补）` = 0.91452,
         `+ 逐取值 target encoding` = 0.94805)
for (i in seq_along(ref))
  cat(sprintf("  %-34s %.5f\n", names(ref)[i], ref[i]))
for (nm in names(R))
  cat(sprintf("  %-34s %.5f ± %.5f\n", nm, mean(R[[nm]]), sd(R[[nm]])))

cat(sprintf("\n讨论区第 26 帖报告的数字（10 折、含交互）：0.96005\n"))
cat(sprintf("我们最好的 GBDT（xgboost + TE）：           0.96562\n\n"))

d <- R$C_onehot - 0.94805
cat(sprintf("one-hot 相对 target encoding：%+.5f\n", mean(d)))
d2 <- R$C_onehot - R$E_onehot_imputed
cat(sprintf("NA 单独成水平的贡献：        %+.5f（%d/5 折同号）\n",
            mean(d2), sum(d2 > 0)))
d3 <- R$D_onehot_pair - R$C_onehot
cat(sprintf("daily&social 交互的贡献：    %+.5f（%d/5 折同号）\n",
            mean(d3), sum(d3 > 0)))

saveRDS(R, "output/onehot_lr.rds")
cat("\n已保存 output/onehot_lr.rds\n")
