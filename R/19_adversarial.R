# =============================================================================
# 19_adversarial.R —— 对抗验证 + 空重要性检验
#
# 用法：Rscript R/19_adversarial.R
# 产出：output/adversarial.rds + 屏幕报告
#
# -----------------------------------------------------------------------------
# 这两道关卡是我们原来缺的（来自讨论区第 30 帖 Muhammad Faheem 的三道关卡）
# -----------------------------------------------------------------------------
# 我们已有的：累积消融（R/09_ablation.R）。
# 这里补上另外两道：
#
#   **对抗验证** —— 训练一个分类器去区分"这一行来自训练集还是测试集"。
#     如果某个特征在这里排得很高，它就是"划分身份"的代理，而不是信号。
#     第 30 帖的作者正是靠这一步，在信任新特征之前排除了他早先踩过的坑
#     （缺失指示抬高本地 CV、压垮公榜）。
#
#   **空重要性** —— 把目标**打乱**成纯噪声再重训，记录"当真的没有东西可学时，
#     重要性长什么样"。任何得分接近这条噪声基线的特征都只是装饰。
#
# -----------------------------------------------------------------------------
# 读这份输出时必须分清的两件事（讨论区第 41、50、53 帖反复强调）
# -----------------------------------------------------------------------------
#   「缺失对目标有无信息」  和  「缺失对划分身份有无信息」
# 是两个不同的问题，答案分别是「无」和「有」。
# 一个不说明自己指哪一个的句子，就会被读成另一个。
# 所以本脚本的对抗 AUC **不能**被读作特征漂移。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

SEED <- 20260821L
set.seed(SEED)
res <- list()
hr <- function(x) cat("\n", strrep("=", 74), "\n ", x, "\n", strrep("=", 74), "\n", sep = "")

feat <- readRDS("output/features_raw.rds")

# =============================================================================
hr("第一部分：对抗验证 —— 训练集与测试集能被区分吗？")
# =============================================================================
# 五个变体，用来定位可分性到底来自哪里。
# 期望（依据讨论区第 41 帖 Dariush Afshar 的测量）：
#   原始特征约 0.564，仅缺失指示约 0.565，
#   而一旦把缺失填掉或只看完整行，就塌到 0.50 —— 即「漂移的全部就是缺失」。

NUM <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
         "work_study_hours", "sleep_hours", "notifications_per_day",
         "app_opens_per_day", "weekend_screen_time")
CAT <- c("gender", "stress_level", "academic_work_impact")
ALL <- c(NUM, CAT)

# 为了跑得动，抽样到 20 万（训练/测试各按比例抽）
n_take <- 200000L
idx_tr <- sample(which(feat$is_train == 1L), round(n_take * 0.7))
idx_te <- sample(which(feat$is_train == 0L), round(n_take * 0.3))
A <- feat[c(idx_tr, idx_te)]
A[, adv_y := 1L - is_train]          # 1 = 测试集
cat(sprintf("对抗样本：%s 行（训练 %s + 测试 %s）\n",
            format(nrow(A), big.mark = ","),
            format(length(idx_tr), big.mark = ","),
            format(length(idx_te), big.mark = ",")))

adv_run <- function(label, X) {
  y <- A$adv_y
  f <- integer(nrow(X))
  for (cls in c(0L, 1L)) { ii <- which(y == cls); f[sample(ii)] <- rep_len(1:5, length(ii)) }
  fp <- make_xgb(list(eta = 0.1, max_depth = 6), max_rounds = 2000L)
  aucs <- numeric(0)
  for (k in 1:5) {
    set.seed(SEED + k)
    tr <- which(f != k); va <- which(f == k)
    p <- fp(X[tr], y[tr], X[va])
    aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(y[va], as.numeric(p), quiet = TRUE))))
  }
  cat(sprintf("  %-34s 对抗 AUC %.6f\n", label, mean(aucs)))
  mean(aucs)
}

# 变体一：原始特征，保留 NaN，无标志
V1 <- A[, c(ALL, "adv_y"), with = FALSE][, adv_y := NULL]
r1 <- adv_run("原始特征，保留 NaN，无标志", copy(V1))

# 变体二：原始 + 12 个缺失指示
V2 <- copy(V1)
for (cc in ALL) set(V2, j = paste0("is_na_", cc), value = as.integer(is.na(V2[[cc]])))
r2 <- adv_run("原始 + 12 个缺失指示", copy(V2))

# 变体三：仅缺失指示
V3 <- V2[, paste0("is_na_", ALL), with = FALSE]
r3 <- adv_run("仅缺失指示（12 列）", copy(V3))

# 变体四：把缺失填掉（各列中位数 / 众数），指示不给
V4 <- copy(V1)
for (cc in NUM) set(V4, j = cc, value = fifelse(is.na(V4[[cc]]),
                                                median(V4[[cc]], na.rm = TRUE), V4[[cc]]))
for (cc in CAT) {
  lv <- levels(V4[[cc]]); md <- lv[which.max(table(V4[[cc]]))]
  v <- as.character(V4[[cc]]); v[is.na(v)] <- md
  set(V4, j = cc, value = factor(v, levels = lv))
}
r4 <- adv_run("缺失已填补掉", copy(V4))

# 变体五：只用完整行
cm <- complete.cases(A[, ..ALL])
V5 <- A[cm, ..ALL]; y5 <- A$adv_y[cm]
{
  f <- integer(nrow(V5))
  for (cls in c(0L, 1L)) { ii <- which(y5 == cls); f[sample(ii)] <- rep_len(1:5, length(ii)) }
  fp <- make_xgb(list(eta = 0.1, max_depth = 6), max_rounds = 2000L)
  aucs <- numeric(0)
  for (k in 1:5) {
    set.seed(SEED + k); tr <- which(f != k); va <- which(f == k)
    p <- fp(V5[tr], y5[tr], V5[va])
    aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(y5[va], as.numeric(p), quiet = TRUE))))
  }
  r5 <- mean(aucs)
  cat(sprintf("  %-34s 对抗 AUC %.6f  （%s 行）\n", "只用完整行", r5,
              format(nrow(V5), big.mark = ",")))
}

cat("\n判读：\n")
cat(sprintf("  填补掉缺失之后，可分性保留了 %.1f%%（(r4-0.5)/(r1-0.5)）\n",
            100 * (r4 - 0.5) / (r1 - 0.5)))
cat(sprintf("  只看完整行，可分性保留了 %.1f%%\n", 100 * (r5 - 0.5) / (r1 - 0.5)))
cat("\n  若后两个接近 0：**漂移的全部就是缺失，底下没有取值漂移**。\n")
cat("  这是好消息 —— 它意味着私榜的重排不会来自分布漂移，\n")
cat("  也意味着不需要任何精心设计的漂移校正方案。\n")
cat("\n  ⚠ 但这**不是**在说缺失能预测目标。那是另一个问题，\n")
cat("     答案见 R/17_discussion_checks.R 核查 4：n_missing 的目标 AUC = 0.50172。\n")

res$adversarial <- c(raw = r1, with_flags = r2, flags_only = r3,
                     imputed = r4, complete_only = r5)

# =============================================================================
hr("第二部分：空重要性 —— 没有信号时重要性长什么样")
# =============================================================================
# 做法：把 y 打乱，重训，记录增益重要性。真实特征的增益应当远高于这条基线。
# 讨论区第 30 帖用它证明结构性特征（残差、小数位）确实在利用生成器痕迹：
#   daily_screen_time_hours_decimals 是噪声基线的 11.68 倍，other_screen_abs 是 8.83 倍。

sub <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X <- derive_features(copy(train_all[sub]))
y <- train_all$addicted_label[sub]
uc <- setdiff(names(X), c("id", "addicted_label", "is_train"))

gain_of <- function(yy, seed) {
  set.seed(seed)
  M <- as.matrix(X[, ..uc][, lapply(.SD, function(v)
        if (is.factor(v)) as.integer(v) else as.numeric(v))])
  d <- xgboost::xgb.DMatrix(M, label = yy, missing = NA)
  m <- xgboost::xgb.train(list(objective = "binary:logistic", eval_metric = "auc",
                               eta = 0.1, max_depth = 6, subsample = 0.8,
                               colsample_bytree = 0.8, min_child_weight = 10,
                               tree_method = "hist",
                               nthread = parallel::detectCores()),
                          d, nrounds = 300, verbose = 0)
  im <- xgboost::xgb.importance(model = m)
  setNames(im$Gain, im$Feature)[uc]
}

cat("训练真实模型 ...\n")
g_real <- gain_of(y, SEED)
cat("训练 3 个打乱标签的对照 ...\n")
g_null <- sapply(1:3, function(i) gain_of(sample(y), SEED + 100L * i))
g_null_mean <- rowMeans(g_null, na.rm = TRUE)

ni <- data.table(feature = uc,
                 gain_real = as.numeric(g_real[uc]),
                 gain_null = as.numeric(g_null_mean[uc]))
ni[is.na(gain_real), gain_real := 0]
ni[is.na(gain_null) | gain_null <= 0, gain_null := 1e-9]
ni[, ratio := gain_real / gain_null]
setorder(ni, -ratio)
cat("\n")
cat(sprintf("%-24s %10s %10s %8s\n", "特征", "真实增益", "空增益", "倍数"))
for (i in seq_len(nrow(ni)))
  cat(sprintf("%-24s %10.5f %10.5f %8.2f%s\n", ni$feature[i], ni$gain_real[i],
              ni$gain_null[i], ni$ratio[i],
              if (ni$ratio[i] < 2) "   <- 接近噪声" else ""))

cat("\n判读：倍数 < 2 的特征，其增益与「完全没有信号时」难以区分，属于装饰。\n")
res$null_importance <- ni

saveRDS(res, "output/adversarial.rds")
cat("\n已保存 output/adversarial.rds\n")
