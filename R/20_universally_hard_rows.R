# =============================================================================
# 20_universally_hard_rows.R —— 所有模型都判断错误的行长什么样
#
# 用法：Rscript R/20_universally_hard_rows.R
# 产出：output/universally_hard_rows.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 要回答的问题
# -----------------------------------------------------------------------------
# R/17_residual_exceptions.R 用「屏幕时间分箱的多数标签」这个自己造的代理
# 指标去找「例外行」。这个脚本换一个更硬的做法：直接用组长给的
# output/oof/oof_grid_*.rds —— 14 个格子（4 条插补线 × 4 种模型）在
# 同一个 20 万行子样本、同一套折叠上的真实交叉验证预测。
#
# 找出「不管插补方式、不管算法，14 个模型全部判断错误」的那批行，
# 再回头看它们的原始特征长什么样，跟发现 5/9/17 里已经找到的东西是否吻合。
#
# 只读已经存在的真实数据：output/oof/oof_grid_*.rds（组长同步）、
# output/raw_train.rds、output/subsample_200k.rds、output/folds.rds。
# 不训练任何新模型，不做任何插补。
# =============================================================================

suppressMessages(library(data.table))

hr <- function(x) cat("\n", strrep("=", 78), "\n", x, "\n", strrep("=", 78), "\n", sep = "")

train <- readRDS("output/raw_train.rds")
sub   <- readRDS("output/subsample_200k.rds")
folds <- readRDS("output/folds.rds")
y     <- train$addicted_label[sub]
dt    <- train[sub]

# ---- 收集全部 14 格网格预测 --------------------------------------------------
oof_files <- list.files("output/oof", pattern = "^oof_grid_.*\\.rds$", full.names = TRUE)
hr("收集到的网格预测")
cat(sprintf("共 %d 个文件：\n", length(oof_files)))

preds <- list()
for (f in oof_files) {
  nm <- sub("^oof_grid_", "", sub("\\.rds$", "", basename(f)))
  p  <- readRDS(f)
  if (length(p) != length(y)) {
    cat(sprintf("  [跳过] %-18s 长度 %s，应为 %s\n", nm,
                format(length(p), big.mark = ","), format(length(y), big.mark = ",")))
    next
  }
  preds[[nm]] <- p
  cat(sprintf("  [采用] %-18s\n", nm))
}
P <- do.call(cbind, preds)   # 20万 x N模型 矩阵
cat(sprintf("\n矩阵维度：%s 行 x %d 个模型\n", format(nrow(P), big.mark = ","), ncol(P)))

# ---- 每行的平均误差 -----------------------------------------------------------
# |预测概率 - 真实标签|：0 = 完全判断对，1 = 判断得离谱到底
err <- abs(P - y)
mean_err <- rowMeans(err, na.rm = TRUE)
n_confident_wrong <- rowSums(
  (P > 0.9 & y == 0) | (P < 0.1 & y == 1)   # 各模型「非常自信但判断反了」的次数
)

hr("整体误差分布")
cat(sprintf("每行平均误差（14 个模型平均）的分位数：\n"))
print(round(quantile(mean_err, c(0, .5, .9, .95, .99, .999, 1)), 4))

# ---- 挑出「普遍判断错误」的行 ------------------------------------------------
# 定义：平均误差 > 0.5（大多数模型都判断反了），而不是某一个模型偶尔犯错
thresh <- 0.5
hard_idx <- which(mean_err > thresh)
hr("普遍判断错误的行（14 个模型平均误差 > 0.5）")
cat(sprintf("行数：%s（占子样本 %.2f%%）\n",
            format(length(hard_idx), big.mark = ","), 100 * length(hard_idx) / nrow(P)))
cat(sprintf("其中全部 %d 个模型一致判断反了（各模型误差都 > 0.5）的行：%s\n",
            ncol(P), format(sum(rowSums(err > 0.5) == ncol(P)), big.mark = ",")))

# ---- 这些行的原始特征长什么样 ------------------------------------------------
dt[, mean_err := mean_err]
dt[, is_hard  := mean_err > thresh]
dt[, row_id   := .I]

hr("难行 vs 其余行：特征对比")
NUM_COLS <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "notifications_per_day",
              "app_opens_per_day", "weekend_screen_time")

cmp <- rbindlist(lapply(NUM_COLS, function(col) {
  hard <- dt[is_hard == TRUE][[col]]
  easy <- dt[is_hard == FALSE][[col]]
  hard <- hard[!is.na(hard)]; easy <- easy[!is.na(easy)]
  tt <- t.test(hard, easy)
  data.table(feature = col, mean_hard = mean(hard), mean_easy = mean(easy),
             diff = mean(hard) - mean(easy), p = tt$p.value)
}))
cat(sprintf("%-26s %10s %10s %10s %10s\n", "特征", "难行均值", "其余均值", "差", "p"))
for (i in seq_len(nrow(cmp))) {
  r <- cmp[i]
  cat(sprintf("%-26s %10.3f %10.3f %+10.3f %10.2e\n",
              r$feature, r$mean_hard, r$mean_easy, r$diff, r$p))
}

hr("难行的成瘾率 vs 缺失特征个数")
cat(sprintf("难行成瘾率：%.4f；其余行成瘾率：%.4f；全体：%.4f\n",
            mean(dt[is_hard == TRUE]$addicted_label),
            mean(dt[is_hard == FALSE]$addicted_label),
            mean(dt$addicted_label)))

miss_cols <- c(NUM_COLS, "gender", "stress_level", "academic_work_impact")
dt[, n_missing_row := rowSums(is.na(.SD)), .SDcols = miss_cols]
cat(sprintf("难行平均缺失特征数：%.2f；其余行：%.2f\n",
            mean(dt[is_hard == TRUE]$n_missing_row),
            mean(dt[is_hard == FALSE]$n_missing_row)))

# ---- 跟发现 17（残差例外）的重叠度 -------------------------------------------
hr("跟 R/17_residual_exceptions.R 的『例外行』定义是否重叠")
one <- dt[!is.na(daily_screen_time_hours), .(row_id, daily_screen_time_hours, addicted_label)]
one[, bin := floor(daily_screen_time_hours)]
bin_rate <- one[, .(rate = mean(addicted_label)), by = bin]
one <- merge(one, bin_rate, by = "bin")
one[, majority_label := as.integer(rate >= 0.5)]
one[, is_exception17  := addicted_label != majority_label]

overlap <- merge(dt[, .(row_id, is_hard)], one[, .(row_id, is_exception17)],
                  by = "row_id", all = FALSE)
tab <- table(hard_20 = overlap$is_hard, exception_17 = overlap$is_exception17)
print(tab)
if ("TRUE" %in% rownames(tab) && "TRUE" %in% colnames(tab)) {
  cat(sprintf("\n「20 号脚本的难行」里，同时也是「17 号脚本的例外行」的比例：%.1f%%\n",
              100 * tab["TRUE", "TRUE"] / sum(tab["TRUE", ])))
}

saveRDS(list(mean_err = mean_err, hard_idx = hard_idx, feature_compare = cmp,
             n_models = ncol(P), models_used = colnames(P)),
        "output/universally_hard_rows.rds")

hr("完成")
cat("结果已存至 output/universally_hard_rows.rds\n")
