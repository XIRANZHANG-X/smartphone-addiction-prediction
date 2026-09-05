# =============================================================================
# 39_train_test_shift.R —— 训练集 / 测试集分布一致性检查
#
# 用法：Rscript R/39_train_test_shift.R
# 产出：output/train_test_shift.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 要回答的问题
# -----------------------------------------------------------------------------
# CV 和 LB 已经验证过同向（+0.00136 的差距有合理解释，来自训练数据量而非
# 过拟合），侧面说明没有明显的协变量偏移，但没人正面画过/测过
# 「12 个特征在 train/test 上的分布是否一致」。这是标准 EDA 里该做的一环。
#
# 只读 output/raw_train.rds、output/raw_test.rds —— 都是已经存在的真实
# 竞赛数据（01_load.R 的产物），不训练任何模型、不做任何插补。
#
# 效应量优先于 p 值：两个样本量差 2 倍多（69.1万 vs 29.6万），
# KS/卡方的 p 值在这个规模下任何微小差异都会显著，跟发现 2 审查时
# 用的原则一样——只看 D 值 / 比例差，不看 p 值。
# =============================================================================

suppressMessages(library(data.table))

train <- readRDS("output/raw_train.rds")
test  <- readRDS("output/raw_test.rds")

hr <- function(x) cat("\n", strrep("=", 78), "\n", x, "\n", strrep("=", 78), "\n", sep = "")

NUM_COLS <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "notifications_per_day",
              "app_opens_per_day", "weekend_screen_time")
CAT_COLS <- c("gender", "stress_level", "academic_work_impact")

hr("样本量")
cat(sprintf("训练集 %s 行，测试集 %s 行（比例 %.2f : 1）\n",
            format(nrow(train), big.mark = ","), format(nrow(test), big.mark = ","),
            nrow(train) / nrow(test)))

# ---- 一、数值列：均值/标准差/KS D 值 ----------------------------------------
hr("一、数值特征分布对比")
cat(sprintf("%-26s %10s %10s %8s %10s %10s %8s\n",
            "特征", "train 均值", "test 均值", "差", "train sd", "test sd", "KS D"))

num_res <- list()
for (col in NUM_COLS) {
  tr <- train[[col]]; te <- test[[col]]
  tr <- tr[!is.na(tr)]; te <- te[!is.na(te)]
  ks <- suppressWarnings(ks.test(tr, te))
  num_res[[col]] <- data.table(
    feature = col, mean_train = mean(tr), mean_test = mean(te),
    diff = mean(tr) - mean(te), sd_train = sd(tr), sd_test = sd(te),
    ks_D = unname(ks$statistic), ks_p = ks$p.value
  )
  cat(sprintf("%-26s %10.3f %10.3f %+8.3f %10.3f %10.3f %8.4f\n",
              col, mean(tr), mean(te), mean(tr) - mean(te), sd(tr), sd(te),
              unname(ks$statistic)))
}
num_tab <- rbindlist(num_res)

# ---- 二、类别列：各类别占比差 ------------------------------------------------
hr("二、类别特征分布对比")
cat_res <- list()
for (col in CAT_COLS) {
  tr <- train[[col]]; te <- test[[col]]
  pt <- prop.table(table(tr, useNA = "no"))
  pe <- prop.table(table(te, useNA = "no"))
  lv <- union(names(pt), names(pe))
  d <- data.table(feature = col, level = lv,
                   p_train = as.numeric(pt[lv]), p_test = as.numeric(pe[lv]))
  d[is.na(p_train), p_train := 0]; d[is.na(p_test), p_test := 0]
  d[, diff := p_train - p_test]
  cat_res[[col]] <- d
  cat(sprintf("\n%s：\n", col))
  for (i in seq_len(nrow(d))) {
    cat(sprintf("  %-12s train %6.2f%%   test %6.2f%%   差 %+.2f%%\n",
                d$level[i], 100 * d$p_train[i], 100 * d$p_test[i], 100 * d$diff[i]))
  }
}
cat_tab <- rbindlist(cat_res)

# ---- 三、缺失率对比 -----------------------------------------------------------
hr("三、每列缺失率对比（train vs test 缺失机制是否一致）")
all_cols <- c(NUM_COLS, CAT_COLS)
miss_res <- rbindlist(lapply(all_cols, function(col) {
  mt <- mean(is.na(train[[col]])); me <- mean(is.na(test[[col]]))
  data.table(feature = col, miss_train = mt, miss_test = me, diff = mt - me)
}))
cat(sprintf("%-26s %12s %12s %10s\n", "特征", "train 缺失率", "test 缺失率", "差"))
for (i in seq_len(nrow(miss_res))) {
  r <- miss_res[i]
  cat(sprintf("%-26s %11.2f%% %11.2f%% %+9.2f%%\n",
              r$feature, 100 * r$miss_train, 100 * r$miss_test, 100 * r$diff))
}

# ---- 四、总结：有没有值得警惕的偏移 ------------------------------------------
hr("四、总结")
big_ks   <- num_tab[ks_D > 0.01]
big_cat  <- cat_tab[abs(diff) > 0.01]
big_miss <- miss_res[abs(diff) > 0.005]

if (nrow(big_ks) == 0 && nrow(big_cat) == 0 && nrow(big_miss) == 0) {
  cat("没有发现任何特征在 train/test 之间有值得警惕的分布偏移\n")
  cat("（数值特征 KS D 全部 <= 0.01；类别占比差全部 <= 1 个百分点；\n")
  cat(" 缺失率差全部 <= 0.5 个百分点）。这跟 CV/LB 已验证同向是互相印证的：\n")
  cat("本地验证之所以可信，前提之一就是 train/test 本来就是同一分布的随机切分，\n")
  cat("这里是对这个前提的正面验证，而不是只靠 LB 分数间接推断。\n")
} else {
  if (nrow(big_ks))   { cat("数值特征里 KS D > 0.01 的：\n"); print(big_ks) }
  if (nrow(big_cat))  { cat("类别特征里占比差 > 1 个百分点的：\n"); print(big_cat) }
  if (nrow(big_miss)) { cat("缺失率差 > 0.5 个百分点的：\n"); print(big_miss) }
}

saveRDS(list(numeric = num_tab, categorical = cat_tab, missing = miss_res),
        "output/train_test_shift.rds")

hr("完成")
cat("结果已存至 output/train_test_shift.rds\n")
