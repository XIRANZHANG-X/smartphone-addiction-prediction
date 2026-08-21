# =============================================================================
# 01_load.R —— 读取原始数据
#
# 作用：把 data/raw/ 下的三个 csv 读进来，统一类型，存成 rds 供后续脚本快速加载。
# 用法：source("R/01_load.R")
# 产出：output/raw_train.rds
#       output/raw_test.rds
#       output/sample_submission.rds
#
# 为什么用 data.table::fread 而不是 read.csv：
#   train.csv 有 69 万行。read.csv 在这个规模上要跑几十秒且内存翻几倍，
#   fread 一两秒就能读完。本项目全程用 data.table。
# =============================================================================

library(data.table)

# ---- 路径 -------------------------------------------------------------------
dir_raw <- file.path("data", "raw")
dir_out <- "output"
if (!dir.exists(dir_out)) dir.create(dir_out, recursive = TRUE)

f_train  <- file.path(dir_raw, "train.csv")
f_test   <- file.path(dir_raw, "test.csv")
f_sample <- file.path(dir_raw, "sample_submission.csv")

for (f in c(f_train, f_test, f_sample)) {
  if (!file.exists(f)) {
    stop("找不到数据文件：", f,
         "\n数据没有放进 git 仓库（约 68MB，且竞赛数据不宜公开再分发）。",
         "\n请从 Kaggle 竞赛页下载 train.csv / test.csv / sample_submission.csv，",
         "\n放进 data/raw/ 目录后重新运行。")
  }
}

# ---- 列定义 -----------------------------------------------------------------
# 显式写出来，一是当文档，二是防止 fread 的类型推断在不同机器上给出不同结果。
COLS_NUMERIC <- c(
  "age",
  "daily_screen_time_hours",
  "social_media_hours",
  "gaming_hours",
  "work_study_hours",
  "sleep_hours",
  "notifications_per_day",
  "app_opens_per_day",
  "weekend_screen_time"
)

COLS_CATEGORICAL <- c(
  "gender",               # Male / Female / Other
  "stress_level",         # High / Medium / Low
  "academic_work_impact"  # Yes / No
)

COL_TARGET <- "addicted_label"
COL_ID     <- "id"

# ---- 读取 -------------------------------------------------------------------
cat("读取数据 ...\n")

# na.strings = c("", "NA")：原始 csv 里缺失值是空字符串，必须显式声明，
# 否则类别列的缺失会变成 "" 这个字面值而不是 NA。
train <- fread(f_train,  na.strings = c("", "NA"))
test  <- fread(f_test,   na.strings = c("", "NA"))
subm  <- fread(f_sample, na.strings = c("", "NA"))

# ---- 类型统一 ---------------------------------------------------------------
# 类别列转 factor，并且 train 和 test 必须用同一套 levels，
# 否则做 one-hot 或喂给模型时列会对不上。
for (col in COLS_CATEGORICAL) {
  lv <- sort(unique(c(
    as.character(train[[col]]),
    as.character(test[[col]])
  )))
  lv <- lv[!is.na(lv)]
  set(train, j = col, value = factor(train[[col]], levels = lv))
  set(test,  j = col, value = factor(test[[col]],  levels = lv))
}

# 数值列统一成 double（fread 可能把某些列读成 integer）
for (col in COLS_NUMERIC) {
  set(train, j = col, value = as.numeric(train[[col]]))
  set(test,  j = col, value = as.numeric(test[[col]]))
}

# 目标变量保持 integer 0/1。
# 注意：不要转成 factor —— xgboost 和 glmnet 都要求数值型 0/1 标签。
set(train, j = COL_TARGET, value = as.integer(train[[COL_TARGET]]))

# ---- 完整性检查 -------------------------------------------------------------
# 这些是已知的事实（见 docs/项目说明.md 第二节）。如果对不上，
# 说明数据文件不是我们分析的那一份，后面所有结论都不成立，必须立刻停下。
stopifnot(
  "训练集行数不是 691369"        = nrow(train) == 691369L,
  "测试集行数不是 296302"        = nrow(test)  == 296302L,
  "提交模板行数与测试集不一致"   = nrow(subm)  == nrow(test),
  "训练集缺少目标列"             = COL_TARGET %in% names(train),
  "测试集不应该有目标列"         = !(COL_TARGET %in% names(test)),
  "训练集 id 有重复"             = !anyDuplicated(train[[COL_ID]]),
  "测试集 id 有重复"             = !anyDuplicated(test[[COL_ID]]),
  "目标列有缺失"                 = !anyNA(train[[COL_TARGET]]),
  "目标列不是 0/1"               = all(train[[COL_TARGET]] %in% c(0L, 1L))
)

pos_rate <- mean(train[[COL_TARGET]])
if (abs(pos_rate - 0.7094) > 0.001) {
  warning("正例率 ", round(pos_rate, 4), " 与已知的 0.7094 不符，请确认数据版本。")
}

# ---- 存盘 -------------------------------------------------------------------
saveRDS(train, file.path(dir_out, "raw_train.rds"))
saveRDS(test,  file.path(dir_out, "raw_test.rds"))
saveRDS(subm,  file.path(dir_out, "sample_submission.rds"))

# ---- 报告 -------------------------------------------------------------------
cat("\n---- 数据概况 ----\n")
cat(sprintf("训练集  %s 行 × %d 列\n", format(nrow(train), big.mark = ","), ncol(train)))
cat(sprintf("测试集  %s 行 × %d 列\n", format(nrow(test),  big.mark = ","), ncol(test)))
cat(sprintf("正例率  %.4f\n", pos_rate))

cat("\n---- 缺失率 ----\n")
feat_cols <- c(COLS_NUMERIC, COLS_CATEGORICAL)
miss_rate <- vapply(feat_cols, function(c) mean(is.na(train[[c]])), numeric(1))
for (i in order(-miss_rate)) {
  cat(sprintf("  %-26s %5.2f%%\n", feat_cols[i], 100 * miss_rate[i]))
}

n_complete <- sum(stats::complete.cases(train[, ..feat_cols]))
cat(sprintf("\n完整行 %s (%.2f%%)，至少缺一个的行 %.2f%%\n",
            format(n_complete, big.mark = ","),
            100 * n_complete / nrow(train),
            100 * (1 - n_complete / nrow(train))))

cat("\n已存至 output/raw_train.rds 等三个文件。\n")
cat('下一步：source("R/03_features.R")\n')
