# =============================================================================
# 03_features.R —— 特征工程（全组唯一真源）
#
# 用法：source("R/03_features.R")
# 产出：output/features_raw.rds   插补前的共享特征矩阵（含 NA）
#       derive_features()         供四条插补线在折内调用的派生特征函数
#
# -----------------------------------------------------------------------------
# 为什么特征要分两批
# -----------------------------------------------------------------------------
# 批一（本脚本直接产出，全组共享）：
#   原始 12 列。这些不依赖插补，可以安全地在折外一次性算好。
#
# 批二（derive_features()，各线在插补之后自己调）：
#   other_screen、weekend_ratio、social_share 等比值/差值特征
#   这些必须放在插补之后。原因很简单：它们全是几个列算出来的，
#   任一分量是 NA，结果就是 NA。插补前算等于白算。
#
# 因此本脚本导出的是「函数」而不是「算好的列」——函数共享保证口径一致，
# 调用时机由各线自己决定。
#
# 重要：不要往这个文件里加你自己的实验性特征。要加就加在你自己的
#       R/06_model_<你的名字>.R 里。效果好了再找组长升级到共享层。
# =============================================================================

library(data.table)

# -----------------------------------------------------------------------------
# 派生特征函数
# -----------------------------------------------------------------------------
#' 计算派生特征
#'
#' @param dt data.table，必须含有 9 个数值列。既可以是插补后的（无 NA），
#'   也可以是含 NA 的（L1 原生 NaN 线就是这么用的，派生列会自然带 NA，
#'   交给 xgboost/lightgbm 原生处理——这是该线的定义特征，不是缺陷）。
#' @return 原地修改并返回 dt（data.table 语义）
#'
derive_features <- function(dt) {
  stopifnot(is.data.table(dt))

  # 除零保护：屏幕时间最低分箱是 0–2 小时，确实存在接近 0 的值。
  # 分母为 0 时结果设为 NA 而不是 Inf —— Inf 会让 glmnet 直接报错，
  # 而 NA 至少能被树模型处理，也能被后续插补捕捉。
  safe_div <- function(num, den) {
    ifelse(is.na(num) | is.na(den) | den <= 0, NA_real_, num / den)
  }

  # --- 发现 1：屏幕时间的第三分量 -------------------------------------------
  # daily_screen >= social + gaming 在 451,246 行中 100% 成立，
  # 所以这一列在数据完整时保证 >= 0（均值 3.71，标准差 1.94）。
  # 它代表"既不是社交也不是游戏"的屏幕时间：浏览器、视频、办公等。
  dt[, other_screen := daily_screen_time_hours - social_media_hours - gaming_hours]

  # --- 发现 2：周末与日常的比值 ---------------------------------------------
  # 两列相关 +0.80，比值稳定在 1.33 左右（标准差 0.50）。
  dt[, weekend_ratio := safe_div(weekend_screen_time, daily_screen_time_hours)]

  # --- 屏幕时间的构成占比 ---------------------------------------------------
  # 绝对时长和占比携带的信息不同：每天刷 2 小时社交，
  # 对总屏幕 3 小时的人和总屏幕 10 小时的人，含义完全不一样。
  dt[, social_share := safe_div(social_media_hours, daily_screen_time_hours)]
  dt[, gaming_share := safe_div(gaming_hours,       daily_screen_time_hours)]

  # --- 发现 8：空闲时间占比（软特征） ---------------------------------------
  # 24 - 睡眠 - 工作 = 理论可支配时间。注意这不是硬约束：
  # 实测有 1.84% 的行算出来是负数，所以 safe_div 会把它们变成 NA。
  dt[, free_frac := safe_div(daily_screen_time_hours,
                             24 - sleep_hours - work_study_hours)]

  # --- 已移除：screen_social = 屏幕时间 + 社交时间 --------------------------
  # 由发现 1 可知 social 本身就是 screen 的一个分量，二者相加等价于
  # 2*social + gaming + other —— 社交被算了两遍，没有任何理由让权重停在 1
  # （扫描 screen + w*social，最优 w = 1.75 而非 1）。
  # 它与自身分量回归 R^2 = 1.000000，是精确线性组合，零新信息：
  # 对树模型只是冗余，对 glmnet 造成设计矩阵秩亏。
  # 剔除依据的完整检验见 R/16_screensocial_audit.R。

  dt[]
}

# 派生特征的列名，供下游做特征选择/消融时引用
DERIVED_COLS <- c("other_screen", "weekend_ratio", "social_share",
                  "gaming_share", "free_frac")

# -----------------------------------------------------------------------------
# 构建批一：features_raw
# -----------------------------------------------------------------------------
# 这一段在 source() 时执行。如果产物已存在就跳过，
# 这样各线 source 本文件拿 derive_features() 时不会反复重算。
# 需要强制重建时：FORCE_REBUILD <- TRUE; source("R/03_features.R")

.build_features_raw <- function(with_na_ind = FALSE) {
  dir_out <- "output"
  f_train <- file.path(dir_out, "raw_train.rds")
  f_test  <- file.path(dir_out, "raw_test.rds")

  if (!file.exists(f_train) || !file.exists(f_test)) {
    stop("找不到 output/raw_train.rds，请先运行 source(\"R/01_load.R\")")
  }

  train <- readRDS(f_train)
  test  <- readRDS(f_test)

  feat_cols <- c(
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
    "gender", "stress_level", "academic_work_impact"
  )

  # 训练集和测试集垂直拼接后统一处理，保证两边的列完全一致。
  # is_train 标记用于事后拆回。
  train[, is_train := 1L]
  test[,  is_train := 0L]
  test[,  addicted_label := NA_integer_]

  all_dt <- rbindlist(list(train, test), use.names = TRUE, fill = FALSE)

  # --- 缺失指示（默认不生成）------------------------------------------------
  # 曾经默认生成 12 个 is_na_<col> 加 1 个 n_missing，共 13 列。现已移除：
  #
  #   缺失机制已量化确认为 MCAR（见 docs/项目说明.md 发现 6）：
  #   12 列的 missing/present 成瘾率差全部 < 0.0042，缺失与标签无关。
  #   消融实验证实删掉这 13 列 AUC 变化 +0.00003（p = 0.448），
  #   置换重要性全部 <= 0.00001，其中 11 列为负。
  #
  #   一句话：它们是为了检验「缺失本身是否携带信息」而造的诊断变量，
  #   检验结论是否定的，因此不进入模型特征集。
  #
  # 复现那次消融时才需要它们：
  #   WITH_NA_INDICATORS <- TRUE; FORCE_REBUILD <- TRUE; source("R/03_features.R")
  if (with_na_ind) {
    for (col in feat_cols) {
      set(all_dt, j = paste0("is_na_", col), value = as.integer(is.na(all_dt[[col]])))
    }
    na_cols <- paste0("is_na_", feat_cols)
    all_dt[, n_missing := rowSums(.SD), .SDcols = na_cols]
  } else {
    na_cols <- character(0)
  }

  saveRDS(all_dt, file.path(dir_out, "features_raw.rds"))

  # --- 报告 -----------------------------------------------------------------
  cat("---- features_raw 已构建 ----\n")
  cat(sprintf("总行数   %s（训练 %s + 测试 %s）\n",
              format(nrow(all_dt), big.mark = ","),
              format(sum(all_dt$is_train == 1L), big.mark = ","),
              format(sum(all_dt$is_train == 0L), big.mark = ",")))
  cat(sprintf("总列数   %d\n", ncol(all_dt)))
  cat(sprintf("  原始特征 %d + 缺失指示 %d + id/target/is_train 3\n",
              length(feat_cols), length(na_cols) + as.integer(with_na_ind)))
  cat(sprintf("  插补后再加 %d 个派生特征 -> 建模特征共 %d 个\n",
              length(DERIVED_COLS),
              length(feat_cols) + length(na_cols) + as.integer(with_na_ind) +
                length(DERIVED_COLS)))

  cat("\n每行缺失个数的分布：\n")
  n_miss_vec <- rowSums(is.na(all_dt[, ..feat_cols]))
  tb <- table(n_miss_vec)
  for (k in names(tb)) {
    cat(sprintf("  缺 %2s 个：%8s 行 (%5.2f%%)\n",
                k, format(tb[[k]], big.mark = ","), 100 * tb[[k]] / nrow(all_dt)))
  }

  invisible(all_dt)
}

# 执行
if (!exists("FORCE_REBUILD"))     FORCE_REBUILD     <- FALSE
if (!exists("WITH_NA_INDICATORS")) WITH_NA_INDICATORS <- FALSE
if (FORCE_REBUILD || !file.exists(file.path("output", "features_raw.rds"))) {
  .build_features_raw(with_na_ind = WITH_NA_INDICATORS)
  cat('\n下一步：source("R/04_folds.R")\n')
} else {
  cat("features_raw.rds 已存在，跳过构建。\n")
  cat("（需要强制重建：FORCE_REBUILD <- TRUE 后重新 source）\n")
}
