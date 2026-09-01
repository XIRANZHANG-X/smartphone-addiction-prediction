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

  # --- 发现 1：预算余量（四项残差）------------------------------------------
  # ⚠ 这一列曾经只减两项，那是错的。生成器强制的约束是**四项**：
  #     daily >= social + gaming + work_study   在 421,427 行上 100% 成立，
  #     且残差最小值恰为 0.000（三项版最小值 0.100 —— 说明它只是推论）。
  #
  # 为什么这一列树替代不了：一次分裂是一个特征加一个阈值。四项的边界要树
  # 用矩形去铺一张超平面，无论多深都到不了；直接给它就是一次分裂的事。
  # 实测证实了这一点 —— 它是唯一在 target encoding 之上仍然有效的派生特征
  #（+0.00064，5/5 折同号），而 max_bin 与小数位特征都被 TE 完全吸收了。
  #
  # 语义：work_study 是**在手机上做的**工作/学习，所以它在屏幕时间之内。
  dt[, other_screen := daily_screen_time_hours -
                       (social_media_hours + gaming_hours + work_study_hours)]

  # --- 发现 2：周末与日常的比值 ---------------------------------------------
  # 两列相关 +0.80，比值稳定在 1.33 左右（标准差 0.50）。
  dt[, weekend_ratio := safe_div(weekend_screen_time, daily_screen_time_hours)]

  # --- 屏幕时间的构成占比 ---------------------------------------------------
  # 绝对时长和占比携带的信息不同：每天刷 2 小时社交，
  # 对总屏幕 3 小时的人和总屏幕 10 小时的人，含义完全不一样。
  dt[, social_share := safe_div(social_media_hours, daily_screen_time_hours)]
  dt[, gaming_share := safe_div(gaming_hours,       daily_screen_time_hours)]

  # --- 发现 8（已更正）：空闲时间占比 ---------------------------------------
  # ⚠ 分母曾经是 24 - sleep - work，那把 work 减了两遍 —— 由发现 1 可知
  #   work 已经在 screen 里面。当初测出的 1.84% 负值是我们自己造出来的。
  #   去掉重复扣减后：24 - sleep 的最小值是 +3.99，是干净的硬约束。
  dt[, free_frac := safe_div(daily_screen_time_hours, 24 - sleep_hours)]

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
# 逐取值 target encoding（发现 10、11：生成器的格点）
# -----------------------------------------------------------------------------
# 为什么这不是普通的"类别特征编码"
# -----------------------------------------------------------------------------
# 这份数据是生成出来、并被舍入到一个格点上的（发现 11）。所以**精确取值**本身
# 携带信息，而且这个信息是**非单调**的：把取值当成一个量级去分裂，
# 捕捉不到"5.23 这个值恰好对应什么"。
#
# 样本内查找表的上界（R/17_discussion_checks.R 核查 6）：
#   daily_screen_time_hours   查找表 0.90197  vs  原始值 0.88955
#   weekend_screen_time       查找表 0.89523  vs  原始值 0.88099
# 差出来的就是原始值的单调排序捕捉不到的那部分。
#
# 折内实测（R/18_new_features.R）：+0.00391，Cohen's d = 6.57，5/5 折同号，
# 而安慰剂列是 −0.00010、1/5 同号。这是本项目**单项收益最大的一次改动**，
# 超过了四条插补线之间的全部差距（0.00297）。
#
# -----------------------------------------------------------------------------
# ⚠ 纪律：只能在训练折内部拟合
# -----------------------------------------------------------------------------
# 讨论区第 34 帖有一次现成的教训：有人在 CV 循环**之外**交叉拟合编码器，
# 以为"只要用同一套 k 折就没问题"。broccoli beef 推导出这是泄漏 ——
# 在第 k 次迭代里，X₋ₖ 中的每个样本都是用含 yₖ 的统计量编码的。
# 原作者改到折内之后**公榜大跌**，说明原来的 CV 是虚高的。
#
# 所以这两个函数被设计成 fit / apply 分离，与插补器同一个形状，
# 由 R/06_framework.R 在折内调用。不要在别处调 fit。
# =============================================================================

#' 参与 target encoding 的列
#'
#' 只对**数值**列做。类别列本来就是少数几个水平，树自己处理即可。
TE_COLS <- c("daily_screen_time_hours", "social_media_hours",
             "weekend_screen_time", "gaming_hours",
             "work_study_hours", "sleep_hours",
             "notifications_per_day", "app_opens_per_day")

#' 平滑强度：稀疏取值向全局均值收缩的力度
#'
#' (sum_y + prior*m) / (n + m)。m = 20 意味着一个只出现 20 次的取值，
#' 其编码有一半来自先验。取值数最多的列有 1437 个不同取值、69 万行，
#' 所以多数取值的 n 在数百量级，m=20 只影响长尾。
TE_SMOOTH <- 20

.te_table <- function(v, y, prior) {
  d <- data.table(v = v, y = y)[!is.na(v)]
  tb <- d[, .(s = sum(y), n = .N), by = v]
  tb[, enc := (s + prior * TE_SMOOTH) / (n + TE_SMOOTH)]
  tb[, .(v, enc)]
}

#' 在训练折上拟合编码器
#'
#' @param X 训练折的特征（data.table）
#' @param y 训练折的标签
#' @return 一个可以喂给 apply_target_encoder() 的对象
fit_target_encoder <- function(X, y) {
  prior <- mean(y)
  list(prior = prior,
       maps  = lapply(setNames(TE_COLS, TE_COLS),
                      function(cc) .te_table(X[[cc]], y, prior)))
}

#' 把编码器应用到任意一份数据上（原地加 te_* 列）
#'
#' 训练折里没出现过的取值、以及本来就缺失的取值，都回落到训练折的全局均值。
#' 让这两种情况共享同一个含义是有意的：对模型来说它们都是"这里没有信息"。
apply_target_encoder <- function(enc, dt) {
  stopifnot(is.data.table(dt))
  for (cc in names(enc$maps)) {
    e <- enc$maps[[cc]][data.table(v = dt[[cc]]), on = "v", x.enc]
    set(dt, j = paste0("te_", cc), value = fifelse(is.na(e), enc$prior, e))
  }
  dt[]
}

#' te_* 列名，供消融/重要性引用
TE_OUT_COLS <- paste0("te_", TE_COLS)

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
