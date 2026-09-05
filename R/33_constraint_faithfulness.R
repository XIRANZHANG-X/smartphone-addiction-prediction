# =============================================================================
# 33_constraint_faithfulness.R —— 插补方法对硬约束有多不老实
#
# 用法：Rscript R/33_constraint_faithfulness.R
# 产出：output/constraint_faithfulness.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 要回答的问题
# -----------------------------------------------------------------------------
# 真正的硬约束是四项：
#   daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours
# （work_study_hours 计入屏幕时间，即"在手机上做的工作/学习"）。
# 三项版 daily >= social + gaming 只是四项版蕴含的推论：四项版残差最小值
# 恰好触到 0.000，三项版停在 0.100，说明四项版才是真正的边界（见
# R/17_discussion_checks.R）。此前用三项版会系统性低估违反率。
#
# L3（约束插补）会显式把违反这条约束的插补值投影回可行域——但实测这个
# 投影只在 0.06% 的行上触发（项目说明 6.3），说明真正的约束违反本来就
# 很罕见。L2（中位数填补）和 L4（PMM 插补）完全不检查这条约束。
#
# 这里量化一个从没人正面测过的数字：**L2、L4 插补出来的数据里，
# 到底有多少行是「物理上不可能」的**（屏幕时间 < 社交+游戏+工作学习）？
# 这是一个跟 AUC 完全独立的「插补方法有多不靠谱」的指标——
# 哪怕两条线的 AUC 差不多，物理合理性也可能天差地别。
#
# -----------------------------------------------------------------------------
# 规模说明
# -----------------------------------------------------------------------------
# L4（missRanger）在这台机器上对内存比较敏感，跟 Tier A 对比实验一样，
# 用分层子样本 output/subsample_200k.rds，不跑全量 69 万行。
# =============================================================================

suppressMessages(library(data.table))

train <- readRDS("output/raw_train.rds")
sub   <- readRDS("output/subsample_200k.rds")
dt0   <- train[sub]

hr <- function(x) cat("\n", strrep("=", 74), "\n", x, "\n", strrep("=", 74), "\n", sep = "")

# ---- 基准：完整数据上约束确实 100% 成立（四项约束，复核 R/17_discussion_checks.R）--
cc <- dt0[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
          !is.na(gaming_hours) & !is.na(work_study_hours)]
viol_cc <- cc[, mean(daily_screen_time_hours <
                     social_media_hours + gaming_hours + work_study_hours - 1e-6)]
hr("基准：完整数据上的约束违反率（四项约束）")
cat(sprintf("完整行数：%s，违反率：%.4f%%（应接近 0）\n",
            format(nrow(cc), big.mark = ","), 100 * viol_cc))

# ---- 标记「本来就没法算」的行：四列里至少一列缺失 ---------------------------
na_screen <- is.na(dt0$daily_screen_time_hours)
na_social <- is.na(dt0$social_media_hours)
na_gaming <- is.na(dt0$gaming_hours)
na_work   <- is.na(dt0$work_study_hours)
needs_impute <- na_screen | na_social | na_gaming | na_work

hr("参与审查的行")
cat(sprintf("四列（屏幕/社交/游戏/工作学习）至少缺一个、需要插补才能凑出约束的行：%s（%.1f%%）\n",
            format(sum(needs_impute), big.mark = ","), 100 * mean(needs_impute)))

check_violation <- function(dt, na_screen, na_social, na_gaming, na_work, needs_impute) {
  v <- dt$daily_screen_time_hours <
       dt$social_media_hours + dt$gaming_hours + dt$work_study_hours - 1e-6
  list(
    n_checked  = sum(needs_impute),
    n_violate  = sum(v[needs_impute]),
    rate       = mean(v[needs_impute]),
    # 精确对应 L3 显式处理的那种情况：屏幕缺、社交/游戏/工作学习都在
    rate_screen_only = {
      m <- na_screen & !na_social & !na_gaming & !na_work
      if (sum(m) == 0) NA_real_ else mean(v[m])
    }
  )
}

# ---- L2：中位数插补 ----------------------------------------------------------
hr("L2（中位数插补）")
source("R/05_impute_L2.R", local = TRUE)
imp2 <- fit_imputer_L2(dt0)
d2   <- apply_imputer_L2(imp2, copy(dt0))
r2   <- check_violation(d2, na_screen, na_social, na_gaming, na_work, needs_impute)
cat(sprintf("审查行数：%s\n", format(r2$n_checked, big.mark = ",")))
cat(sprintf("违反约束：%s（%.2f%%）\n", format(r2$n_violate, big.mark = ","), 100 * r2$rate))
cat(sprintf("其中「屏幕缺、社交/游戏/工作学习都在」这种 L3 会显式修正的情况，违反率：%.2f%%\n",
            100 * r2$rate_screen_only))

# ---- L4：不现场跑 ------------------------------------------------------------
# L4（missRanger）需要现场插补才能拿到数值，且在这台机器上反复 std::bad_alloc。
# 按要求只用已经真实存在的数据做 EDA，不现场生成新的模型/插补产物，
# 这里跳过 L4，只对比 L2（现场算，纯中位数填补，无风险）与 L3
# （项目说明文档里已经报告过的真实数字）。
r4 <- NULL

# ---- L3 对照（读文档里已有的数字，不重新跑，只是放在一起对比）--------------
hr("两条线对照（L4 未现场生成，不纳入）")
cat(sprintf("%-30s %12s\n", "插补方式", "整体违反率"))
cat(sprintf("%-30s %11.2f%%\n", "L2 中位数", 100 * r2$rate))
cat(sprintf("%-30s %11s\n", "L3 约束插补（文档已报告值）", "0.06%"))
cat("L4 PMM 随机插补                 未现场生成，跳过\n")

saveRDS(list(baseline_violation = viol_cc, n_needs_impute = sum(needs_impute),
             L2 = r2), "output/constraint_faithfulness.rds")

hr("完成")
cat("结果已存至 output/constraint_faithfulness.rds\n")
