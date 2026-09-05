# =============================================================================
# 32_residual_exceptions.R —— 谁打破了近乎确定的规律
#
# 用法：Rscript R/32_residual_exceptions.R
# 产出：output/residual_exceptions.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 要回答的问题
# -----------------------------------------------------------------------------
# 发现 5：按 daily_screen_time_hours 分箱后，成瘾比例几乎单调（图 2）。
# 发现 9：notifications_per_day / app_opens_per_day 只在标签「未饱和」的
#         中间箱里才有干净的单调效应，高低两端因天花板效应而失效。
#
# 这两条发现从没被直接拼在一起问过一个问题：
#   在 daily_screen_time_hours 已经把大多数人分到「几乎确定成瘾」或
#   「几乎确定不成瘾」之后，那一小撮**不跟大多数人走**的例外行，
#   是不是恰好可以用 notifications / app_opens 解释？
#
# 口径对齐 R/12_figures.R 图 2：只用 daily_screen_time_hours 单变量
# （screen_social 已因重复计数于 2026-08-24 从特征集删除，见
#  R/16_screensocial_audit.R），floor() 分箱，剔除 n<=500 的箱。
# =============================================================================

suppressMessages(library(data.table))

train <- readRDS("output/raw_train.rds")

hr <- function(x) cat("\n", strrep("=", 74), "\n", x, "\n", strrep("=", 74), "\n", sep = "")

# ---- 第一步：按图 2 的口径分箱，算出每箱的「多数标签」-----------------------
one <- train[!is.na(daily_screen_time_hours),
             .(daily_screen_time_hours, addicted_label,
               notifications_per_day, app_opens_per_day)]
one[, bin := floor(daily_screen_time_hours)]

bin_stat <- one[, .(n = .N, rate = mean(addicted_label)), by = bin][n > 500][order(bin)]
one <- merge(one, bin_stat[, .(bin, rate, n_bin = n)], by = "bin")

# 多数标签：该箱成瘾比例 >= 0.5 就是「大概率成瘾」，反之「大概率不成瘾」
one[, majority_label := as.integer(rate >= 0.5)]
one[, is_exception   := addicted_label != majority_label]

hr("第一步：整体例外率")
cat(sprintf("参与分析的行数（daily_screen_time_hours 不缺，且落在 n>500 的箱内）：%s\n",
            format(nrow(one), big.mark = ",")))
cat(sprintf("其中「不跟大多数人走」的例外行：%s（%.2f%%）\n",
            format(sum(one$is_exception), big.mark = ","),
            100 * mean(one$is_exception)))

hr("第二步：例外率在各箱的分布（两端箱子例外率应该更低——那才叫「近乎确定」）")
tab <- one[, .(n = .N, n_exception = sum(is_exception),
               exc_rate = mean(is_exception)), by = bin][order(bin)]
for (i in seq_len(nrow(tab))) {
  r <- tab[i]
  cat(sprintf("  第 %2d 箱  n=%-7s 例外 %-6s (%.2f%%)\n",
              r$bin, format(r$n, big.mark = ","),
              format(r$n_exception, big.mark = ","), 100 * r$exc_rate))
}

# ---- 第三步：控制 bin 之后，notif / app_opens 还能解释多少残差 --------------
# 限制在 notifications_per_day、app_opens_per_day 也不缺的行，两个模型对比：
#   模型 A：label ~ factor(bin)                         —— 只用屏幕时间分箱
#   模型 B：label ~ factor(bin) + notif + app_opens      —— 再加这两个特征
# AUC 从 A 到 B 的提升，就是「在屏幕时间已经解释的部分之外，这两个特征
# 还能解释多少」——直接回答「谁打破了规律」这个问题。
two <- one[!is.na(notifications_per_day) & !is.na(app_opens_per_day)]

fast_auc <- function(y, p) {
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

two[, bin_f := factor(bin)]
mA <- glm(addicted_label ~ bin_f, data = two, family = binomial())
mB <- glm(addicted_label ~ bin_f + notifications_per_day + app_opens_per_day,
          data = two, family = binomial())

aucA <- fast_auc(two$addicted_label, predict(mA, type = "response"))
aucB <- fast_auc(two$addicted_label, predict(mB, type = "response"))

cf <- summary(mB)$coefficients
b_notif <- cf["notifications_per_day", ]
b_app   <- cf["app_opens_per_day", ]

hr("第三步：控制屏幕时间分箱后，notif / app_opens 解释了多少残差")
cat(sprintf("参与拟合的行数：%s\n", format(nrow(two), big.mark = ",")))
cat(sprintf("模型 A（仅 bin）           AUC = %.5f\n", aucA))
cat(sprintf("模型 B（bin + 两个特征）   AUC = %.5f\n", aucB))
cat(sprintf("提升                       %+.5f\n\n", aucB - aucA))
cat(sprintf("notifications_per_day  系数 = %+.5f   p = %.2e   （%s）\n",
            b_notif["Estimate"], b_notif["Pr(>|z|)"],
            if (b_notif["Estimate"] < 0) "越多通知，越不容易是「意外成瘾」——方向与发现 9 一致"
            else "方向与发现 9 相反"))
cat(sprintf("app_opens_per_day      系数 = %+.5f   p = %.2e   （%s）\n",
            b_app["Estimate"], b_app["Pr(>|z|)"],
            if (b_app["Estimate"] > 0) "越多打开次数，越容易是「意外成瘾」——方向与发现 9 一致"
            else "方向与发现 9 相反"))

# ---- 第四步：具体挑两个「两端」箱，看例外行到底长什么样 --------------------
# 两端箱子的例外最不寻常：大概率不成瘾的箱里那个反而成瘾的人，
# 大概率成瘾的箱里那个反而没成瘾的人，到底 notif/app_opens 有没有不一样。
hr("第四步：两端箱子里，例外行 vs 多数行的 notif / app_opens 对比")

pick_bins <- tab[n >= 2000][order(bin)]
lo_bin <- pick_bins$bin[1]                      # 最低的大样本箱（多数=不成瘾）
hi_bin <- pick_bins$bin[nrow(pick_bins)]         # 最高的大样本箱（多数=成瘾）

describe_bin <- function(b) {
  d <- two[bin == b]
  maj <- d[is_exception == FALSE]
  exc <- d[is_exception == TRUE]
  cat(sprintf("\n第 %d 箱（该箱多数标签 = %s，n=%s，例外 n=%s）：\n",
              b, if (unique(d$majority_label) == 1) "成瘾" else "不成瘾",
              format(nrow(d), big.mark = ","), nrow(exc)))
  if (nrow(exc) < 5) { cat("  例外样本太少，跳过统计检验\n"); return(invisible(NULL)) }
  for (col in c("notifications_per_day", "app_opens_per_day")) {
    tt <- t.test(exc[[col]], maj[[col]])
    cat(sprintf("  %-24s 多数行均值 %.2f   例外行均值 %.2f   差值 %+.2f   p=%.4f\n",
                col, mean(maj[[col]]), mean(exc[[col]]), tt$estimate[1] - tt$estimate[2], tt$p.value))
  }
}
describe_bin(lo_bin)
describe_bin(hi_bin)

# ---- 存盘 -------------------------------------------------------------------
saveRDS(list(
  bin_table       = tab,
  model_A         = mA,
  model_B         = mB,
  auc_A           = aucA,
  auc_B           = aucB,
  auc_gain        = aucB - aucA,
  coef_notif      = b_notif,
  coef_app_opens  = b_app,
  lo_bin          = lo_bin,
  hi_bin          = hi_bin
), "output/residual_exceptions.rds")

hr("完成")
cat("结果已存至 output/residual_exceptions.rds\n")
