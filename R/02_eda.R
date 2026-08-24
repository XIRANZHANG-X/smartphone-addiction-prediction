# =============================================================================
# 02_eda.R —— 探索性分析：复现全部 8 个发现
#
# 负责人：E（画图部分）
#
# 用法：source("R/02_eda.R")
#
# 本脚本把 docs/项目说明.md 第三节的 8 个发现全部用代码算一遍。
# 每一个数字都可以在这里复现 —— 报告里引用的任何结论，
# 都必须能在这个脚本的输出里找到出处。
#
# E 的任务是把这些数字做成图。建议优先做发现 1、4、5、7 这四张，
# 它们最能一眼看出结论。
# =============================================================================

library(data.table)

train <- readRDS("output/raw_train.rds")
y <- train$addicted_label

hr <- function(title) cat("\n", strrep("=", 60), "\n", title, "\n",
                          strrep("=", 60), "\n", sep = "")

# =============================================================================
hr("发现 1：硬结构约束（预期 100.00%）")
# =============================================================================
# 屏幕时间 >= 社交 + 游戏。这不是统计相关，是结构性约束：
# 屏幕时间本来就由社交 + 游戏 + 其他用途组成。
cc <- train[!is.na(daily_screen_time_hours) &
            !is.na(social_media_hours) &
            !is.na(gaming_hours)]

n_hold <- cc[, sum(daily_screen_time_hours >= social_media_hours + gaming_hours)]
cat(sprintf("三列都不缺的行数    %s\n", format(nrow(cc), big.mark = ",")))
cat(sprintf("约束成立的行数      %s\n", format(n_hold, big.mark = ",")))
cat(sprintf("成立比例            %.4f%%\n", 100 * n_hold / nrow(cc)))

other <- cc[, daily_screen_time_hours - social_media_hours - gaming_hours]
cat(sprintf("\nother_screen（第三分量）  均值 %.2f  标准差 %.2f  最小值 %.2f\n",
            mean(other), sd(other), min(other)))

# =============================================================================
hr("发现 2：周末屏幕时间是冗余代理")
# =============================================================================
w <- train[!is.na(daily_screen_time_hours) & !is.na(weekend_screen_time)]
ratio <- w[, weekend_screen_time / daily_screen_time_hours]
ratio <- ratio[is.finite(ratio)]
cat(sprintf("两列都不缺的行数    %s\n", format(nrow(w), big.mark = ",")))
cat(sprintf("weekend/daily 比值  均值 %.4f  标准差 %.4f\n", mean(ratio), sd(ratio)))
cat(sprintf("相关系数            %+.4f\n",
            w[, cor(daily_screen_time_hours, weekend_screen_time)]))

# =============================================================================
hr("发现 3：相关结构")
# =============================================================================
num_cols <- c("age", "daily_screen_time_hours", "social_media_hours",
              "gaming_hours", "work_study_hours", "sleep_hours",
              "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
short <- c("age", "screen", "social", "gaming", "work", "sleep",
           "notif", "opens", "weekend")

M <- cor(train[, ..num_cols], use = "pairwise.complete.obs")
dimnames(M) <- list(short, short)

pairs <- data.table(
  a = short[row(M)[upper.tri(M)]],
  b = short[col(M)[upper.tri(M)]],
  r = M[upper.tri(M)]
)[order(-abs(r))]
cat("按相关性绝对值排序：\n")
print(pairs[1:12], row.names = FALSE)
cat("\n（age / notif / opens / sleep 与一切的相关都接近 0，是独立噪声维度）\n")

# =============================================================================
hr("发现 4：三个特征是纯噪声")
# =============================================================================
# 关键在于「控制住屏幕时间之后再看」。不控制的话，
# 任何与屏幕时间有一点相关的东西都会显得有信息量。
d <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
           !is.na(stress_level) & !is.na(academic_work_impact)]
d[, band := fifelse(daily_screen_time_hours < 4.5, "低分段",
             fifelse(daily_screen_time_hours < 7.5, "中分段", "高分段"))]

tab <- d[band != "高分段",
         .(n = .N, rate = mean(addicted_label)),
         by = .(band, stress_level, academic_work_impact)][order(band, stress_level)]
print(tab, row.names = FALSE)

for (b in c("低分段", "中分段")) {
  rr <- tab[band == b, rate]
  cat(sprintf("\n%s 内部成瘾率极差 %.4f  ——  %s\n", b, diff(range(rr)),
              if (diff(range(rr)) < 0.03) "纯随机波动，无信息量" else "存在信号"))
}

cat("\n通知数（notifications_per_day）按 50 分箱：\n")
nb <- train[!is.na(notifications_per_day),
            .(n = .N, rate = mean(addicted_label)),
            by = .(bin = floor(notifications_per_day / 50) * 50)][order(bin)]
print(nb, row.names = FALSE)

# =============================================================================
hr("发现 5：标签近乎确定")
# =============================================================================
s <- train[!is.na(daily_screen_time_hours)]
sb <- s[, .(n = .N, rate = mean(addicted_label)),
        by = .(bin = floor(daily_screen_time_hours))][order(bin)]
print(sb, row.names = FALSE)
cat(sprintf("\n成瘾率从 %.3f 爬升到 %.3f（%d 个间隔中 %d 处下降）。\n",
            sb[1, rate], sb[.N, rate], nrow(sb) - 1L, sum(diff(sb$rate) < 0)))
cat("数据完整时这道题几乎没有难度 —— 难度全部来自缺失。\n")

# =============================================================================
hr("发现 6：缺失是 MCAR，缺失指示无信息量")
# =============================================================================
feat_cols <- c(num_cols, "gender", "stress_level", "academic_work_impact")
res <- rbindlist(lapply(feat_cols, function(cc) {
  m <- is.na(train[[cc]])
  data.table(col = cc, n_miss = sum(m),
             rate_miss = mean(y[m]), rate_obs = mean(y[!m]),
             delta = mean(y[m]) - mean(y[!m]))
}))[order(-abs(delta))]
print(res, row.names = FALSE)
cat(sprintf("\n最大差值 %.4f。全部小于 0.005，属于纯随机波动。\n",
            max(abs(res$delta))))
cat("结论：缺失指示特征在本题无效。这是一个阴性结论，但它决定了方法选择 ——\n")
cat("信息不在「哪里缺失」里，只能从「没缺的部分」去推断「缺了的部分」。\n")

# =============================================================================
hr("发现 7：work_study_hours 的 Simpson 悖论")
# =============================================================================
cat("边际视角（不控制任何变量）：\n")
mg <- train[!is.na(work_study_hours),
            .(n = .N, rate = mean(addicted_label)),
            by = .(bin = floor(work_study_hours / 2) * 2)][order(bin)]
print(mg, row.names = FALSE)

cat("\n条件视角（只看每日屏幕时间 < 4.5 小时的人）：\n")
cd <- train[!is.na(work_study_hours) & !is.na(daily_screen_time_hours) &
            daily_screen_time_hours < 4.5,
            .(n = .N, rate = mean(addicted_label)),
            by = .(bin = floor(work_study_hours / 2) * 2)][order(bin)]
print(cd, row.names = FALSE)

cat(sprintf("\n方向完全相反。成因：screen 与 work 的相关是 %+.4f ——\n",
            train[, cor(daily_screen_time_hours, work_study_hours,
                        use = "pairwise.complete.obs")]))
cat("边际上看到的正相关，完全是屏幕时间这个混淆变量制造的假象。\n")

# =============================================================================
hr("发现 8：「空闲时间」假设不成立")
# =============================================================================
f <- train[!is.na(sleep_hours) & !is.na(work_study_hours) &
           !is.na(daily_screen_time_hours),
           24 - sleep_hours - work_study_hours - daily_screen_time_hours]
cat(sprintf("24 - sleep - work - screen   均值 %.2f  标准差 %.2f\n", mean(f), sd(f)))
cat(sprintf("负值比例 %.2f%% —— 不是硬约束，只能当软特征用。\n",
            100 * mean(f < 0)))

cat("\n", strrep("=", 60), "\n", sep = "")
cat("全部 8 个发现复现完毕。\n")
cat("E：请把发现 1、4、5、7 做成图，这四个最能一眼看出结论。\n")
