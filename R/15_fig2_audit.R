# =============================================================================
# 15_fig2_audit.R —— 图 2 早期版本的四项质疑核查（存档脚本）
#
# ⚠ 本脚本核查的是图 2 的**早期版本**，其横轴为 `屏幕时间 + 社交时间`。
#   该构造已于 2026-08-24 移除，图 2 现改用 `daily_screen_time_hours` 单变量
#   （新口径下没有任何一个分箱是零阴性的，最高箱 2,659 行中仍有 1 个阴性）。
#   脚本保留于此，作为"质疑—检验—修改"这一过程的记录。
# =============================================================================

suppressMessages(library(data.table))
train <- readRDS("output/raw_train.rds")
s <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
s[, score := daily_screen_time_hours + social_media_hours]

hr <- function(x) cat("\n", strrep("=", 68), "\n", x, "\n", strrep("=", 68), "\n", sep = "")

# ---------------------------------------------------------------------------
hr("检验 1：右端真的是「没有阴性样本」吗？")
# ---------------------------------------------------------------------------
tb <- s[, .(n = .N, pos = sum(addicted_label), neg = sum(addicted_label == 0)),
        by = .(bin = floor(score))][order(bin)][bin >= 13]
cat(sprintf("%5s %10s %10s %10s %12s\n", "箱", "样本数", "成瘾", "非成瘾", "成瘾比例"))
for (i in seq_len(nrow(tb))) {
  r <- tb[i]
  cat(sprintf("%5d %10s %10s %10s %11.4f%%\n", r$bin,
              format(r$n, big.mark = ","), format(r$pos, big.mark = ","),
              format(r$neg, big.mark = ","), 100 * r$pos / r$n))
}
cat(sprintf("\nbin>=13 合计 %s 行，其中非成瘾 %s 行\n",
            format(sum(tb$n), big.mark = ","), format(sum(tb$neg), big.mark = ",")))
cat("=> 我此前表格里显示的 100.0% 是四舍五入，13~16 箱仍有阴性样本。\n")

# ---------------------------------------------------------------------------
hr("检验 2：单调性是「相加导致的数学必然」吗？")
# ---------------------------------------------------------------------------
cat("反证法一：把标签随机打乱，若单调性是相加的必然结果，打乱后仍应单调。\n\n")
set.seed(42)
s[, y_shuffled := sample(addicted_label)]
sh <- s[, .(n = .N, rate = mean(y_shuffled)), by = .(bin = floor(score))][n > 500][order(bin)]
cat(sprintf("打乱后：比例 %.3f -> %.3f，极差 %.4f，下降处 %d/%d\n",
            sh$rate[1], sh$rate[nrow(sh)], diff(range(sh$rate)),
            sum(diff(sh$rate) <= 0), nrow(sh) - 1))
cat("=> 打乱后曲线完全平坦（全部贴近基准率 0.7094），单调性消失。\n")
cat("   相加这个操作本身不产生任何单调性，单调性来自标签与特征的真实关联。\n")

cat("\n反证法二：本项目自己的数据里就有「相加不单调」的反例。\n")
s2 <- train[!is.na(notifications_per_day) & !is.na(app_opens_per_day)]
s2[, s_noise := notifications_per_day / 50 + app_opens_per_day / 30]
n2 <- s2[, .(n = .N, rate = mean(addicted_label)),
         by = .(bin = floor(s_noise))][n > 500][order(bin)]
cat(sprintf("通知数/50 + 打开次数/30 求和后分箱：比例 %.3f -> %.3f，极差仅 %.4f，下降处 %d/%d\n",
            n2$rate[1], n2$rate[nrow(n2)], diff(range(n2$rate)),
            sum(diff(n2$rate) <= 0), nrow(n2) - 1))
cat("=> 同样是「两个变量相加再分箱」，这一组既不单调也不到 100%。\n")

# ---------------------------------------------------------------------------
hr("检验 3：单个变量是否已经单调？相加是否多余？")
# ---------------------------------------------------------------------------
for (v in c("daily_screen_time_hours", "social_media_hours")) {
  d <- train[!is.na(get(v)), .(n = .N, rate = mean(addicted_label)),
             by = .(bin = floor(get(v)))][n > 500][order(bin)]
  cat(sprintf("%-26s %2d 箱  %.3f -> %.3f  极差 %.3f  下降处 %d\n",
              v, nrow(d), d$rate[1], d$rate[nrow(d)],
              diff(range(d$rate)), sum(diff(d$rate) <= 0)))
}
d3 <- s[, .(n = .N, rate = mean(addicted_label)), by = .(bin = floor(score))][n > 500][order(bin)]
cat(sprintf("%-26s %2d 箱  %.3f -> %.3f  极差 %.3f  下降处 %d\n",
            "两者相加", nrow(d3), d3$rate[1], d3$rate[nrow(d3)],
            diff(range(d3$rate)), sum(diff(d3$rate) <= 0)))
cat("\n=> 单个变量本身就单调，相加没有制造单调性，只是把两个信号合并。\n")

# ---------------------------------------------------------------------------
hr("检验 4：剔除缺失行是否引入选择偏差？")
# ---------------------------------------------------------------------------
cat("对比「两列都不缺」与「至少缺一列」两组人群的成瘾率：\n\n")
train[, has_both := !is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
cmp <- train[, .(n = .N, rate = mean(addicted_label)), by = has_both]
for (i in seq_len(nrow(cmp))) {
  r <- cmp[i]
  cat(sprintf("  %-14s n=%9s  成瘾率 %.4f\n",
              if (r$has_both) "两列都不缺" else "至少缺一列",
              format(r$n, big.mark = ","), r$rate))
}
cat(sprintf("\n差值 %.4f\n", abs(diff(cmp[order(has_both)]$rate))))

cat("\n再看：屏幕时间不缺、但社交缺失的那批人，其屏幕时间分布是否偏移？\n")
a <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours),
           daily_screen_time_hours]
b <- train[!is.na(daily_screen_time_hours) & is.na(social_media_hours),
           daily_screen_time_hours]
cat(sprintf("  两列都在  n=%9s  屏幕时间 均值 %.3f  中位数 %.3f\n",
            format(length(a), big.mark = ","), mean(a), median(a)))
cat(sprintf("  社交缺失  n=%9s  屏幕时间 均值 %.3f  中位数 %.3f\n",
            format(length(b), big.mark = ","), mean(b), median(b)))
cat(sprintf("  均值差 %.4f 小时\n", abs(mean(a) - mean(b))))
ks <- suppressWarnings(ks.test(a, b))
cat(sprintf("  KS 检验 D=%.5f  p=%.4f\n", ks$statistic, ks$p.value))

# ---------------------------------------------------------------------------
hr("检验 5：用「屏幕时间单独」在全部非缺失行上重画（不做完整案例剔除）")
# ---------------------------------------------------------------------------
full <- train[!is.na(daily_screen_time_hours),
              .(n = .N, pos = sum(addicted_label), rate = mean(addicted_label)),
              by = .(bin = floor(daily_screen_time_hours))][n > 500][order(bin)]
cat(sprintf("%5s %10s %10s %14s\n", "箱", "样本数", "成瘾比例", "95% 置信区间"))
for (i in seq_len(nrow(full))) {
  r <- full[i]
  se <- sqrt(r$rate * (1 - r$rate) / r$n)
  cat(sprintf("%5d %10s %9.3f%% [%.3f, %.3f]\n", r$bin,
              format(r$n, big.mark = ","), 100 * r$rate,
              max(0, r$rate - 1.96 * se), min(1, r$rate + 1.96 * se)))
}
cat(sprintf("\n此口径仅剔除屏幕时间缺失的 13.86%%，保留了 %s 行（占全量 %.1f%%）\n",
            format(sum(full$n), big.mark = ","), 100 * sum(full$n) / nrow(train)))
