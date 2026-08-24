suppressMessages(library(data.table))
train <- readRDS("output/raw_train.rds")
s <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
s[, score := daily_screen_time_hours + social_media_hours]

cat("========== 第一步：原始的连续取值长什么样 ==========\n")
set.seed(3)
ex <- s[sample(.N, 8), .(daily_screen_time_hours, social_media_hours,
                         score, addicted_label)]
ex[, 落入第几箱 := floor(score)]
print(ex, row.names = FALSE)
cat(sprintf("\n共 %s 行，score 取值 %.2f ~ %.2f，几乎每行都不一样\n",
            format(nrow(s), big.mark = ","), min(s$score), max(s$score)))
cat(sprintf("不同取值个数：%s —— 不分箱的话没法统计\n",
            format(uniqueN(s$score), big.mark = ",")))

cat("\n========== 第二步：floor() 归箱后的完整结果 ==========\n")
sb <- s[, .(n = .N, n_addicted = sum(addicted_label),
            rate = mean(addicted_label)),
        by = .(bin = floor(score))][order(bin)]
cat(sprintf("%6s %10s %12s %12s %10s\n",
            "箱", "区间", "样本数", "其中成瘾", "成瘾比例"))
for (i in seq_len(nrow(sb))) {
  r <- sb[i]
  mark <- if (r$n <= 500) "   <- n<=500，图中已剔除" else ""
  cat(sprintf("%6d  [%2d, %2d)  %12s %12s %9.1f%%%s\n",
              r$bin, r$bin, r$bin + 1,
              format(r$n, big.mark = ","),
              format(r$n_addicted, big.mark = ","),
              100 * r$rate, mark))
}

cat("\n========== 第三步：为什么要剔除小样本箱 ==========\n")
small <- sb[n <= 500]
if (nrow(small)) {
  cat("被剔除的箱：\n")
  for (i in seq_len(nrow(small))) {
    r <- small[i]
    se <- sqrt(r$rate * (1 - r$rate) / r$n)
    cat(sprintf("  第 %2d 箱  n=%-5s 成瘾比例 %.1f%%  标准误 ±%.1f%%\n",
                r$bin, r$n, 100 * r$rate, 100 * se))
  }
  cat("样本太少时比例估计不稳，一两个样本就能让点上下跳动。\n")
}

cat("\n========== 第四步：箱宽的取舍 ==========\n")
for (w in c(0.5, 1, 2, 4)) {
  t <- s[, .(n = .N, rate = mean(addicted_label)),
         by = .(b = floor(score / w) * w)][n > 500][order(b)]
  # 单调性检验：相邻箱是否始终递增
  d <- diff(t$rate)
  cat(sprintf("箱宽 %.1f 小时：%2d 个箱，比例 %.3f -> %.3f，%s\n",
              w, nrow(t), t$rate[1], t$rate[nrow(t)],
              if (all(d > 0)) "全程单调递增" else
                sprintf("有 %d 处下降", sum(d <= 0))))
}
