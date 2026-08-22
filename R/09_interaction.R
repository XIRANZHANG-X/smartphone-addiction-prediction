# =============================================================================
# 定位 notifications_per_day / app_opens_per_day 的作用机制
# 边际平坦 + 消融损失大  =>  必然是纯交互效应。找出与谁交互。
# =============================================================================
suppressMessages(library(data.table))
train <- readRDS("output/raw_train.rds")

cat("==================== 边际（复核）====================\n")
for (v in c("notifications_per_day", "app_opens_per_day")) {
  w <- if (v == "notifications_per_day") 50 else 30
  d <- train[!is.na(get(v)), .(n = .N, rate = mean(addicted_label)),
             by = .(bin = floor(get(v) / w) * w)][order(bin)]
  cat(sprintf("\n%s（每 %d 一箱）：\n", v, w))
  cat(sprintf("  极差 %.4f\n", diff(range(d[n > 1000, rate]))))
  print(d[n > 1000], row.names = FALSE)
}

cat("\n\n==================== 条件于屏幕时间 ====================\n")
d <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
d[, score := daily_screen_time_hours + social_media_hours]
# 按 screen+social 分成 5 段，段内看通知数的效应
d[, band := cut(score, breaks = quantile(score, 0:5/5), include.lowest = TRUE,
                labels = c("最低20%","次低","中间","次高","最高20%"))]

for (v in c("notifications_per_day", "app_opens_per_day")) {
  w <- if (v == "notifications_per_day") 80 else 50
  cat(sprintf("\n---- %s 在各屏幕时间段内的效应 ----\n", v))
  tb <- d[!is.na(get(v)),
          .(n = .N, rate = mean(addicted_label)),
          by = .(band, bin = floor(get(v) / w) * w)][n > 500][order(band, bin)]
  wide <- dcast(tb, band ~ bin, value.var = "rate")
  print(wide, row.names = FALSE, digits = 3)
  cat("  各段内极差：")
  rng <- tb[, .(spread = diff(range(rate))), by = band]
  cat(paste(sprintf("%s=%.3f", rng$band, rng$spread), collapse = "  "), "\n")
}

cat("\n\n==================== 中间段放大 ====================\n")
cat("（如果是交互，效应应该集中在标签最不确定的中间段）\n")
mid <- d[band == "中间" & !is.na(notifications_per_day) & !is.na(app_opens_per_day)]
cat(sprintf("中间段 n=%d，整体成瘾率 %.4f\n", nrow(mid), mean(mid$addicted_label)))
q <- mid[, .(n = .N, rate = mean(addicted_label)),
         by = .(notif_q = cut(notifications_per_day,
                              breaks = quantile(notifications_per_day, 0:4/4),
                              include.lowest = TRUE,
                              labels = c("Q1低","Q2","Q3","Q4高")))][order(notif_q)]
print(q, row.names = FALSE)
cat(sprintf("中间段内通知数四分位的成瘾率极差：%.4f\n", diff(range(q$rate))))
