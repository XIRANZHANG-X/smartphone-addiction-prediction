# =============================================================================
# 16_screensocial_audit.R —— screen_social 的剔除依据（存档脚本）
#
# 用法：Rscript R/16_screensocial_audit.R
#
# 背景：项目早期把 screen_social = 屏幕时间 + 社交时间 当作派生特征之一，
#       并在文档里称其为「近似充分统计量」。这个说法不成立：由发现 1 可知
#       social 本身就是 screen 的一个分量，二者相加等价于
#       2*social + gaming + other —— 社交被算了两遍。
#
# 本脚本的四项检验就是剔除它的依据：
#   1. 作为单变量打分器它确实有效（AUC 0.912 > screen 单独的 0.889），
#      但这只说明社交是屏幕时间里最有预测力的成分；
#   2. 扫描 screen + w*social，最优 w = 1.75 而非 1 —— 权重 1 只是
#      「两个量相加」这个动作的副产物，不是任何推导的结果；
#   3. 在模型里置换重要性仅 0.00096，可有可无；
#   4. 与自身两个分量回归 R^2 = 1.000000 —— 精确线性组合，零新信息，
#      对树模型是冗余，对 glmnet 造成设计矩阵秩亏。
#
# 结论：该变量已从 R/03_features.R 移除，派生特征由 6 个减为 5 个。
#       脚本保留于此，供报告引用与复现。
#
# 注：检验 3 读取 output/importance.rds。该文件若是在移除之后重新生成的，
#     里面不再有 screen_social 这一行，该检验会自动跳过 —— 属正常现象，
#     历史数值见上方与 docs/项目说明.md。
# =============================================================================

suppressMessages(library(data.table))
train <- readRDS("output/raw_train.rds")

fast_auc <- function(y, p) {
  ok <- !is.na(p)
  y <- y[ok]; p <- p[ok]
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
hr <- function(x) cat("\n", strrep("=", 70), "\n", x, "\n", strrep("=", 70), "\n", sep = "")

# 只在三列都不缺的行上比较，保证各方案面对同一批样本
d <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
           !is.na(gaming_hours)]
y <- d$addicted_label
d[, other := daily_screen_time_hours - social_media_hours - gaming_hours]
cat(sprintf("比较基准：三列均不缺的 %s 行\n", format(nrow(d), big.mark = ",")))

hr("检验 1：作为单变量打分器，各方案的 AUC")

cand <- list(
  "screen 单独"                       = d$daily_screen_time_hours,
  "social 单独"                       = d$social_media_hours,
  "gaming 单独"                       = d$gaming_hours,
  "other 单独（残差分量）"            = d$other,
  "screen + social（已剔除，社交算2遍）" = d$daily_screen_time_hours + d$social_media_hours,
  "screen + gaming"                   = d$daily_screen_time_hours + d$gaming_hours,
  "screen + social + gaming"          = d$daily_screen_time_hours + d$social_media_hours + d$gaming_hours,
  "2*social + gaming + other（等价展开）" = 2 * d$social_media_hours + d$gaming_hours + d$other
)
res <- data.table(方案 = names(cand),
                  AUC = vapply(cand, function(v) fast_auc(y, v), 0))
setorder(res, -AUC)
for (i in seq_len(nrow(res)))
  cat(sprintf("  %-38s %.5f\n", res$方案[i], res$AUC[i]))

cat(sprintf("\nscreen+social 相对 screen 单独：%+.5f\n",
            res[方案 == "screen + social（已剔除，社交算2遍）", AUC] -
            res[方案 == "screen 单独", AUC]))

hr("检验 2：换个权重会更好吗？—— 扫描 screen + w*social")

ws <- seq(0, 3, by = 0.25)
aw <- vapply(ws, function(w) fast_auc(y, d$daily_screen_time_hours + w * d$social_media_hours), 0)
for (i in seq_along(ws))
  cat(sprintf("  w = %.2f   AUC %.5f%s\n", ws[i], aw[i],
              if (ws[i] == 1) "   <- 曾用的权重" else
              if (aw[i] == max(aw)) "   <- 最优" else ""))
cat(sprintf("\n最优权重 w=%.2f，比 w=1 高 %+.5f\n", ws[which.max(aw)], max(aw) - aw[ws == 1]))
cat("=> w=1 并非最优，它只是「两个量相加」这个动作的副产物。\n")

hr("检验 3：把它从模型里单独删掉，AUC 掉多少")
if (file.exists("output/importance.rds")) {
  im <- readRDS("output/importance.rds")$table
  for (f in c("screen_social", "other_screen", "social_share",
              "gaming_share", "weekend_ratio", "free_frac")) {
    r <- im[feature == f]
    if (nrow(r))
      cat(sprintf("  %-16s 置换重要性 %.5f   增益占比 %5.2f%%\n",
                  f, r$perm_drop, 100 * r$gain))
  }
  cat("\n=> screen_social 的增益占比很高但置换重要性接近 0：\n")
  cat("   树频繁用它做分裂（因为它是个方便的组合），但一旦打乱，\n")
  cat("   模型立刻改用 screen 与 social 两个分量补回来 —— 它并非不可或缺。\n")
}

hr("检验 4：它与自身分量的相关性")
cs <- d$daily_screen_time_hours + d$social_media_hours
cat(sprintf("  与 screen 的相关   %.4f\n", cor(cs, d$daily_screen_time_hours)))
cat(sprintf("  与 social 的相关   %.4f\n", cor(cs, d$social_media_hours)))
m <- summary(lm(cs ~ d$daily_screen_time_hours + d$social_media_hours))
cat(sprintf("  被两个分量线性解释的比例 R^2 = %.6f\n", m$r.squared))
cat("=> R^2 = 1，它是两个分量的精确线性组合，不含任何新信息。\n")
cat("   对线性模型而言这会造成设计矩阵秩亏；对树模型只是多一列冗余。\n")
