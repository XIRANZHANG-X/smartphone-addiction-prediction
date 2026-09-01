# =============================================================================
# 17_discussion_checks.R —— 在我们自己的数据上核查讨论区的说法
#
# 用法：Rscript R/17_discussion_checks.R
# 产出：output/discussion_checks.rds + 屏幕报告
#
# -----------------------------------------------------------------------------
# 为什么要有这个文件
# -----------------------------------------------------------------------------
# Kaggle 讨论区（存档见 kaggle_discussions/讨论区全文中译.md）里有大量实验
# 数字，但它们：
#   1. 互相矛盾（"缺失是否携带信号"一题，8 篇的结论与 2 篇相反）
#   2. 口径各异（有人 5 折、有人 10 折、有人重复 holdout、有人样本内编码）
#   3. 至少有两处是作者自己事后撤回的
#
# 所以在改任何代码之前，先把可以零成本核查的说法放到**我们自己的数据**上过一遍。
# 每一节都标注了它对应讨论区的哪一帖。
#
# 本脚本只做「不需要训练模型」的检查；需要训练的放在 18_new_features.R。
# =============================================================================

suppressMessages({library(data.table)})
train <- readRDS("output/raw_train.rds")
test  <- readRDS("output/raw_test.rds")
res   <- list()

hr <- function(x) cat("\n", strrep("=", 74), "\n ", x, "\n", strrep("=", 74), "\n", sep = "")
pct <- function(x) sprintf("%.4f%%", 100 * x)

fast_auc <- function(y, p) {
  ok <- !is.na(p); y <- y[ok]; p <- p[ok]
  r <- rank(p, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# =============================================================================
hr("核查 1：预算约束是三项还是四项？（讨论区第 7、29、40 帖）")
# =============================================================================
# 我们的发现 1 只验证了 daily >= social + gaming。
# 讨论区（ryota517 最先发布，Georgy Mamarin / Dariush Afshar 反复引用）说的是
#   daily >= social + gaming + work_study
# 这是一个**更紧**的约束，我们从来没测过。
# 如果它也 100% 成立，那么我们的 other_screen 就少减了一项。

c3 <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours)]
c4 <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours) & !is.na(work_study_hours)]

ok3 <- c3[, mean(daily_screen_time_hours >= social_media_hours + gaming_hours)]
ok4 <- c4[, mean(daily_screen_time_hours >=
                 social_media_hours + gaming_hours + work_study_hours)]

cat(sprintf("三项约束  daily >= social + gaming            %s  （%s 行）\n",
            pct(ok3), format(nrow(c3), big.mark = ",")))
cat(sprintf("四项约束  daily >= social + gaming + work      %s  （%s 行）\n",
            pct(ok4), format(nrow(c4), big.mark = ",")))

c4[, resid4 := daily_screen_time_hours -
               (social_media_hours + gaming_hours + work_study_hours)]
c4[, resid3 := daily_screen_time_hours - social_media_hours - gaming_hours]

cat(sprintf("\n四项残差 resid4  均值 %.3f  标准差 %.3f  最小 %.3f  负值比例 %s\n",
            c4[, mean(resid4)], c4[, sd(resid4)], c4[, min(resid4)],
            pct(c4[, mean(resid4 < 0)])))
cat(sprintf("三项残差 resid3  均值 %.3f  标准差 %.3f  最小 %.3f\n",
            c4[, mean(resid3)], c4[, sd(resid3)], c4[, min(resid3)]))

cat(sprintf("\n单变量 AUC：resid4 %.5f   resid3 %.5f   差 %+.5f\n",
            fast_auc(c4$addicted_label, c4$resid4),
            fast_auc(c4$addicted_label, c4$resid3),
            fast_auc(c4$addicted_label, c4$resid4) -
              fast_auc(c4$addicted_label, c4$resid3)))
cat(sprintf("两者相关 %.4f\n", c4[, cor(resid4, resid3)]))

res$budget <- list(ok3 = ok3, ok4 = ok4,
                   auc_resid4 = fast_auc(c4$addicted_label, c4$resid4),
                   auc_resid3 = fast_auc(c4$addicted_label, c4$resid3),
                   n4 = nrow(c4))

# =============================================================================
hr("核查 2：小数格点（讨论区第 28 帖 tomasa2、第 30 帖 Faheem）")
# =============================================================================
# 说法：daily_screen_time_hours 的第一位小数能让正例率摆动 8.5 个百分点，
#       而这与手机使用行为无关，纯粹是生成器写数字的方式。
# 如果成立，这是一类树**无法**通过分裂重建的信息：target encoding 把每个
# 精确取值分开处理，无法把「所有以 .2 结尾的」汇集起来。

d <- train[!is.na(daily_screen_time_hours)]
d[, d1 := as.integer(round((daily_screen_time_hours -
                            floor(daily_screen_time_hours)) * 10)) %% 10L]
tb <- d[, .(n = .N, rate = mean(addicted_label)), by = d1][order(d1)]
print(tb, row.names = FALSE)
swing1 <- diff(range(tb$rate))
cat(sprintf("\n第一位小数摆幅 %.4f（讨论区报告 0.085）\n", swing1))

# 第二位小数作为对照：讨论区说它一文不值
d[, d2 := as.integer(round((daily_screen_time_hours * 100 -
                            floor(daily_screen_time_hours * 100)))) %% 10L]
tb2 <- d[, .(n = .N, rate = mean(addicted_label)), by = d2][n > 1000][order(d2)]
swing2 <- if (nrow(tb2) > 1) diff(range(tb2$rate)) else NA_real_
cat(sprintf("第二位小数摆幅 %.4f（对照，应接近 0）\n", swing2))

# 取值的离散程度：格点有多细
nd <- sapply(c("daily_screen_time_hours", "social_media_hours", "gaming_hours",
               "work_study_hours", "sleep_hours", "weekend_screen_time",
               "notifications_per_day", "app_opens_per_day", "age"),
             function(c) length(unique(na.omit(train[[c]]))))
cat("\n各列的不同取值数（决定 max_bin 该设多大）：\n")
for (i in order(-nd)) cat(sprintf("  %-26s %6d\n", names(nd)[i], nd[i]))

res$lattice <- list(swing_d1 = swing1, swing_d2 = swing2, n_distinct = nd,
                    table_d1 = tb)

# =============================================================================
hr("核查 3：训练/测试的缺失率是否不同？（讨论区第 53 帖 Dariush Afshar）")
# =============================================================================
# 我们的发现 6 证明的是「缺失与目标无关」。
# 这一帖问的是另一个问题：「缺失与划分身份是否相关」。两者可以同时成立。
# 如果成立，它是**不要做缺失指示**的第二个独立理由，也是解读对抗验证的前提。

cols <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
          "work_study_hours", "sleep_hours", "notifications_per_day",
          "app_opens_per_day", "weekend_screen_time",
          "gender", "stress_level", "academic_work_impact")

a <- sapply(cols, function(c) mean(is.na(train[[c]])))
b <- sapply(cols, function(c) mean(is.na(test[[c]])))
n1 <- nrow(train); n2 <- nrow(test)
pbar <- (a * n1 + b * n2) / (n1 + n2)
se   <- sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
z    <- (b - a) / se

mr <- data.table(col = cols, train_pct = 100 * a, test_pct = 100 * b,
                 diff_pp = 100 * (b - a), z = z)
setorder(mr, diff_pp)
print(mr[, .(col, train_pct = round(train_pct, 2), test_pct = round(test_pct, 2),
             diff_pp = round(diff_pp, 2), z = round(z, 1))], row.names = FALSE)
cat(sprintf("\n|z| 最小 %.1f，最大 %.1f；%d 列在测试集缺失更多，%d 列更少\n",
            min(abs(z)), max(abs(z)), sum(b > a), sum(b < a)))
cat("=> 缺失率按划分不同，但这与「缺失是否预测目标」是两个问题。\n")

res$miss_shift <- mr

# =============================================================================
hr("核查 4：n_missing 对目标的单变量 AUC（复核我们的发现 6）")
# =============================================================================
nm <- rowSums(is.na(train[, ..cols]))
cat(sprintf("n_missing 单变量 AUC = %.5f （讨论区多人报告 0.502 / 0.50172）\n",
            fast_auc(train$addicted_label, nm)))
byn <- data.table(n = nm, y = train$addicted_label)[
         , .(rows = .N, rate = mean(y)), by = n][order(n)]
print(byn[rows > 500], row.names = FALSE)
res$n_missing_auc <- fast_auc(train$addicted_label, nm)

# =============================================================================
hr("核查 5：原始数据的两规则查找表（讨论区第 51、52 帖）")
# =============================================================================
# broccoli beef 的理论：原始 7500 行是 y ~ Bernoulli(p)，
#   p = 1    若 daily > 8  或 social > 4
#   p = 0    若 daily <= 6 且 social <= 4
#   p = 0.5  其他
# Busya PRIME 测出：这两条规则在原始数据上 AUC 0.9888，在合成数据上只有 0.835。
# 我们手上没有原始数据，但可以在**竞赛数据**上复现后半句。

s <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
rule <- rep(0.5, nrow(s))
rule[s$daily_screen_time_hours > 8 | s$social_media_hours > 4] <- 1
rule[s$daily_screen_time_hours <= 6 & s$social_media_hours <= 4] <- 0
cat(sprintf("两规则在竞赛数据上的 AUC = %.5f （讨论区报告 0.835）\n",
            fast_auc(s$addicted_label, rule)))
cat(sprintf("  规则判为 1 的区域，实际正例率 %.4f（原始数据应为 1.000）\n",
            s[daily_screen_time_hours > 8 | social_media_hours > 4,
              mean(addicted_label)]))
cat(sprintf("  规则判为 0 的区域，实际正例率 %.4f（原始数据应为 0.000）\n",
            s[daily_screen_time_hours <= 6 & social_media_hours <= 4,
              mean(addicted_label)]))
mid <- s[daily_screen_time_hours > 6 & daily_screen_time_hours <= 8 &
         social_media_hours <= 4]
cat(sprintf("  「中间区」占 %.1f%%，正例率 %.4f（原始数据是抛硬币 0.456）\n",
            100 * nrow(mid) / nrow(s), mid[, mean(addicted_label)]))
cat("\n=> 生成器把硬规则抹成了平滑场，并在原本是噪声的中间区写入了结构。\n")
cat("   这解释了为什么我们的模型能到 0.96 而两条规则只有 0.83。\n")

res$two_rule_auc <- fast_auc(s$addicted_label, rule)

# =============================================================================
hr("核查 6：exact-value target encoding 的上限（讨论区第 34、40 帖）")
# =============================================================================
# 讨论区把「按精确取值而非量级编码」称为本数据集最大的单一杠杆（+0.0027~0.0032）。
# 这里只测它的**样本内上限**——即一个取值查找表能达到的最好情况。
# 样本内是偏乐观的，所以这是上界；真实增益要在折内拟合才知道（见 18 号脚本）。

for (cc in c("daily_screen_time_hours", "social_media_hours", "weekend_screen_time")) {
  dd <- train[!is.na(get(cc))]
  te <- dd[, .(m = mean(addicted_label), n = .N), by = cc]
  setnames(te, cc, "v")
  dd[, v := get(cc)]
  enc <- te[dd, on = "v", x.m]
  cat(sprintf("%-26s 取值数 %5d   样本内查找表 AUC %.5f   原始值 AUC %.5f\n",
              cc, nrow(te), fast_auc(dd$addicted_label, enc),
              fast_auc(dd$addicted_label, dd[[cc]])))
}
cat("\n=> 差值就是「非单调性」的量：查找表能捕捉原始值的单调排序捕捉不到的东西。\n")
cat("   注意这是样本内数字，必然偏高；折内拟合的真实增益见 R/18_new_features.R。\n")

saveRDS(res, "output/discussion_checks.rds")
cat("\n已保存 output/discussion_checks.rds\n")
