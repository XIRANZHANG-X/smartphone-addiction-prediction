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
hr("发现 1：硬结构约束 —— 是四项，不是三项（预期 100.00%）")
# =============================================================================
# ⚠ 本节曾经只验证三项版 daily >= social + gaming，那是不完整的。
#   竞赛讨论区（ryota517 最先发布）用的是更紧的四项版本，我们复核后确认它同样
#   100% 成立，且**残差最小值恰为 0.000**，而三项版最小值是 0.100。
#
#   残差最小值就是判据：三项版是被四项版蕴含的推论，
#   四项版才是生成器真正强制执行的那条边界。
#
#   语义上的意外：work_study_hours 被算在屏幕时间**里面**（在手机上做的工作/学习）。
#   这与发现 8 的隐含假设相反，见那一节的更正。
#
#   完整核查见 R/17_discussion_checks.R，缘由见 docs/讨论区核查.md 2.1。

c3 <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours)]
c4 <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours) & !is.na(work_study_hours)]

r3 <- c3[, daily_screen_time_hours - social_media_hours - gaming_hours]
r4 <- c4[, daily_screen_time_hours -
           (social_media_hours + gaming_hours + work_study_hours)]

cat(sprintf("三项 daily >= social + gaming          %8.4f%%  （%s 行）  残差最小值 %.3f\n",
            100 * mean(r3 >= 0), format(nrow(c3), big.mark = ","), min(r3)))
cat(sprintf("四项 daily >= social + gaming + work   %8.4f%%  （%s 行）  残差最小值 %.3f  <- 真正的边界\n",
            100 * mean(r4 >= 0), format(nrow(c4), big.mark = ","), min(r4)))

cat(sprintf("\nresid4（预算余量）均值 %.3f  标准差 %.3f\n", mean(r4), sd(r4)))
cat(sprintf("resid3（旧口径）  均值 %.3f  标准差 %.3f\n", mean(r3), sd(r3)))
cat(sprintf("两者相关 %.4f —— 不是同一列\n",
            c4[, cor(daily_screen_time_hours - social_media_hours - gaming_hours,
                     daily_screen_time_hours -
                       (social_media_hours + gaming_hours + work_study_hours))]))

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
hr("发现 8（已更正）：「空闲时间」是硬约束 —— 我们原先把 work 减了两遍")
# =============================================================================
# ⚠ 本节的原结论是错的，错因由发现 1 的更正直接暴露出来。
#
#   原口径：24 - sleep - work - screen  →  1.84% 为负，于是我们断定
#           「不是硬约束，只能当软特征用」。
#
#   但发现 1 证明 work_study_hours 被算在 screen **里面**
#   （daily >= social + gaming + work 恒成立）。所以那个式子把 work 减了两遍，
#   负值是我们自己造出来的。
#
#   去掉重复扣减之后，它是一个干净的硬约束。

f <- train[!is.na(sleep_hours) & !is.na(work_study_hours) &
           !is.na(daily_screen_time_hours),
           24 - sleep_hours - work_study_hours - daily_screen_time_hours]
g <- train[!is.na(sleep_hours) & !is.na(daily_screen_time_hours),
           24 - sleep_hours - daily_screen_time_hours]

cat(sprintf("旧口径 24 - sleep - work - screen   均值 %.2f  最小 %+.2f  负值 %.2f%%\n",
            mean(f), min(f), 100 * mean(f < 0)))
cat(sprintf("新口径 24 - sleep - screen          均值 %.2f  最小 %+.2f  负值 %.2f%%  <- 硬约束\n",
            mean(g), min(g), 100 * mean(g < 0)))
cat("\n=> 「一天只有 24 小时」这个假设是成立的，当初测出 1.84% 负值\n")
cat("   纯粹是因为 work_study 被扣了两次。R/03_features.R 的 free_frac\n")
cat("   分母已随之更正。详见 docs/讨论区核查.md 2.1。\n")

# =============================================================================
hr("发现 10：生成器把硬规则抹成了平滑场")
# =============================================================================
# 这一条解释了一件我们长期没解释的事：为什么模型能到 0.96，
# 而「看起来就是全部信号」的两条规则只有 0.86。
#
# 原始 7500 行数据集本质上是一张两规则查找表（竞赛讨论区第 52 帖）：
#   p = 1    若 daily > 8  或 social > 4
#   p = 0    若 daily <= 6 且 social <= 4
#   p = 0.5  其他（占原始数据 14%，纯抛硬币）
# 这两条规则在**原始**数据上 AUC 0.9888。
#
# 在**竞赛**数据上跑同样的规则，边界不见了。差出来的那一截就是生成器加进去的结构。

s <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
rule <- rep(0.5, nrow(s))
rule[s$daily_screen_time_hours > 8 | s$social_media_hours > 4] <- 1
rule[s$daily_screen_time_hours <= 6 & s$social_media_hours <= 4] <- 0

auc_rule <- local({
  r <- rank(rule, ties.method = "average")
  n1 <- as.numeric(sum(s$addicted_label == 1)); n0 <- nrow(s) - n1
  (sum(r[s$addicted_label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
})
cat(sprintf("两规则在竞赛数据上的 AUC  %.5f  （原始数据上是 0.9888）\n", auc_rule))
cat(sprintf("  规则判 1 的区域 实际正例率 %.4f （原始应为 1.000）\n",
            s[daily_screen_time_hours > 8 | social_media_hours > 4, mean(addicted_label)]))
cat(sprintf("  规则判 0 的区域 实际正例率 %.4f （原始应为 0.000）\n",
            s[daily_screen_time_hours <= 6 & social_media_hours <= 4, mean(addicted_label)]))
mid <- s[daily_screen_time_hours > 6 & daily_screen_time_hours <= 8 & social_media_hours <= 4]
cat(sprintf("  「中间区」占 %.1f%%，正例率 %.4f （原始数据是抛硬币 0.456）\n",
            100 * nrow(mid) / nrow(s), mid[, mean(addicted_label)]))
cat("\n=> 生成器不只把硬规则模糊化了，它在原本是纯噪声的中间区**写入了真实结构**。\n")
cat("   所以标签不是一个确定函数，而是一个平滑概率场的伯努利抽样 ——\n")
cat("   而 AUC 奖励的正是拟合那个场。这也是发现 5 的机制。\n")

# =============================================================================
hr("发现 11：生成器的舍入格点 —— 小数位携带信号")
# =============================================================================
# 取值被写在 0.01 的格点上。第一位小数让正例率摆动最多 0.1047，
# 而手机使用行为解释不了这件事：这纯粹是生成器写数字的方式。
#
# 关键在于**这是 target encoding 捡不起来的**：编码把每个精确取值分开处理，
# 无法把「所有以 .2 结尾的」汇集起来。所以它和按取值编码是互补而非重复。

cat(sprintf("%-26s %8s %8s\n", "列", "第1位", "第2位"))
for (cc in c("weekend_screen_time", "daily_screen_time_hours", "sleep_hours",
             "social_media_hours", "gaming_hours", "work_study_hours")) {
  dd <- train[!is.na(get(cc))]
  iv <- as.integer(round(dd[[cc]] * 100))
  s1 <- dd[, .(r = mean(addicted_label)), by = .(k = (iv %/% 10L) %% 10L)][, diff(range(r))]
  s2 <- dd[, .(r = mean(addicted_label)), by = .(k =  iv %% 10L)][, diff(range(r))]
  cat(sprintf("%-26s %8.4f %8.4f\n", cc, s1, s2))
}
cat("\n每个数字背后有 5~7 万行，单元格标准误约 0.0019，所以 0.085 约为 45 个标准误。\n")
cat("⚠ 与讨论区第 28 帖不一致：对方称第二位小数「一文不值」。我们测出六列中有三列
")
cat("  （weekend / social / gaming）第二位摆幅**不低于甚至大于**第一位。两位都进候选。
")

# =============================================================================
hr("发现 12：训练/测试的缺失率逐列不同（与发现 6 不矛盾）")
# ⚠ 归属更正：这项检查**不是本节首创**。组员 E 在 2026-08-26 的
#   R/19_train_test_shift.R 第三节就逐列对比过 train/test 缺失率。
#   本节新增的只有两样：(a) 二项标准误与 z 值（他用的是 0.5 个百分点的
#   经验阈值，没有算显著性）；(b)「缺失与目标无关」和「缺失与划分身份
#   相关」是两个不同问题这一区分。发现权属于 E。
# =============================================================================
# 发现 6 说的是「缺失与**目标**无关」。这一条说的是「缺失与**划分身份**强相关」。
# 两句都对，问的是不同的问题 ——
# 一个不说明自己指哪一个的句子，就会被读成另一个。

test <- readRDS("output/raw_test.rds")
cols12 <- c(num_cols, "gender", "stress_level", "academic_work_impact")
a <- sapply(cols12, function(c) mean(is.na(train[[c]])))
b <- sapply(cols12, function(c) mean(is.na(test[[c]])))
n1 <- nrow(train); n2 <- nrow(test)
pb <- (a * n1 + b * n2) / (n1 + n2)
z  <- (b - a) / sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2))
mr <- data.table(col = cols12, train_pct = round(100 * a, 2),
                 test_pct = round(100 * b, 2), diff_pp = round(100 * (b - a), 2),
                 z = round(z, 1))[order(diff_pp)]
print(mr, row.names = FALSE)
cat(sprintf("\n12 列全部不同，|z| 从 %.1f 到 %.1f；%d 列测试集缺失更多，%d 列更少。\n",
            min(abs(z)), max(abs(z)), sum(b > a), sum(b < a)))
cat("=> 这是不做缺失指示的**第二个独立理由**：它们编码划分身份。\n")
cat("   也意味着在训练集上拟合的填补统计量，面对的缺失混合与测试集不同。\n")

cat("\n", strrep("=", 60), "\n", sep = "")
cat("全部 12 个发现复现完毕（发现 9 天花板效应在 R/09_interaction.R）。\n")
cat("E：请把发现 1、4、5、7 做成图，这四个最能一眼看出结论。\n")
