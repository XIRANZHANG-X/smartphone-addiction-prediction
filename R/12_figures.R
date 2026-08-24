# =============================================================================
# 12_figures.R —— 生成全部图表
#
# 用法：Rscript R/12_figures.R
# 产出：reports/figures/*.png
#
# 绘图规范遵循「绘图代码参考/绘图规范与总结.md」：
#   - dpi = 300，ggsave 尺寸用英寸
#   - showtext + SimHei 处理中文
#   - Tableau 10 低饱和配色
#   - 黑色加粗边框 linewidth = 2，刻度线 linewidth = 1
#   - 虚线浅灰主网格 grey70，全局字体加粗
#   - 参考线一律 linetype = "dashed"
#
# -----------------------------------------------------------------------------
# 一条贯穿全文件的原则：**小差异不要用零基线柱状图**
# -----------------------------------------------------------------------------
# 本项目各算法的 AUC 差 0.03，而算法内部不同插补策略的差异只有 0.001~0.01。
# 柱状图从 0 起画，后者会被完全压平，四根柱子看上去一模一样。
# 凡是要展示小差异的地方，一律用点线图 + 各自缩放的纵轴，或者直接画差值。
# =============================================================================

suppressMessages({
  library(data.table); library(ggplot2); library(scales)
  library(showtext);   library(sysfonts); library(patchwork)
})

# ---- 中文字体 ---------------------------------------------------------------
font_path <- "C:/Windows/Fonts/simhei.ttf"
if (file.exists(font_path)) {
  font_add("SimHei", font_path)
  CN <- "SimHei"
} else {
  CN <- "sans"
  message("未找到 SimHei，中文可能显示为方块")
}
showtext_auto()
showtext_opts(dpi = 300)   # 不设的话 showtext 的字号会和 ggsave 的 dpi 对不上

# ---- 配色（Tableau 10 低饱和） ----------------------------------------------
BLUE   <- "#4E79A7"; ORANGE <- "#F28E2B"; RED  <- "#E15759"
TEAL   <- "#76B7B2"; GREEN  <- "#59A14F"; GREY <- "grey45"

DIR <- "reports/figures"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)

# ---- 统一主题 ---------------------------------------------------------------
theme_proj <- function(base_size = 16) {
  theme_bw(base_size = base_size) +
    theme(
      text             = element_text(family = CN, face = "bold"),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 2),
      axis.title       = element_text(face = "bold", size = base_size + 4),
      axis.text        = element_text(face = "bold", size = base_size,
                                      color = "black"),
      axis.ticks       = element_line(color = "black", linewidth = 1),
      panel.grid.major = element_line(linetype = "dashed", color = "grey70",
                                      linewidth = 0.6),
      panel.grid.minor = element_line(linetype = "dashed", color = "grey85",
                                      linewidth = 0.4),
      plot.title       = element_text(face = "bold", size = base_size + 6,
                                      hjust = 0),
      plot.subtitle    = element_text(face = "plain", size = base_size,
                                      color = "grey30", hjust = 0),
      legend.title     = element_text(face = "bold"),
      legend.text      = element_text(face = "bold"),
      legend.key       = element_rect(fill = "transparent", color = NA),
      strip.text       = element_text(face = "bold", size = base_size + 1),
      strip.background = element_rect(fill = "grey92", color = "black",
                                      linewidth = 1)
    )
}

save_fig <- function(p, name, w = 9, h = 6.5) {
  f <- file.path(DIR, paste0(name, ".png"))
  ggsave(f, p, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("  已保存 %s\n", f))
}

# ---- 数据 -------------------------------------------------------------------
cat("读取数据 ...\n")
train <- readRDS("output/raw_train.rds")

# =============================================================================
# 图 1：硬结构约束（发现 1）
# =============================================================================
cat("\n[1/7] 硬结构约束 ...\n")

cc <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours)]
set.seed(1)
samp <- cc[sample(.N, 12000)]
samp[, part_sum := social_media_hours + gaming_hours]

p1 <- ggplot(samp, aes(x = part_sum, y = daily_screen_time_hours)) +
  geom_point(color = BLUE, alpha = 0.25, size = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              linewidth = 1.2, color = RED) +
  annotate("text", x = 0.6, y = 15.4, hjust = 0, family = CN, fontface = "bold",
           size = 5.5, color = RED, label = "y = x（约束边界）") +
  annotate("text", x = 0.6, y = 14.0, hjust = 0, family = CN, fontface = "bold",
           size = 4.8, color = GREY,
           label = "全部 451,246 行无一例外落在边界上方") +
  labs(
    title    = "图 1  屏幕时间的结构性约束",
    subtitle = "每日屏幕时间 ≥ 社交媒体时间 + 游戏时间，在完整数据上 100% 成立",
    x = "社交媒体时间 + 游戏时间（小时）",
    y = "每日屏幕时间（小时）"
  ) +
  coord_cartesian(xlim = c(0, 12), ylim = c(0, 16)) +
  theme_proj()

save_fig(p1, "fig1_硬结构约束")

# =============================================================================
# 图 2：观测特征对标签的解释力（发现 5）
# =============================================================================
cat("[2/7] 标签单调性 ...\n")

# -----------------------------------------------------------------------------
# 这张图被质疑过，质疑意见与实测回应记录在此，避免以后重复争论：
#
# 质疑一「把两个时长相加，右端必然 100%，是数学必然而非发现」
#   反证：把标签随机打乱后重新分箱，极差从 0.862 塌到 0.028，
#   17 个间隔里 9 处下降 —— 同样的相加操作，换个标签就没有单调性。
#   另有反例：通知数/50 + 打开次数/30 同样相加，极差仅 0.0895 且不单调。
#   单调性来自特征与标签的真实关联，不是算术。
#
# 质疑二「右端没有阴性样本」
#   部分属实但被显示精度掩盖了：13~16 箱仍有 127 个阴性样本
#   （99.77%~99.98%），真正零阴性的只有 17 箱以上，占全量 1.4%。
#   本版改用四位小数并标注阴性样本数，不再显示成整齐的 100.0%。
#
# 质疑三「剔除缺失行引入选择偏差」
#   实测：两列都不缺 vs 至少缺一列，成瘾率 0.7092 对 0.7099，差 0.0007；
#   屏幕时间分布均值差 0.011 小时，KS 的 D=0.0085。
#   （KS 的 p<0.0001 不说明问题 —— 50 万样本上任何微小差异都会显著，
#     要看的是 D 值。）这与发现 6「缺失为 MCAR」互相印证。
#
# 质疑四「应该去掉 100% 区间只画 0~12 小时」
#   不采纳。截断高值区间等于藏数据，本项目不做这种处理。
#   改为把不确定性显式画出来（置信区间 + 样本量），让读者自行判断。
#
# 据此本版做了三处改进：
#   (a) 主面板改用**单个变量**，保留 86.1% 的数据而非 72.6%
#   (b) 加 95% 置信区间
#   (c) 副面板加**打乱标签的对照线**，直接展示效应的真实性
# -----------------------------------------------------------------------------

# --- (a) 单变量 + 置信区间（仅剔除该列缺失，保留 86.1% 数据）----------------
one <- train[!is.na(daily_screen_time_hours),
             .(n = .N, pos = sum(addicted_label), rate = mean(addicted_label)),
             by = .(bin = floor(daily_screen_time_hours))][n > 500][order(bin)]
one[, se := sqrt(rate * (1 - rate) / n)]
one[, `:=`(lo = pmax(0, rate - 1.96 * se), hi = pmin(1, rate + 1.96 * se))]

p2a <- ggplot(one, aes(x = bin, y = rate)) +
  geom_hline(yintercept = 0.7094, linetype = "dashed",
             linewidth = 0.9, color = GREY) +
  annotate("text", x = 13, y = 0.665, hjust = 1, family = CN, fontface = "bold",
           size = 4.2, color = GREY, label = "全体基准率 0.7094") +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = BLUE, alpha = 0.25) +
  geom_line(color = BLUE, linewidth = 1.4) +
  geom_point(aes(size = n), shape = 21, fill = BLUE, color = "black", stroke = 1) +
  scale_size_continuous(range = c(2.5, 7), name = "样本数",
                        labels = label_comma()) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 14, 2)) +
  labs(subtitle = "(a) 仅用每日屏幕时间一个变量（保留 86.1% 的数据，阴影为 95% 置信区间）",
       x = "每日屏幕时间（小时）", y = "成瘾比例") +
  theme_proj(14)

# --- (b) 真实标签 vs 打乱标签 ------------------------------------------------
s <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
s[, score := daily_screen_time_hours + social_media_hours]
set.seed(42)
s[, y_shuf := sample(addicted_label)]

real <- s[, .(n = .N, rate = mean(addicted_label)),
          by = .(bin = floor(score))][n > 500][order(bin)][, grp := "真实标签"]
shuf <- s[, .(n = .N, rate = mean(y_shuf)),
          by = .(bin = floor(score))][n > 500][order(bin)][, grp := "标签随机打乱（对照）"]
cmp <- rbind(real, shuf)
cmp[, grp := factor(grp, levels = c("真实标签", "标签随机打乱（对照）"))]

p2b <- ggplot(cmp, aes(x = bin, y = rate, color = grp, linetype = grp)) +
  geom_hline(yintercept = 0.7094, linetype = "dotted",
             linewidth = 0.8, color = GREY) +
  geom_line(linewidth = 1.4) +
  geom_point(size = 2.8) +
  scale_color_manual(values = c(BLUE, RED), name = NULL) +
  scale_linetype_manual(values = c("solid", "dashed"), name = NULL) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 20, 4)) +
  labs(subtitle = "(b) 对照检验：同样的「两变量相加再分箱」，打乱标签后曲线立刻塌平",
       x = "屏幕时间 + 社交时间（小时）", y = NULL) +
  theme_proj(14) +
  theme(legend.position = "top")

p2 <- (p2a | p2b) +
  plot_annotation(
    title    = "图 2  观测特征对标签的解释力",
    subtitle = paste0("成瘾比例随使用时长上升，最高箱达 99.98%（13~16 箱仍有 127 个阴性样本，",
                      "真正零阴性的仅占全量 1.4%）。\n右图证明该趋势并非「相加」这一操作的算术产物：",
                      "打乱标签后极差由 0.862 降至 0.028。"),
    theme = theme(
      plot.title    = element_text(family = CN, face = "bold", size = 22),
      plot.subtitle = element_text(family = CN, size = 13, color = "grey30")
    )
  )

save_fig(p2, "fig2_标签单调性", w = 15, h = 6.8)

# =============================================================================
# 图 3：Simpson 悖论（发现 7）
# =============================================================================
cat("[3/7] Simpson 悖论 ...\n")

mg <- train[!is.na(work_study_hours),
            .(n = .N, rate = mean(addicted_label)),
            by = .(bin = floor(work_study_hours / 2) * 2)][n > 1000][order(bin)]
mg[, view := "边际视角（不控制任何变量）"]

cd <- train[!is.na(work_study_hours) & !is.na(daily_screen_time_hours) &
            !is.na(social_media_hours) &
            daily_screen_time_hours + social_media_hours < 6,
            .(n = .N, rate = mean(addicted_label)),
            by = .(bin = floor(work_study_hours / 2) * 2)][n > 1000][order(bin)]
cd[, view := "条件视角（只看低屏幕时间人群）"]

sp <- rbind(mg, cd)
sp[, view := factor(view, levels = unique(view))]

p3 <- ggplot(sp, aes(x = factor(bin), y = rate, fill = view)) +
  geom_col(width = 0.6, color = "black", linewidth = 0.9) +
  geom_text(aes(label = percent(rate, accuracy = 0.1)),
            vjust = -0.6, family = CN, fontface = "bold", size = 5) +
  facet_wrap(~ view) +
  scale_fill_manual(values = c(BLUE, ORANGE), guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, 1.08), breaks = seq(0, 1, 0.2)) +
  labs(
    title    = "图 3  工作学习时间的 Simpson 悖论",
    subtitle = "同一个变量，边际上正相关、条件上负相关 —— 单变量分析会骗人",
    x = "每日工作 / 学习时间（小时，分箱下界）",
    y = "成瘾比例"
  ) +
  theme_proj()

save_fig(p3, "fig3_Simpson悖论", w = 12, h = 6.5)

# =============================================================================
# 图 4：天花板效应（发现 9）★ 全项目最重要的一张
# =============================================================================
cat("[4/7] 天花板效应 ...\n")

d <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours)]
d[, score := daily_screen_time_hours + social_media_hours]
qs <- quantile(d$score, 0:5 / 5)
BANDS <- c("最低 20%", "次低", "中间", "次高", "最高 20%")
d[, band := cut(score, breaks = qs, include.lowest = TRUE, labels = BANDS)]

# --- (a) 边际视角：两个特征都又平又跳 ---------------------------------------
marg <- rbind(
  train[!is.na(notifications_per_day),
        .(n = .N, rate = mean(addicted_label)),
        by = .(x = floor(notifications_per_day / 50) * 50)][n > 2000
      ][, feat := "每日通知数"],
  train[!is.na(app_opens_per_day),
        .(n = .N, rate = mean(addicted_label)),
        by = .(x = floor(app_opens_per_day / 30) * 30)][n > 2000
      ][, feat := "每日应用打开次数"]
)

p4a <- ggplot(marg, aes(x = x, y = rate, color = feat, group = feat)) +
  geom_hline(yintercept = 0.7094, linetype = "dashed",
             linewidth = 0.9, color = "grey55") +
  geom_line(linewidth = 1.3) +
  geom_point(size = 3.2) +
  scale_color_manual(values = c("每日通知数" = BLUE,
                                "每日应用打开次数" = ORANGE), name = NULL) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.55, 0.82)) +
  labs(subtitle = "(a) 边际视角：曲线又平又跳，看不出任何规律",
       x = "特征取值（分箱下界）", y = "成瘾比例") +
  theme_proj(14) +
  theme(legend.position = "top")

# --- (b) 分段后的带符号效应量 -----------------------------------------------
# 每段内「最高分箱的成瘾率 - 最低分箱的成瘾率」。
# 用带符号的差值而不是极差，这样效应的方向也能一并看出来。
band_effect <- function(col, w, nm) {
  t <- d[!is.na(get(col)),
         .(n = .N, rate = mean(addicted_label)),
         by = .(band, b = floor(get(col) / w) * w)][n > 500][order(band, b)]
  t[, .(effect = rate[.N] - rate[1]), by = band][, feat := nm][]
}
be <- rbind(band_effect("notifications_per_day", 80, "每日通知数"),
            band_effect("app_opens_per_day",     50, "每日应用打开次数"))
be[, band := factor(band, levels = BANDS)]

p4b <- ggplot(be, aes(x = band, y = effect, fill = feat)) +
  annotate("rect", xmin = 3.5, xmax = 5.5, ymin = -0.105, ymax = 0.105,
           alpha = 0.10, fill = RED) +
  geom_col(position = position_dodge(0.75), width = 0.68,
           color = "black", linewidth = 0.9) +
  geom_hline(yintercept = 0, linewidth = 1, color = "black") +
  geom_text(aes(label = sprintf("%+.3f", effect),
                vjust = ifelse(effect < 0, 1.5, -0.7)),
            position = position_dodge(0.75),
            family = CN, fontface = "bold", size = 4.2) +
  annotate("text", x = 4.5, y = 0.092, family = CN, fontface = "bold",
           size = 4.8, color = RED, label = "标签已饱和，效应归零") +
  scale_fill_manual(values = c("每日通知数" = BLUE,
                               "每日应用打开次数" = ORANGE), name = NULL) +
  scale_y_continuous(limits = c(-0.105, 0.105)) +
  labs(subtitle = "(b) 按屏幕时间分段后：低 60% 有清晰效应，高 40% 完全消失",
       x = "屏幕时间分段", y = "段内成瘾率变化（高分箱 - 低分箱）") +
  theme_proj(14) +
  theme(legend.position = "none")

p4 <- (p4a / p4b) +
  plot_layout(heights = c(1, 1.2)) +
  plot_annotation(
    title    = "图 4  天花板效应：为什么边际分析会看走眼",
    subtitle = paste0("两个特征在低分段的效应方向相反（通知数为负、打开次数为正），",
                      "在高分段同时归零；\n边际统计把三者平均，信号被抵消并稀释到看不见"),
    theme = theme(
      plot.title    = element_text(family = CN, face = "bold", size = 22),
      plot.subtitle = element_text(family = CN, size = 14, color = "grey30")
    )
  )

save_fig(p4, "fig4_天花板效应", w = 11, h = 10)

# =============================================================================
# 图 5：核心交互效应 —— 插补策略在四个算法上的表现
# =============================================================================
cat("[5/7] 核心交互效应 ...\n")

metas <- list.files("output/oof", pattern = "^meta_grid_", full.names = TRUE)
g <- rbindlist(lapply(metas, function(f) {
  m <- readRDS(f)
  data.table(algo = sub("^L[0-9]+_", "", m$model), impute = m$impute,
             auc = m$cv_mean, sd = m$cv_sd)
}))

lab <- c(L1 = "L1\n不插补", L2 = "L2\n中位数",
         L3 = "L3\n条件均值", L4 = "L4\n随机插补")
g[, impute_lab := factor(lab[impute], levels = lab)]
g[, algo_lab := factor(
  ifelse(algo %in% c("xgboost", "lightgbm"),
         paste0(algo, "\n可原生处理缺失"),
         paste0(algo, "\n必须插补")),
  levels = c("xgboost\n可原生处理缺失", "lightgbm\n可原生处理缺失",
             "ranger\n必须插补", "glmnet\n必须插补"))]
g[, best := auc == max(auc), by = algo]

# 关键：**不用零基线柱状图**（第一版就栽在这里）。
# 四个算法之间 AUC 差 0.03，算法内部只差 0.001~0.01，
# 柱状图从 0 起会把后者完全压平，四根柱子看着一模一样。
p5 <- ggplot(g, aes(x = impute_lab, y = auc, group = 1)) +
  geom_line(color = "grey50", linewidth = 1.1, linetype = "dashed") +
  geom_errorbar(aes(ymin = auc - sd, ymax = auc + sd),
                width = 0.16, linewidth = 0.9, color = "grey30") +
  geom_point(aes(fill = best), shape = 21, size = 5.5,
             color = "black", stroke = 1.3) +
  geom_text(aes(label = sprintf("%.4f", auc)), vjust = -1.9,
            family = CN, fontface = "bold", size = 4.1) +
  facet_wrap(~ algo_lab, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("TRUE" = RED, "FALSE" = BLUE), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.14, 0.22))) +
  labs(
    title    = "图 5  核心发现：同一插补策略，在两类模型上结论完全相反",
    subtitle = paste0("红点为该算法内的最优。左侧两个算法能自行表示「未知」，",
                      "不插补最好、越插补越差；\n右侧两个必须插补，条件均值插补反而领先。",
                      "各子图纵轴范围不同 —— 算法之间的差距远大于插补策略之间的差距"),
    x = NULL, y = "交叉验证 AUC"
  ) +
  theme_proj(14) +
  theme(axis.text.x = element_text(size = 12, lineheight = 0.95))

save_fig(p5, "fig5_核心交互效应", w = 15, h = 6.8)

# =============================================================================
# 图 6：特征消融
# =============================================================================
cat("[6/7] 特征消融 ...\n")

if (file.exists("output/ablation.rds")) {
  ab <- readRDS("output/ablation.rds")
  base <- mean(ab[["L1_full"]]$auc)
  keys <- c("L1_no_notif", "L1_no_opens", "L1_no_naind",
            "L1_no_cat3", "L1_no_age", "L1_noderiv")
  nm <- c("每日通知数", "每日应用打开次数", "12 个缺失指示",
          "性别 / 压力 / 学业影响", "年龄", "6 个派生特征")

  ad <- rbindlist(lapply(seq_along(keys), function(i) {
    if (is.null(ab[[keys[i]]])) return(NULL)
    data.table(feat = nm[i], delta = mean(ab[[keys[i]]]$auc) - base)
  }))
  ad[, feat := factor(feat, levels = feat[order(delta)])]
  ad[, harmful := delta < -0.001]

  p6 <- ggplot(ad, aes(x = feat, y = delta, fill = harmful)) +
    geom_col(width = 0.6, color = "black", linewidth = 0.9) +
    geom_hline(yintercept = 0, linewidth = 1, color = "black") +
    geom_text(aes(label = sprintf("%+.5f", delta),
                  hjust = ifelse(delta < 0, 1.15, -0.15)),
              family = CN, fontface = "bold", size = 4.8) +
    scale_fill_manual(values = c("TRUE" = RED, "FALSE" = TEAL), guide = "none") +
    coord_flip(ylim = c(-0.011, 0.004)) +
    labs(
      title    = "图 6  特征消融：删掉它，AUC 变化多少",
      subtitle = paste0("红色 = 删掉后明显变差，是真正有用的特征；",
                        "青色 = 删掉几乎无影响"),
      x = NULL, y = "AUC 变化量"
    ) +
    theme_proj()

  save_fig(p6, "fig6_特征消融", w = 10.5, h = 6.5)
} else {
  cat("  跳过：缺 output/ablation.rds\n")
}

# =============================================================================
# 图 7：CV 与 Kaggle 榜单的对应关系
# =============================================================================
cat("[7/7] CV 与 LB 对应 ...\n")

lb <- data.table(
  提交 = c("第一次提交\n七成员集成", "第二次提交\n含调参版九成员"),
  本地交叉验证 = c(0.96477, 0.96487),
  Kaggle榜单   = c(0.96613, 0.96627)
)
lbm <- melt(lb, id.vars = "提交", variable.name = "口径", value.name = "AUC")
lbm[, 口径 := factor(gsub("Kaggle榜单", "Kaggle 榜单", 口径),
                     levels = c("本地交叉验证", "Kaggle 榜单"))]

# 同样避开零基线：两个数只差 0.0014，从 0 起画会完全看不出来
p7 <- ggplot(lbm, aes(x = 提交, y = AUC, color = 口径, group = 口径)) +
  geom_line(aes(linetype = 口径), linewidth = 1.3) +
  geom_point(size = 5.5) +
  geom_text(aes(label = sprintf("%.5f", AUC)), vjust = -1.5,
            family = CN, fontface = "bold", size = 4.8, show.legend = FALSE) +
  scale_color_manual(values = c(BLUE, ORANGE), name = NULL) +
  scale_linetype_manual(values = c("dashed", "solid"), name = NULL) +
  coord_cartesian(ylim = c(0.9640, 0.9670)) +
  labs(
    title    = "图 7  本地验证与榜单的对应关系",
    subtitle = paste0("两次提交的差值稳定在 +0.0014。第二次提交前据此预测 0.96623，",
                      "\n实际 0.96627，误差 4×10⁻⁵ —— 本地验证可信，无需靠提交次数试错"),
    x = NULL, y = "AUC"
  ) +
  theme_proj() +
  theme(legend.position = "top")

save_fig(p7, "fig7_CV与榜单对应", w = 9.5, h = 6.5)

cat("\n全部图表已生成于 reports/figures/\n")
