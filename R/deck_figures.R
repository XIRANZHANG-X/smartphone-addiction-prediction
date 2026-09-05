# =============================================================================
# deck_figures.R —— 答辩 PPT 的英文数据图（NTU crimson/sand/orange 配色）
#
# 用法（从仓库根目录）：
#   "D:/R/R-4.6.1/bin/Rscript.exe" R/deck_figures.R
#
# 产出：写进 ppt-master 项目的 images/ 目录。每张图**不带标题**——
#       幻灯片的 caption 拥有解释权（house style §7）。
# 配色对应 user-deck-preferences.md：
#   deep crimson #81021F (结构/论点系列), orange #FFA54C (强调), sand #E9DBD2,
#   neutral grey (其余), hairline #B6848E.
# 数字来源：output/raw_train.rds（原始数据）、output/importance.rds（置换重要性，
#   已训练的模型，直接读现成输出，不重训）。
# =============================================================================

suppressMessages({
  library(data.table)
  library(ggplot2)
  library(ragg)
})

CRIMSON <- "#81021F"
ORANGE  <- "#FFA54C"
SAND    <- "#E9DBD2"
GREY    <- "#8A8A8A"
DARK    <- "#1A1A1A"
HAIR    <- "#B6848E"

OUTDIR  <- "D:/PyCharm/PyStudy/ppt-master/projects/smartphone_addiction_defense_ppt169_20260905/images"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

theme_deck <- function(base_size = 13) {
  theme_bw(base_size = base_size) +
    theme(
      text             = element_text(family = "Times New Roman"),
      axis.title       = element_text(face = "bold", size = base_size + 2, color = DARK),
      axis.text        = element_text(face = "bold", size = base_size - 1, color = DARK),
      axis.ticks       = element_line(color = "black", linewidth = 0.9),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 1.4),
      panel.grid.major = element_line(linetype = "dashed", color = "grey78", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      legend.text      = element_text(face = "bold", size = base_size - 1),
      legend.title     = element_blank(),
      legend.key       = element_rect(fill = "transparent"),
      legend.background = element_rect(fill = "white", color = NA),
      plot.margin      = margin(6, 8, 6, 6)
    )
}

save_deck <- function(p, name, w, h) {
  f <- file.path(OUTDIR, paste0(name, ".png"))
  ragg::agg_png(f, width = w, height = h, units = "in", res = 300, bg = "white")
  print(p)
  dev.off()
  cat("saved", f, "\n")
}

train <- readRDS("output/raw_train.rds")

# =============================================================================
# 图 A：屏幕时间对标签的解释力 —— 真实标签 vs 打乱标签对照（(b) 面板口径）
# =============================================================================
s <- train[!is.na(daily_screen_time_hours)]
set.seed(42)
s[, y_shuf := sample(addicted_label)]

real <- s[, .(n = .N, rate = mean(addicted_label)),
          by = .(bin = floor(daily_screen_time_hours))][n > 500][order(bin)]
shuf <- s[, .(n = .N, rate = mean(y_shuf)),
          by = .(bin = floor(daily_screen_time_hours))][n > 500][order(bin)]
range_real <- diff(range(real$rate))
range_shuf <- diff(range(shuf$rate))
cat(sprintf("monotonicity: real range %.3f, shuffled range %.3f\n", range_real, range_shuf))

cmp <- rbind(
  real[, grp := "Observed labels"],
  shuf[, grp := "Shuffled labels (control)"]
)

pA <- ggplot(cmp, aes(x = bin, y = rate, color = grp, linetype = grp)) +
  geom_hline(yintercept = 0.7094, linetype = "dotted", linewidth = 0.7, color = GREY) +
  geom_line(linewidth = 1.15) +
  geom_point(size = 2.2) +
  scale_color_manual(values = c(CRIMSON, GREY)) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 14, 2)) +
  labs(x = "Daily screen time (hours, 1-hour bins)",
       y = "Addiction rate") +
  theme_deck() +
  theme(legend.position = "top")

save_deck(pA, "fig01_screen_monotonicity_EN", w = 7.6, h = 4.3)

# =============================================================================
# 图 A2：四项硬约束 —— 所有点落在 y = x 上方（四项口径，421,427 行）
# =============================================================================
cc <- train[!is.na(daily_screen_time_hours) & !is.na(social_media_hours) &
            !is.na(gaming_hours) & !is.na(work_study_hours)]
cat(sprintf("constraint rows: %d, min residual %.3f\n",
            nrow(cc), min(cc[, daily_screen_time_hours -
                             social_media_hours - gaming_hours - work_study_hours])))
set.seed(1)
samp <- cc[sample(.N, 12000)]
samp[, rhs := social_media_hours + gaming_hours + work_study_hours]

pA2 <- ggplot(samp, aes(x = rhs, y = daily_screen_time_hours)) +
  geom_point(color = CRIMSON, alpha = 0.22, size = 1.0) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 1.0, color = DARK) +
  annotate("text", x = 0.5, y = 15.6, hjust = 0, family = "Times New Roman",
           fontface = "italic", size = 4.2, color = DARK,
           label = "screen \u2265 social + gaming + work/study \u2014 421,427 rows, all above the line") +
  annotate("text", x = 0.5, y = 14.4, hjust = 0, family = "Times New Roman",
           fontface = "italic", size = 4.2, color = ORANGE,
           label = "minimum residual = 0.000") +
  coord_cartesian(xlim = c(0, 14), ylim = c(0, 16)) +
  labs(x = "Social + gaming + work/study hours",
       y = "Daily screen time (hours)") +
  theme_deck()

save_deck(pA2, "fig01b_constraint_EN", w = 7.6, h = 4.3)

# =============================================================================
# 图 B：三个"无信息"列 —— 在固定屏幕时间分段内，各取值的成瘾率基本恒定
# =============================================================================
d <- train[!is.na(daily_screen_time_hours)]
dq <- as.numeric(quantile(d$daily_screen_time_hours, c(1 / 3, 2 / 3)))
d[, band := ifelse(daily_screen_time_hours <= dq[1], "Low screen",
            ifelse(daily_screen_time_hours <= dq[2], "Mid screen", "High screen"))]
d[, band := factor(band, levels = c("Low screen", "Mid screen", "High screen"))]

noise_agg <- function(col, nm) {
  dd <- d[!is.na(get(col))]
  dd[, .(rate = mean(addicted_label), n = .N),
     by = .(band, v = as.character(get(col)))][n >= 400][, col := nm][]
}
nb <- rbind(
  noise_agg("gender", "Gender"),
  noise_agg("stress_level", "Stress level"),
  noise_agg("academic_work_impact", "Academic / work impact")
)
nb[, v := factor(v, levels = unique(v))]

pB <- ggplot(nb, aes(x = v, y = rate, color = band, group = band)) +
  geom_hline(yintercept = 0.7094, linetype = "dotted", linewidth = 0.6, color = GREY) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_point(size = 2.3) +
  facet_wrap(~ col, scales = "free_x") +
  scale_color_manual(values = c(CRIMSON, ORANGE, GREY)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.28, 1.04),
                     breaks = seq(0.3, 1.0, 0.2)) +
  labs(x = NULL, y = "Addiction rate") +
  theme_deck(12) +
  theme(legend.position = "top",
        strip.text = element_text(face = "bold", size = 12, color = DARK),
        axis.text.x = element_text(angle = 0))

save_deck(pB, "figB_noise_cols_EN", w = 8.6, h = 4.0)

# =============================================================================
# 图 C：置换重要性 —— 前五名全是编码列（橙色 = 逐取值编码列，灰 = 其余）
# =============================================================================
imp <- readRDS("output/importance.rds")$table
EN_LAB <- c(
  te_daily_screen_time_hours = "Encoded \u00b7 screen time",
  te_weekend_screen_time      = "Encoded \u00b7 weekend screen",
  te_notifications_per_day    = "Encoded \u00b7 notifications",
  te_app_opens_per_day        = "Encoded \u00b7 app opens",
  te_social_media_hours       = "Encoded \u00b7 social media",
  weekend_screen_time         = "Weekend screen (raw)",
  work_study_hours            = "Work / study hours (raw)",
  social_share                = "Social share (ratio)",
  daily_screen_time_hours     = "Screen time (raw)",
  other_screen                = "Budget slack (4-term residual)",
  social_media_hours          = "Social media (raw)",
  gaming_share                = "Gaming share (ratio)"
)
top <- imp[feature %in% names(EN_LAB)][order(perm_drop)][, .(feature, perm_drop)]
top[, label := EN_LAB[feature]]
top[, is_te := startsWith(feature, "te_")]
top[, label := factor(label, levels = label)]

pC <- ggplot(top, aes(x = perm_drop, y = label, fill = is_te)) +
  geom_col(width = 0.68, color = "black", linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.4f", perm_drop)),
            hjust = -0.12, family = "Times New Roman", fontface = "bold",
            size = 3.3, color = DARK) +
  scale_fill_manual(values = c(ORANGE, "grey82"), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Permutation importance (drop in validation AUC)",
       y = NULL) +
  theme_deck(12) +
  theme(axis.text.y = element_text(face = "bold", size = 10.5))

save_deck(pC, "fig08_permutation_importance_EN", w = 8.8, h = 5.2)

# =============================================================================
# P04 的"高瘦"变体：三列排版用，宽 ~350px、高 ~320px，字号加大保证可读。
# 数据与上方英文版完全一致，只是画布更高、字体更大、标注更少。
# =============================================================================
theme_tall <- function(base_size = 19) {
  theme_bw(base_size = base_size) +
    theme(
      text             = element_text(family = "Times New Roman"),
      axis.title       = element_text(face = "bold", size = base_size + 1, color = DARK),
      axis.text        = element_text(face = "bold", size = base_size - 2, color = DARK),
      axis.ticks       = element_line(color = "black", linewidth = 1),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 1.4),
      panel.grid.major = element_line(linetype = "dashed", color = "grey78", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      legend.text      = element_text(face = "bold", size = base_size - 2),
      legend.title     = element_blank(),
      legend.key       = element_rect(fill = "transparent"),
      legend.background = element_rect(fill = "white", color = NA),
      plot.margin      = margin(4, 6, 4, 4)
    )
}

save_tall <- function(p, name, w, h) {
  f <- file.path(OUTDIR, paste0(name, ".png"))
  ragg::agg_png(f, width = w, height = h, units = "in", res = 300, bg = "white")
  print(p)
  dev.off()
  cat("saved", f, "\n")
}

# --- 图 A tall：单调性（真实 vs 打乱） ---
pAt <- ggplot(cmp, aes(x = bin, y = rate, color = grp, linetype = grp)) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 2.6) +
  scale_color_manual(values = c(CRIMSON, GREY)) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 14, 4)) +
  labs(x = "Daily screen time (h)",
       y = "Addiction rate") +
  theme_tall() +
  theme(legend.position = "top", axis.title.x = element_text(margin = margin(t = 14)))
save_tall(pAt, "fig04a_monotonicity_tall", w = 4.1, h = 3.9)

# --- 图 C tall：四项约束 ---
pCt <- ggplot(samp, aes(x = rhs, y = daily_screen_time_hours)) +
  geom_point(color = CRIMSON, alpha = 0.20, size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 1.1, color = DARK) +
  coord_cartesian(xlim = c(0, 14), ylim = c(0, 16)) +
  labs(x = "Social + gaming + work/study",
       y = "Daily screen time (h)") +
  theme_tall() +
  theme(axis.title.x = element_text(margin = margin(t = 14)))
save_tall(pCt, "fig04c_constraint_tall", w = 4.1, h = 3.9)

# --- 图 B tall：三噪声列，竖向三面板 ---
nbv <- nb[order(col, band, v)]
pBt <- ggplot(nbv, aes(x = v, y = rate, color = band, group = band)) +
  geom_line(linewidth = 1.0, alpha = 0.85) +
  geom_point(size = 2.6) +
  facet_wrap(~ col, scales = "free_x", ncol = 1) +
  scale_color_manual(values = c(CRIMSON, ORANGE, GREY)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.28, 1.04), breaks = seq(0.3, 1.0, 0.2)) +
  labs(x = NULL, y = "Addiction rate") +
  theme_tall(16) +
  theme(legend.position = "top",
        strip.text = element_text(face = "bold", size = 15, color = DARK),
        panel.spacing = unit(0.6, "lines"))
save_tall(pBt, "fig04b_noise_tall", w = 4.1, h = 3.9)

cat("done\n")