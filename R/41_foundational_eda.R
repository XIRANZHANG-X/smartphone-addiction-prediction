# =============================================================================
# 41_foundational_eda.R —— 基础层 EDA：分布、相关性、缺失联合结构
#
# 用法：Rscript R/41_foundational_eda.R
# 产出：reports/figures/fig9_特征分布.png
#       reports/figures/fig10_相关性热力图.png
#       reports/figures/fig11_缺失联合结构.png
#       output/foundational_eda.rds
#
# -----------------------------------------------------------------------------
# 为什么要补这一层
# -----------------------------------------------------------------------------
# 现有的 8 张官方图和 R/17~20 的补充分析，全部是「先有假设、再去验证」的
# 叙事型分析（分箱看成瘾率、查残差、比插补方法……），没有一张图是单纯
# 展示「这份数据本身长什么样」的。这个脚本补的就是这一层最基础的东西：
#   (a) 12 个原始特征各自的分布
#   (b) 特征之间完整的两两相关性（不只是文字点名的几对）
#   (c) 缺失是不是「联合独立」的（发现 6 只验证过边际独立）
#
# 只读 output/raw_train.rds —— 已经存在的真实竞赛数据，不训练模型、
# 不做插补。
#
# 配色与主题跟随 R/12_figures.R 的设计系统（Tableau 10 低饱和、
# SimHei 中文加粗、黑色加粗边框），方便直接拼进现有汇报材料。
# =============================================================================

suppressMessages({
  library(data.table); library(ggplot2); library(scales)
  library(showtext); library(sysfonts)
})

font_path <- "C:/Windows/Fonts/simhei.ttf"
if (file.exists(font_path)) { font_add("SimHei", font_path); CN <- "SimHei" } else { CN <- "sans" }
showtext_auto(); showtext_opts(dpi = 300)

BLUE <- "#4E79A7"; ORANGE <- "#F28E2B"; RED <- "#E15759"
TEAL <- "#76B7B2"; GREEN <- "#59A14F"; GREY <- "grey45"

DIR <- "reports/figures"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)

theme_proj <- function(base_size = 14) {
  theme_bw(base_size = base_size) +
    theme(
      text             = element_text(family = CN, face = "bold"),
      panel.border     = element_rect(color = "black", fill = NA, linewidth = 1.6),
      axis.title       = element_text(face = "bold", size = base_size + 3),
      axis.text        = element_text(face = "bold", size = base_size - 1, color = "black"),
      axis.ticks       = element_line(color = "black", linewidth = 0.8),
      panel.grid.major = element_line(linetype = "dashed", color = "grey70", linewidth = 0.5),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = base_size + 6, hjust = 0),
      plot.subtitle    = element_text(face = "plain", size = base_size, color = "grey30", hjust = 0),
      strip.text       = element_text(face = "bold", size = base_size),
      strip.background = element_rect(fill = "grey92", color = "black", linewidth = 0.8)
    )
}
save_fig <- function(p, name, w = 12, h = 8) {
  f <- file.path(DIR, paste0(name, ".png"))
  ggsave(f, p, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("  已保存 %s\n", f))
}

cat("读取数据 ...\n")
train <- readRDS("output/raw_train.rds")

NUM_COLS <- c("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
              "work_study_hours", "sleep_hours", "notifications_per_day",
              "app_opens_per_day", "weekend_screen_time")
CAT_COLS <- c("gender", "stress_level", "academic_work_impact")
NAME_CN <- c(
  age = "年龄", daily_screen_time_hours = "每日屏幕时间", social_media_hours = "社交媒体时间",
  gaming_hours = "游戏时间", work_study_hours = "工作学习时间", sleep_hours = "睡眠时间",
  notifications_per_day = "每日通知数", app_opens_per_day = "每日打开次数",
  weekend_screen_time = "周末屏幕时间", gender = "性别", stress_level = "压力水平",
  academic_work_impact = "学业工作影响"
)

# =============================================================================
# 图 9：12 个原始特征各自的分布
# =============================================================================
cat("[1/3] 特征分布 ...\n")

num_long <- rbindlist(lapply(NUM_COLS, function(cc) {
  x <- train[[cc]]; x <- x[!is.na(x)]
  data.table(feature = NAME_CN[cc], value = x)
}))
num_long[, feature := factor(feature, levels = unname(NAME_CN[NUM_COLS]))]

p9a <- ggplot(num_long, aes(x = value)) +
  geom_histogram(bins = 40, fill = BLUE, color = "black", linewidth = 0.3, alpha = 0.85) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  labs(title = "图 9a  九个数值特征的分布（训练集，剔除缺失）",
       x = NULL, y = "行数") +
  theme_proj(13)
save_fig(p9a, "fig9a_数值特征分布", w = 13, h = 10)

cat_long <- rbindlist(lapply(CAT_COLS, function(cc) {
  x <- train[[cc]]; x <- x[!is.na(x)]
  tb <- as.data.table(table(x)); setnames(tb, c("level", "n"))
  tb[, feature := NAME_CN[cc]][, prop := n / sum(n)]
}))

p9b <- ggplot(cat_long, aes(x = level, y = prop, fill = feature)) +
  geom_col(color = "black", linewidth = 0.4, width = 0.65) +
  geom_text(aes(label = percent(prop, accuracy = 0.1)), vjust = -0.5,
            family = CN, fontface = "bold", size = 4) +
  facet_wrap(~ feature, scales = "free_x", ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.18))) +
  scale_fill_manual(values = c(BLUE, ORANGE, TEAL), guide = "none") +
  labs(title = "图 9b  三个类别特征的分布（训练集，剔除缺失）", x = NULL, y = "占比") +
  theme_proj(13)
save_fig(p9b, "fig9b_类别特征分布", w = 11, h = 4.5)

# =============================================================================
# 图 10：完整的相关性热力图（9 个数值特征）
# =============================================================================
cat("[2/3] 相关性热力图 ...\n")

M <- as.matrix(train[, ..NUM_COLS])
colnames(M) <- unname(NAME_CN[NUM_COLS])
cor_m <- cor(M, use = "pairwise.complete.obs")

cor_long <- as.data.table(as.table(cor_m))
setnames(cor_long, c("Var1", "Var2", "value"))

p10 <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.2f", value)), family = CN, fontface = "bold",
            size = 3.6, color = ifelse(abs(cor_long$value) > 0.5, "white", "black")) +
  scale_fill_gradient2(low = "#4E79A7", mid = "white", high = "#E15759",
                       midpoint = 0, limits = c(-1, 1), name = "相关系数") +
  labs(title = "图 10  九个数值特征的完整相关性矩阵", x = NULL, y = NULL,
       subtitle = "Pearson 相关系数，两两成对剔除缺失（pairwise complete）") +
  theme_proj(13) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1),
        panel.grid = element_blank())
save_fig(p10, "fig10_相关性热力图", w = 10, h = 8.5)

# =============================================================================
# 图 11：缺失的联合结构
# =============================================================================
cat("[3/3] 缺失联合结构 ...\n")

ALL_COLS <- c(NUM_COLS, CAT_COLS)
na_mat <- as.data.table(lapply(ALL_COLS, function(cc) as.integer(is.na(train[[cc]]))))
setnames(na_mat, ALL_COLS)
colnames(na_mat) <- unname(NAME_CN[ALL_COLS])

na_cor <- cor(as.matrix(na_mat))
na_cor_long <- as.data.table(as.table(na_cor))
setnames(na_cor_long, c("Var1", "Var2", "value"))

p11a <- ggplot(na_cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.2f", value)), family = CN, fontface = "bold", size = 3.2) +
  scale_fill_gradient2(low = "#4E79A7", mid = "white", high = "#E15759",
                       midpoint = 0, limits = c(-0.05, 0.05), oob = squish,
                       name = "相关系数") +
  labs(title = "图 11a  12 个缺失指示列的两两相关性", x = NULL, y = NULL,
       subtitle = "接近 0 说明缺失是彼此独立发生的（跟发现 6 的边际 MCAR 结论互相印证）") +
  theme_proj(11) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1), panel.grid = element_blank())
save_fig(p11a, "fig11a_缺失相关性", w = 10, h = 8.5)

# 每行缺失个数：实际分布 vs 12 个独立 MCAR 硬币各自抛出来的理论二项分布
miss_cols <- ALL_COLS
n_missing <- rowSums(is.na(train[, ..miss_cols]))
p_miss <- vapply(miss_cols, function(cc) mean(is.na(train[[cc]])), numeric(1))
p_bar <- mean(p_miss)   # 用平均缺失率近似二项分布参数

obs <- as.data.table(table(n_missing))
setnames(obs, c("k", "n")); obs[, k := as.integer(k)]
obs[, prop := n / sum(n)][, grp := "实际观测"]

k_seq <- 0:length(miss_cols)
theo <- data.table(k = k_seq, prop = dbinom(k_seq, length(miss_cols), p_bar), grp = "独立假设下的理论值")

cmp <- rbind(obs[, .(k, prop, grp)], theo)
cmp[, grp := factor(grp, levels = c("实际观测", "独立假设下的理论值"))]

p11b <- ggplot(cmp, aes(x = k, y = prop, fill = grp)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = c(BLUE, GREY), name = NULL) +
  scale_x_continuous(breaks = k_seq) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "图 11b  每行缺失特征数：实际分布 vs 独立假设下的理论分布",
       subtitle = sprintf("理论曲线 = 12 个独立 MCAR 硬币各自以平均缺失率 %.1f%% 抛出的二项分布", 100 * p_bar),
       x = "一行里缺失的特征个数", y = "占比") +
  theme_proj(13) +
  theme(legend.position = "top")
save_fig(p11b, "fig11b_缺失个数分布对照", w = 11, h = 6.5)

ks_test <- suppressWarnings(ks.test(n_missing, "pbinom", length(miss_cols), p_bar))
cat(sprintf("\n实际缺失个数分布 vs 二项分布的 KS D = %.4f（越接近 0 说明越像独立 MCAR）\n",
            unname(ks_test$statistic)))

saveRDS(list(cor_matrix = cor_m, na_cor_matrix = na_cor,
             n_missing_observed = table(n_missing), p_bar = p_bar,
             ks_D = unname(ks_test$statistic)),
        "output/foundational_eda.rds")

cat("\n全部完成，结果已存至 output/foundational_eda.rds\n")
