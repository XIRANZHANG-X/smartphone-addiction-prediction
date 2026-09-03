# =============================================================================
# 29_paper_figures.R —— 论文用图（英文标签）
#
# 用法：Rscript R/29_paper_figures.R
# 产出：paper/figures/fig1_te_by_family.png
#       paper/figures/fig2_lattice_mechanism.png
#       paper/figures/fig3_size_ladder.png
#
# 与 R/12_figures.R 的关系：那一份是中文的、给中文文档用；这一份是英文的、
# 只画论文真正需要的三张。两者互不覆盖，输出目录也不同。
#
# 绘图规范照 MH6211 那份《绘图规范与总结》：dpi 300、全局加粗、
# Times New Roman、黑色加粗面板边框、Tableau 10 低饱和配色、虚线浅灰网格。
# =============================================================================

suppressMessages({
  library(data.table); library(ggplot2); library(patchwork)
})

dir.create("paper/figures", showWarnings = FALSE, recursive = TRUE)

# ggsave() 在这里走的是 ragg / systemfonts 渲染后端，它不认 windowsFonts() 注册的
# 别名，而是自己按字体族名向系统字体表查询。所以直接把 FONT 设成字体的真实族名
# "Times New Roman"，而不是转一道 windowsFonts() 别名。
# 是否真的存在这个字体，用 systemfonts::system_fonts()（ragg/systemfonts 自己解析
# 字体时依据的同一张表）实际查一次族名——查得到才诚实地判定"有"；
# systemfonts::match_fonts() 在查不到时会静默换成别的字体而不报错，不能拿来判断有没有，
# 所以不用它来做存在性检查。查不到就老实回落到默认无衬线（空字符串），不假装成功。
FONT <- tryCatch({
  if ("Times New Roman" %in% systemfonts::system_fonts()$family) "Times New Roman" else ""
}, error = function(e) "")

#' 论文统一主题
PAPER_THEME <- theme_bw(base_family = FONT) +
  theme(
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 2),
    axis.ticks       = element_line(color = "black", linewidth = 1),
    axis.title       = element_text(face = "bold", size = 20),
    axis.text        = element_text(face = "bold", size = 16, color = "black"),
    legend.title     = element_text(face = "bold", size = 16),
    legend.text      = element_text(face = "bold", size = 15),
    plot.title       = element_text(face = "bold", size = 22),
    plot.subtitle    = element_text(face = "bold", size = 15, color = "grey30"),
    strip.text       = element_text(face = "bold", size = 16),
    panel.grid.major = element_line(linetype = "dashed", color = "grey70",
                                    linewidth = 0.6),
    panel.grid.minor = element_line(linetype = "dashed", color = "grey85",
                                    linewidth = 0.4),
    legend.key       = element_rect(fill = "transparent", color = NA)
  )

# Tableau 10 低饱和；命名向量，避免顺序错乱
ALGO_COLORS <- c(
  "xgboost"  = "#4E79A7",
  "lightgbm" = "#F28E2B",
  "ranger"   = "#59A14F",
  "glmnet"   = "#E15759"
)

# -----------------------------------------------------------------------------
# 图 1：逐取值编码的收益，按模型族
# -----------------------------------------------------------------------------
# 这是论文的核心图。同一个预处理，在四个模型族上的收益跨越一个数量级：
# GBDT 约 +0.004，随机森林 5 倍于此，线性模型 8 倍于此。
te <- readRDS("output/te_by_family.rds")

gains <- rbindlist(lapply(names(te), function(a) {
  data.table(algo = a, fold = seq_along(te[[a]]$on),
             gain = te[[a]]$on - te[[a]]$off)
}))
gains[, algo := factor(algo, levels = names(ALGO_COLORS))]

summ <- gains[, .(m = mean(gain), se = sd(gain) / sqrt(.N)), by = algo]
summ[, `:=`(lo = m - 1.96 * se, hi = m + 1.96 * se)]

fig1 <- ggplot(summ, aes(x = algo, y = m, fill = algo)) +
  geom_col(width = 0.65, color = "black", linewidth = 0.8, alpha = 0.85) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 1.1) +
  geom_point(data = gains, aes(x = algo, y = gain),
             inherit.aes = FALSE, shape = 21, size = 3, stroke = 0.9,
             fill = "white", color = "black",
             position = position_jitter(width = 0.10, height = 0, seed = 1)) +
  geom_text(aes(label = sprintf("%+.5f", m), y = hi),
            vjust = -0.8, size = 5.5, fontface = "bold", family = FONT) +
  scale_fill_manual(values = ALGO_COLORS, guide = "none") +
  scale_y_continuous(limits = c(0, 0.040),
                     breaks = seq(0, 0.04, 0.01),
                     minor_breaks = seq(0, 0.04, 0.005),
                     expand = expansion(mult = c(0, 0.08))) +
  labs(
    x = NULL,
    y = "AUC gain from exact-value target encoding",
    title = "The same preprocessing step is worth 8x more to a linear model",
    subtitle = paste0("Bars: mean over 5 frozen folds, 95% CI. Points: individual folds. ",
                      "All four families: 5/5 folds agree in sign.")
  ) +
  PAPER_THEME

ggsave("paper/figures/fig1_te_by_family.png", fig1,
       width = 13, height = 7, dpi = 300)
cat("已保存 paper/figures/fig1_te_by_family.png\n")

# -----------------------------------------------------------------------------
# 图 2：机制与后果
# -----------------------------------------------------------------------------
# (a) 插补填出的值有多少能在编码表里查到 —— L3 几乎全部查不到
# (b) L3 与 L4 的先后，在三个条件下怎么变 —— 每次只变一个量
#
# ⚠ (b) 必须画三个条件而不是两个。只画「Tier A 无编码」对「全量有编码」
#    是混淆比较：那两个条件同时变了编码和样本量。

hit <- readRDS("output/lattice_hit.rds")
hs  <- hit[, .(hit_rate = sum(n_hit) / sum(n_imputed)), by = line]
hs[, line := factor(line, levels = c("L2", "L3", "L4"))]
hs[, lab := c("L2\nmedian", "L3\nregression", "L4\nPMM")[as.integer(line)]]

LINE_COLORS <- c("L2" = "#76B7B2", "L3" = "#E15759", "L4" = "#4E79A7")

# 标签精度用 %.4f%%：%.2f%% 会把 L2 的 100.000000% 和 L4 的 99.999033% 一起
# 印成 "100.00%"，掩盖 L4 在 206846 条里有 2 条没命中格点这件事；四位小数
# 能把 L2/L3/L4 都区分开，且与 docs/实验报告.md §10.1 的数字对得上。
p2a <- ggplot(hs, aes(x = line, y = 100 * hit_rate, fill = line)) +
  geom_col(width = 0.6, color = "black", linewidth = 0.8, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.4f%%", 100 * hit_rate)),
            vjust = -0.6, size = 6, fontface = "bold", family = FONT) +
  scale_fill_manual(values = LINE_COLORS, guide = "none") +
  scale_x_discrete(labels = hs$lab) +
  scale_y_continuous(limits = c(0, 112), breaks = seq(0, 100, 25),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "Imputed values found in the encoding table (%)",
       subtitle = "(a) Regression imputation leaves the generator's 0.01 lattice") +
  PAPER_THEME

# (b) 三条件比较
grab <- function(p) if (file.exists(p)) readRDS(p)$cv_mean else NA_real_
cond <- data.table(
  condition = factor(
    c("200k\nno encoding", "200k\nencoding", "Full 691k\nencoding"),
    levels = c("200k\nno encoding", "200k\nencoding", "Full 691k\nencoding")),
  L3 = c(grab("output/archive_pre_te/tierA_grid/meta_grid_L3_xgboost.rds"),
         grab("output/ladder/meta_pool_200k_L3_xgboost.rds"),
         grab("output/oof/meta_L3_xgboost.rds")),
  L4 = c(grab("output/archive_pre_te/tierA_grid/meta_grid_L4_xgboost.rds"),
         grab("output/ladder/meta_pool_200k_L4_xgboost.rds"),
         grab("output/oof/meta_L4_xgboost.rds")))
stopifnot("图 2b 缺数据" = !anyNA(cond$L3) && !anyNA(cond$L4))

cl <- melt(cond, id.vars = "condition", variable.name = "line", value.name = "auc")

# 在「200k, encoding」这一条件下 L3/L4 的 cv_mean 只差 0.00076，若两条线的
# 数值标签都固定摆在各自点的正上方，会在这一格叠成一团读不出来。改成看
# 哪条线在该条件下更高就把标签摆在上面、更低的摆在下面——这样每个条件里
# 两个标签总是分处点的两侧，顺带也把第三个条件里 L3（此时是最低点）的
# 标签从「贴着来向的连线」挪到了点下方，避开了线段。
cl[, is_top := auc == max(auc), by = condition]

# is_top 只解决"同一条件下两条线的标签互相叠"这一种碰撞（在中间条件最要紧，
# 那里 L3/L4 只差 0.00076）。它没管另一种碰撞：某个点自己的线段，会不会正好
# 伸进标签摆放的那一侧——这只可能发生在两端的条件，因为端点的点只有一段唯一
# 相邻的线段，其方向完全由数据决定；中间条件左右都有相邻段，水平方向无处可
# 避，不处理。做法：按 condition 排序后用 data.table::shift() 取每条线在端点
# 唯一相邻点的 auc，判断"这段线贴着点的上方还是下方"——最左条件看的是线离开
# 该点去下一条件（比该点高就贴上方），最右条件看的是线从上一条件进入该点
# （上一条件比该点高，说明线是从上方压下来的，也贴上方）：两端点的"进/出"
# 方向相反，所以这里分别用 next_auc 和 prev_auc 直接比较，不共用一个正负号。
# 当线贴着的那一侧正好是 is_top 选中的 vjust 那一侧，就用水平方向把标签推开
# （最左条件推左、最右条件推右，即推向"没有相邻点、线段到不了"的那一边），
# 而不是翻转 vjust——翻转会重新把这条线的标签和另一条线在同一条件下的标签
# 叠到一起。
setorder(cl, line, condition)
cl[, next_auc := shift(auc, 1L, type = "lead"), by = line]
cl[, prev_auc := shift(auc, 1L, type = "lag"), by = line]
n_cond <- nlevels(cl$condition)
x_idx  <- as.integer(cl$condition)
cl[, seg_occupies_top := fifelse(x_idx == 1L, next_auc > auc,
                           fifelse(x_idx == n_cond, prev_auc > auc, NA))]
cl[, label_conflict := !is.na(seg_occupies_top) & (is_top == seg_occupies_top)]
cl[, label_nudge_x := fifelse(!label_conflict, 0,
                        fifelse(x_idx == 1L, -0.16, 0.16))]

p2b <- ggplot(cl, aes(x = condition, y = auc, group = line, color = line)) +
  geom_line(linewidth = 1.6) +
  geom_point(size = 5, shape = 21, fill = "white", stroke = 1.6) +
  geom_text(aes(label = sprintf("%.5f", auc), vjust = ifelse(is_top, -1.2, 1.8)),
            position = position_nudge(x = cl$label_nudge_x),
            size = 5, fontface = "bold", family = FONT,
            show.legend = FALSE) +
  scale_color_manual(values = LINE_COLORS[c("L3", "L4")],
                     name = NULL,
                     labels = c("L3 (regression)", "L4 (PMM)")) +
  scale_y_continuous(expand = expansion(mult = c(0.10, 0.14))) +
  labs(x = NULL, y = "CV AUC (xgboost)",
       subtitle = paste0("(b) Encoding halves L3's lead at matched n; ",
                         "the reversal needs full data as well")) +
  PAPER_THEME +
  theme(legend.position = c(0.02, 0.06), legend.justification = c(0, 0))

fig2 <- p2a + p2b + plot_layout(widths = c(1, 1.35))
ggsave("paper/figures/fig2_lattice_mechanism.png", fig2,
       width = 16, height = 7, dpi = 300)
cat("已保存 paper/figures/fig2_lattice_mechanism.png\n")
