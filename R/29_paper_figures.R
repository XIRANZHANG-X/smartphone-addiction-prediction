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
