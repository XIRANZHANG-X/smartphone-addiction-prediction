# Preprocessing-Expressiveness Paper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 产出一篇英文课程论文（Markdown）及其三张图与全部支撑脚本，论断为「一项预处理的价值等于该变换有多难被下游模型自己构造出来」。

**Architecture:** 论文的每一个数字都必须来自本仓库某个脚本的产物。先补齐缺失的证据脚本（格点命中率目前无脚本），再画三张图，再写正文，最后用一个自动核对脚本验证正文里的数字与产物一致。

**Tech Stack:** R 4.6.1（绝对路径 `D:/R/R-4.6.1/bin/Rscript.exe`）、data.table、ggplot2、patchwork。论文为 Markdown。

## Global Constraints

- R 不在 PATH，一律用 `"D:/R/R-4.6.1/bin/Rscript.exe"` 调用。
- 工作目录必须是 `C:/Users/Lenovo/Desktop/project`；每条 Bash 命令自带 `cd`（后台命令会重置到 D 盘）。
- **不得引入 `docs/实验报告.md` 中没有的数字。** 新测的数字必须先写进报告再进论文。
- **不得编造引用。** 白名单见 spec §10，白名单外一律 `[CITATION NEEDED]`。
- 论文语言为英文；作者处留 `[Author Name]` / `[Student ID]` / `[Affiliation]`，不代填。
- 赛后数字一律标注「no leaderboard verification」。外推值必须写明是外推。
- 绘图规范照 `C:\Users\Lenovo\Desktop\MH6211_ANALYTICS_SOFTWARE_1\asignment\R代码整理\绘图规范与总结.md`：
  `dpi = 300`；全局 `face = "bold"`；Times New Roman；
  `panel.border = element_rect(color = "black", fill = NA, linewidth = 2)`；
  `axis.ticks = element_line(color = "black", linewidth = 1)`；
  轴标题 18–30 / 刻度 14–22 / 图例 14–18；
  配色 `#4E79A7` `#F28E2B` `#E15759` `#76B7B2` `#59A14F`，用**命名向量**传 `scale_*_manual`；
  网格线 `linetype = "dashed", color = "grey70", linewidth = 0.6`；参考线一律虚线。
- 写 R 脚本用 Write 工具，不用 bash heredoc。
- 每个任务结束时提交一次。

---

### Task 1: 补齐格点命中率的证据脚本

论文第 8 节最核心的机制证据是「L3 填出的值查不到编码表」，量化指标是命中率
0.03% / 100% / 99.98%。**这三个数字目前没有任何脚本产出它们**——是上一轮会话
临时算的，只有结果留在 `docs/讨论区核查.md`。论文不能引用不可复现的测量。

**Files:**
- Create: `R/30_lattice_hit.R`
- Create (产物): `output/lattice_hit.rds`

**Interfaces:**
- Produces: `output/lattice_hit.rds`，一个 data.table，列为
  `line`（"L2"/"L3"/"L4"）、`col`（被插补的列名）、`fold`、`n_imputed`、`n_hit`、`hit_rate`。

- [ ] **Step 1: 写脚本**

```r
# =============================================================================
# 30_lattice_hit.R —— 插补填出的值，有多少能在编码表里查到
#
# 用法：Rscript R/30_lattice_hit.R
# 产出：output/lattice_hit.rds
#
# -----------------------------------------------------------------------------
# 为什么需要这个脚本
# -----------------------------------------------------------------------------
# 「两个预处理互相破坏」这条发现的机制证据是命中率：L3 用回归预测填补，
# 填出的值是任意实数，不在生成器的 0.01 格点上，逐取值编码查不到，
# 只能回落到全局均值。而至少缺一列的行占 61%。
#
# 这三个数字此前是临时算出来的，没有脚本。论文要引用它们，就必须能重算。
#
# -----------------------------------------------------------------------------
# 口径
# -----------------------------------------------------------------------------
# 对每一折、每一条插补线、每一个被编码的列：
#   分母 = 该列在**验证折**中原本缺失、因而被插补器填过的行数
#   分子 = 这些行填出的值，能在**训练折拟合出的编码表**里查到的行数
# 「查到」的定义与 apply_target_encoder() 完全一致：值出现在 enc$maps[[col]]$v 中。
#
# L1 不在此列 —— 它根本不填，缺失就留着缺失。
# =============================================================================

suppressMessages({library(data.table)})
source("R/03_features.R")

SEED <- 20260821L
LINES <- c("L2", "L3", "L4")

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")

train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

cat(sprintf("池 %s 行，%d 折，%d 个编码列\n\n",
            format(nrow(X_pool), big.mark = ","),
            length(unique(f_pool)), length(TE_COLS)))

rows <- list()

for (line in LINES) {
  source(sprintf("R/05_impute_%s.R", line))
  fit_i   <- get(paste0("fit_imputer_",   line))
  apply_i <- get(paste0("apply_imputer_", line))

  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)

    # 1. 记下验证折里哪些格子原本是缺的 —— 插补之后就看不出来了
    was_na <- lapply(setNames(TE_COLS, TE_COLS),
                     function(cc) is.na(X_pool[[cc]][va]))

    # 2. 走与建模完全相同的路径：插补器只在训练折拟合
    imp <- fit_i(X_pool[tr])
    A   <- apply_i(imp, data.table::copy(X_pool[tr]))
    B   <- apply_i(imp, data.table::copy(X_pool[va]))

    # 3. 编码表也只在训练折拟合
    A <- derive_features(A)
    enc <- fit_target_encoder(A, y_pool[tr])

    for (cc in TE_COLS) {
      idx <- which(was_na[[cc]])
      if (!length(idx)) next
      filled <- B[[cc]][idx]
      known  <- enc$maps[[cc]]$v
      n_hit  <- sum(filled %in% known)
      rows[[length(rows) + 1L]] <- data.table(
        line = line, col = cc, fold = k,
        n_imputed = length(idx), n_hit = n_hit,
        hit_rate = n_hit / length(idx))
    }
    cat(sprintf("  %s 第 %d 折完成\n", line, k))
  }
}

res <- rbindlist(rows)
saveRDS(res, "output/lattice_hit.rds")

cat("\n============ 按插补线汇总 ============\n")
s <- res[, .(n_imputed = sum(n_imputed), n_hit = sum(n_hit)), by = line]
s[, hit_rate := n_hit / n_imputed]
for (i in seq_len(nrow(s)))
  cat(sprintf("  %-3s 被插补 %s 个格子，查到 %s 个，命中率 %.4f%%\n",
              s$line[i], format(s$n_imputed[i], big.mark = ","),
              format(s$n_hit[i], big.mark = ","), 100 * s$hit_rate[i]))

cat("\n============ 逐列命中率（%）============\n")
w <- dcast(res[, .(hit_rate = sum(n_hit) / sum(n_imputed)), by = .(line, col)],
           col ~ line, value.var = "hit_rate")
print(w[, lapply(.SD, function(z) if (is.numeric(z)) round(100 * z, 4) else z)])

cat("\n已保存 output/lattice_hit.rds\n")
```

- [ ] **Step 2: 跑，看数字**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/30_lattice_hit.R 2>&1 | tail -30
```
Expected: 三行汇总，L2 接近 100%、L3 远低于 1%、L4 接近 100%。
**若与记录的 0.03–0.19% / 100% / 99.98% 明显不符，以本次重算为准**，
并在 `docs/实验报告.md` §10.1 与本计划的后续任务中同步更正。

- [ ] **Step 3: 把重算值写进报告 §10.1 的命中率表**

用 Edit 工具改 `docs/实验报告.md`，把命中率表换成本次重算的数字，
并在表下加一句：`脚本 R/30_lattice_hit.R`。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/30_lattice_hit.R output/results.md docs/实验报告.md && git commit -m "补齐格点命中率的证据脚本：此前只有数字没有脚本"
```

---

### Task 2: 修 fig1 的陈旧副标题

`R/12_figures.R` 的 fig1 副标题仍写着**三项**约束，而正确的是四项。
这张图现在画的是一个已被推翻的结论。

**Files:**
- Modify: `R/12_figures.R:106`
- Regenerate: `reports/figures/fig1_硬结构约束.png`

- [ ] **Step 1: 确认当前内容**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && sed -n '100,112p' R/12_figures.R
```
Expected: 看到 `subtitle = "每日屏幕时间 ≥ 社交媒体时间 + 游戏时间，在完整数据上 100% 成立"`。

- [ ] **Step 2: 改成四项**

用 Edit 工具把那一行改为：

```r
    subtitle = "每日屏幕时间 ≥ 社交媒体时间 + 游戏时间 + 工作学习时间，在 421,427 行上 100% 成立（残差最小值 0.000）",
```

同时检查该图所画的**数据**是否也是三项残差。若 `12_figures.R` 里
计算的是 `daily - (social + gaming)`，一并改成四项
`daily - (social_media_hours + gaming_hours + work_study_hours)`。

- [ ] **Step 3: 重绘并确认**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/12_figures.R 2>&1 | tail -5 && ls -la reports/figures/fig1_硬结构约束.png
```
Expected: 退出码 0，fig1 的 mtime 是刚才。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/12_figures.R reports/figures/ && git commit -m "修 fig1：副标题与数据仍是被推翻的三项约束"
```

---

### Task 3: 图 1 —— TE 收益按模型族

**Files:**
- Create: `R/29_paper_figures.R`（本任务只写文件头与图 1；图 2、3 在后续任务追加）
- Create (产物): `paper/figures/fig1_te_by_family.png`

**Interfaces:**
- Consumes: `output/te_by_family.rds`，结构为 `list(xgboost = list(off = num[5], on = num[5]), lightgbm = ..., ranger = ..., glmnet = ...)`
- Produces: `PAPER_THEME`（一个 ggplot2 theme 对象）与 `ALGO_COLORS`（命名字符向量），图 2、3 复用

- [ ] **Step 1: 写脚本（文件头 + 共用主题 + 图 1）**

```r
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

# Windows 下注册 Times New Roman；非 Windows 或缺字体时回落到默认无衬线
FONT <- tryCatch({
  windowsFonts(Times = windowsFont("Times New Roman")); "Times"
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
       width = 10, height = 7, dpi = 300)
cat("已保存 paper/figures/fig1_te_by_family.png\n")
```

- [ ] **Step 2: 跑并检查数字**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/29_paper_figures.R && ls -la paper/figures/
```
Expected: 生成 `fig1_te_by_family.png`。

- [ ] **Step 3: 用 Read 工具看图，确认四件事**

打开 `paper/figures/fig1_te_by_family.png`，逐项确认：
1. 四根柱的数值标签是 `+0.00424` / `+0.00475` / `+0.02039` / `+0.03352`；
2. 全部文字是英文，没有中文残留；
3. 面板有黑色粗边框，网格是浅灰虚线；
4. 每根柱上有 5 个散点（逐折）。

任何一项不符就回到 Step 1 修。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/29_paper_figures.R paper/figures/fig1_te_by_family.png && git commit -m "论文图 1：TE 收益按模型族，跨越一个数量级"
```

---

### Task 4: 图 2 —— 格点机制与 L3/L4 的三条件比较

**Files:**
- Modify: `R/29_paper_figures.R`（追加图 2 一节）
- Create (产物): `paper/figures/fig2_lattice_mechanism.png`

**Interfaces:**
- Consumes: `output/lattice_hit.rds`（Task 1 产出）；
  `output/archive_pre_te/tierA_grid/meta_grid_L{3,4}_xgboost.rds`；
  `output/ladder/meta_pool_200k_L{3,4}_xgboost.rds`；
  `output/oof/meta_L{3,4}_xgboost.rds`
- Consumes: `PAPER_THEME`（Task 3 定义）

- [ ] **Step 1: 追加图 2 代码到 `R/29_paper_figures.R` 末尾**

```r
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

p2a <- ggplot(hs, aes(x = line, y = 100 * hit_rate, fill = line)) +
  geom_col(width = 0.6, color = "black", linewidth = 0.8, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.2f%%", 100 * hit_rate)),
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

p2b <- ggplot(cl, aes(x = condition, y = auc, group = line, color = line)) +
  geom_line(linewidth = 1.6) +
  geom_point(size = 5, shape = 21, fill = "white", stroke = 1.6) +
  geom_text(aes(label = sprintf("%.5f", auc)),
            vjust = -1.2, size = 5, fontface = "bold", family = FONT,
            show.legend = FALSE) +
  scale_color_manual(values = c("L3" = "#E15759", "L4" = "#4E79A7"),
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
```

- [ ] **Step 2: 跑**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/29_paper_figures.R 2>&1 | tail -5
```
Expected: 两张图都保存成功，无 `stopifnot` 报错。

- [ ] **Step 3: 用 Read 工具看图，确认**

1. (a) 三根柱，L3 那根几乎贴地，数值标签与 Task 1 的重算值一致；
2. (b) 两条线在第三个条件处**交叉**，前两个条件 L3 在上；
3. 全英文，无中文；
4. (b) 的图例在左下且不遮挡数据。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/29_paper_figures.R paper/figures/fig2_lattice_mechanism.png && git commit -m "论文图 2：格点命中率与 L3/L4 的三条件比较"
```

---

### Task 5: 图 3 —— 样本量阶梯

**Files:**
- Modify: `R/29_paper_figures.R`（追加图 3 一节）
- Create (产物): `paper/figures/fig3_size_ladder.png`

**Interfaces:**
- Consumes: `output/size_ladder.rds`，结构为
  `list(summary = data.table(rung, n, n_cell, spearman, kendall, pick, hit, regret, ...), wide = data.table(model, "50k", "100k", "200k", "400k", "full"), ranks = ..., topk = data.table(rung, top1, top3, top5), raw = ...)`

- [ ] **Step 1: 追加图 3 代码到 `R/29_paper_figures.R` 末尾**

```r
# -----------------------------------------------------------------------------
# 图 3：样本量阶梯
# -----------------------------------------------------------------------------
# (a) 十个候选的 AUC 随样本量怎么走 —— 三条 L3 线是全表唯一下降的
# (b) 与全量排名的秩相关随样本量上升；选择遗憾恒为 0

lad <- readRDS("output/size_ladder.rds")
N_BY_RUNG <- c("50k" = 50000, "100k" = 100000, "200k" = 200000,
               "400k" = 400000, "full" = 691369)

w <- copy(lad$wide)
long <- melt(w, id.vars = "model", variable.name = "rung", value.name = "auc")
long[, n := N_BY_RUNG[as.character(rung)]]
# L3 的三个树模型格是唯一随样本量下降的，单独着色
long[, family := fifelse(model %in% c("L3_xgboost", "L3_lightgbm", "L3_ranger"),
                         "L3 + tree model (declines with n)", "All other candidates")]

p3a <- ggplot(long, aes(x = n, y = auc, group = model, color = family)) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 2.6) +
  scale_color_manual(values = c("L3 + tree model (declines with n)" = "#E15759",
                                "All other candidates" = "#4E79A7"),
                     name = NULL) +
  scale_x_continuous(trans = "log10",
                     breaks = N_BY_RUNG,
                     labels = c("50k", "100k", "200k", "400k", "691k")) +
  labs(x = "Training pool size (log scale)", y = "CV AUC",
       subtitle = "(a) Three cells get worse with more data - all of them L3") +
  PAPER_THEME +
  theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0))

s <- copy(lad$summary)
sl <- melt(s[, .(n, spearman, kendall)], id.vars = "n",
           variable.name = "metric", value.name = "rho")
sl[, metric := factor(metric, levels = c("spearman", "kendall"),
                      labels = c("Spearman rho", "Kendall tau"))]

p3b <- ggplot(sl, aes(x = n, y = rho, group = metric,
                      color = metric, shape = metric)) +
  geom_hline(yintercept = 1, linetype = "dashed",
             color = "grey40", linewidth = 0.9) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 5, fill = "white", stroke = 1.5) +
  scale_color_manual(values = c("Spearman rho" = "#4E79A7",
                                "Kendall tau"  = "#F28E2B"), name = NULL) +
  scale_shape_manual(values = c(21, 24), name = NULL) +
  scale_x_continuous(trans = "log10", breaks = N_BY_RUNG,
                     labels = c("50k", "100k", "200k", "400k", "691k")) +
  scale_y_continuous(limits = c(0.75, 1.03), breaks = seq(0.8, 1.0, 0.05)) +
  annotate("label", x = 100000, y = 0.79,
           label = "Selection regret = 0.00000 at every size\ntop-1, top-3, top-5 sets all match",
           size = 5.2, fontface = "bold", family = FONT,
           label.size = 0.8, fill = "white") +
  labs(x = "Training pool size (log scale)",
       y = "Rank correlation with the full-data ranking",
       subtitle = "(b) Ranking converges; the selection is already correct at 50k") +
  PAPER_THEME +
  theme(legend.position = c(0.98, 0.30), legend.justification = c(1, 0))

fig3 <- p3a + p3b
ggsave("paper/figures/fig3_size_ladder.png", fig3,
       width = 17, height = 7, dpi = 300)
cat("已保存 paper/figures/fig3_size_ladder.png\n")
```

- [ ] **Step 2: 跑三张图**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/29_paper_figures.R 2>&1 | tail -6 && ls -la paper/figures/
```
Expected: 三张 png 都在。

- [ ] **Step 3: 用 Read 工具看图，确认**

1. (a) 红色的三条线在 10 万之后确实下行，蓝色的七条一路上行；
2. (b) 两条线单调上升到 1.0，标注框不遮挡数据点；
3. 横轴是对数刻度且刻度标签是 50k/100k/200k/400k/691k；
4. 全英文。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/29_paper_figures.R paper/figures/ && git commit -m "论文图 3：样本量阶梯，秩相关收敛而选择在 5 万时已正确"
```

---

### Task 6: 论文骨架 + §1–3（Introduction / Data / Protocol）

**Files:**
- Create: `paper/preprocessing-expressiveness.md`

- [ ] **Step 1: 重读 spec 的 §3.1 关键数字表**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && sed -n '/^### 3.1 关键数字/,/^---/p' docs/superpowers/specs/2026-09-03-paper-preprocessing-expressiveness-design.md
```
写正文时每引一个数字都回查这张表。

- [ ] **Step 2: 写标题、作者占位、摘要、§1–3**

用 Write 工具建 `paper/preprocessing-expressiveness.md`，内容包含：

- 标题、`[Author Name]` / `[Student ID]` / `[Affiliation]` 占位、Abstract（150–200 词）
- **§1 Introduction**：设定；「哪种插补最好」是欠定问题；三条贡献
- **§2 Data and Task**：691,369 / 296,302 行、12 特征、正例率 0.7094、61.06% 缺失；
  生成器的 0.01 格点与四项约束（421,427 行 100.0000%，残差最小值 0.000）；
  MCAR 验证（最大差 0.0042）；比赛已结束、赛后数字无榜单验证
- **§3 Experimental Protocol**：冻结折叠（seed 20260821，5 折）；折内拟合；
  配对统计（Cohen's d + 逐折同号，不用 n=5 的 p 值）；安慰剂列（+0.00003、2/5）；
  分辨率下限是配对的性质（0.000098 对 0.000564）；
  **可复现性声明**（lightgbm 三次 0.96039/0.96043/0.96039，偏差 4e-5，低于所有下限）

- [ ] **Step 3: 检查没有中文残留**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && grep -nP '[\x{4e00}-\x{9fa5}]' paper/preprocessing-expressiveness.md | head
```
Expected: 无输出。有则改掉。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "论文 §1-3：引言、数据、实验方案"
```

---

### Task 7: §4–7（网格 + 三个实例）

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`

- [ ] **Step 1: 追加 §4–7**

- **§4 The 4x4 Grid**：14 格全量表（数值见 `docs/实验报告.md` §4.2）；
  树上 L1>L2>L4>L3、glmnet 上 L3>L2>L4；L3 三格的折间 sd 是其余的 4–7 倍
- **§5 Instance 1: Imputation**（引图无）：n=15 配对检验
  L1 vs L2 +0.00039 / d 3.42 / 15/15；L2 vs L3 +0.01273 / d 7.74 / 15/15；
  **glmnet 上 L3 vs L2 +0.00617 / d 21.46 / 15/15，符号相反**
- **§6 Instance 2: Derived Features**（**如实写成较弱的第三个实例**）：
  三项 vs 四项，残差最小值 0.100 对 0.000 才是判据；
  在 TE 之上 +0.00064 / d 3.06 / 5/5，而 max_bin +0.00003 / 2/5、
  小数位 +0.00005 / 3/5 被完全吸收
- **§7 Instance 3: Exact-Value Encoding**（引 Figure 1）：
  +0.00424 / +0.00475 / +0.02039 / +0.03352；
  one-hot 对照（Tier A 0.95583、全量 0.95929）证明线性模型能做查找，
  只是需要为每个取值配一个参数——**并注明这个对照与讨论区第 26 帖不同口径**
  （他 10 折含交互，我们 5 折不含）

- [ ] **Step 2: 逐数字回查**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && grep -oE '0\.9[0-9]{4}' paper/preprocessing-expressiveness.md | sort -u | head -40
```
把输出的每个数字在 `docs/实验报告.md` 里搜一遍，确认都能找到。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "论文 §4-7：4x4 网格与三个实例"
```

---

### Task 8: §8–9（破坏性交互 + 选择迁移）

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`

- [ ] **Step 1: 追加 §8–9**

- **§8 When Two Preprocessing Steps Destroy Each Other**（引 Figure 2）：
  机制；命中率（用 Task 1 的重算值）；
  **预注册预测及其限定条件**——匹配样本量下编码只把 L3 的领先减半
  （−0.00160 → −0.00076），**没有反转**；反转需要编码与足够数据量同时具备
  （全量 +0.00673）。**必须写清这一点，不得只报第三个条件。**
  L3 三格随 n 下降、L4 三格随 n 上升，是同一机制的第二个预测
- **§9 Does Model Selection Transfer from a Subsample?**（引 Figure 3）：
  阶梯设计（嵌套池、20 万那级就是冻结子样本、折号沿用）；
  三个指标（Spearman、top-k、选择遗憾）；
  五级全部 top-1 命中、遗憾 0.00000、top-5 集合一致；
  7 对交换的逐对分辨率下限：1 对是平局（0.00009 < 0.00011），
  6 对全部涉及 L3 三格；
  集成权重迁移代价上界 +0.00003、符号 10/10 一致；
  边界：只对参数个数固定的方法成立（one-hot 反例 0.0035 纯来自样本量）

- [ ] **Step 2: 确认 §8 写了限定条件**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && grep -nc "matched" paper/preprocessing-expressiveness.md
```
Expected: ≥ 1。§8 必须出现「matched sample size」这一层论证；没有就是漏了。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "论文 §8-9：破坏性交互与子样本选择的迁移性"
```

---

### Task 9: §10–11 + 参考文献 + 自动核对

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`
- Create: `R/31_check_paper_numbers.R`

**Interfaces:**
- Consumes: `paper/preprocessing-expressiveness.md`；`output/` 下的全部 meta/结果 rds
- Produces: 打印一份「论文里出现但产物里找不到」的数字清单；有则非零退出

- [ ] **Step 1: 追加 §10–11 与参考文献**

- **§10 Discussion**：统一表述；可操作建议（先问下游模型能不能自己做到；
  两个可能互相干扰的预处理，其组合必须在全量验证一次）；
  **五条效度威胁**（单一合成数据集、阶梯只含 10 个候选、未验证超参数迁移、
  未做 10 折对照、one-hot 对照口径不等价）
- **§11 Conclusion**
- **Acknowledgements**：四项约束、逐取值编码、秩空间 stack 三项想法来自
  竞赛讨论区；我们做的是在自己的冻结折叠上重新测量
- **References**：只列 spec §10 白名单的 14 条；其余 `[CITATION NEEDED]`

- [ ] **Step 2: 写核对脚本**

```r
# =============================================================================
# 31_check_paper_numbers.R —— 论文里的每个 AUC 数字，都必须能在产物里找到
#
# 用法：Rscript R/31_check_paper_numbers.R
#
# 本会话已经三次抓到文档与产物的数字不一致（lightgbm 的 +0.00475、
# 十成员对十四成员的集成分、L3/L4 的混淆比较）。人工核对不可靠，改成自动的。
#
# 做法：把论文里形如 0.9xxxx 的数字全抽出来，与产物里能算出的全部 AUC
# 比对（容差 5e-6，只为吸收打印舍入）。找不到的逐个列出，由人判断
# 是笔误还是一个本来就没有产物支撑的数字 —— 两种都必须处理。
# =============================================================================

suppressMessages({library(data.table)})

PAPER <- "paper/preprocessing-expressiveness.md"
TOL   <- 5e-6

stopifnot("找不到论文" = file.exists(PAPER))

# ---- 1. 收集产物里的全部 AUC -----------------------------------------------
known <- numeric(0)
add <- function(v) known <<- c(known, as.numeric(v[!is.na(v)]))

metas <- c(list.files("output/oof",    "^meta_.*\\.rds$", full.names = TRUE),
           list.files("output/ladder", "^meta_.*\\.rds$", full.names = TRUE),
           list.files("output/repeat", "\\.rds$",         full.names = TRUE),
           list.files("output/archive_pre_te/tierA_grid",
                      "^meta_.*\\.rds$", full.names = TRUE))
for (f in metas) {
  m <- readRDS(f)
  add(m$cv_mean); add(m$oof_auc); add(m$fold_auc)
}

if (file.exists("output/te_by_family.rds")) {
  te <- readRDS("output/te_by_family.rds")
  for (a in names(te)) { add(te[[a]]$on); add(te[[a]]$off) }
}
if (file.exists("output/size_ladder.rds")) {
  lad <- readRDS("output/size_ladder.rds")
  cols <- setdiff(names(lad$wide), "model")
  for (cc in cols) add(lad$wide[[cc]])
}
if (file.exists("output/ensemble_best.rds")) {
  en <- readRDS("output/ensemble_best.rds")
  add(unlist(en$all_scores)); add(en$cv_auc); add(en$single_aucs)
}
for (f in c("output/feature_v2.rds", "output/cheap_wins.rds",
            "output/onehot_lr.rds", "output/alpha_scan.rds",
            "output/ablation.rds", "output/tune_xgboost.rds")) {
  if (!file.exists(f)) next
  x <- readRDS(f)
  add(unlist(rapply(x, function(z) if (is.numeric(z)) z else NA_real_,
                    how = "unlist")))
}
known <- unique(round(known, 8))
cat(sprintf("产物里收集到 %d 个不同的数值\n", length(known)))

# ---- 2. 抽论文里的数字 -------------------------------------------------------
txt <- paste(readLines(PAPER, encoding = "UTF-8", warn = FALSE), collapse = "\n")
found <- unique(as.numeric(regmatches(txt,
           gregexpr("0\\.9[0-9]{4}", txt))[[1]]))
cat(sprintf("论文里出现 %d 个形如 0.9xxxx 的数字\n\n", length(found)))

# ---- 3. 比对 -----------------------------------------------------------------
missing <- found[vapply(found, function(v) !any(abs(known - v) < TOL), logical(1))]

if (!length(missing)) {
  cat("全部数字都能在产物里找到。\n")
} else {
  cat("★ 以下数字在产物里找不到，逐个查明是笔误还是缺证据：\n")
  for (v in sort(missing)) {
    ctx <- regmatches(txt, gregexpr(sprintf(".{0,70}%s.{0,40}",
                      format(v, nsmall = 5)), txt))[[1]]
    cat(sprintf("\n  %.5f\n", v))
    for (c1 in head(ctx, 2)) cat("    ...", gsub("\n", " ", c1), "...\n")
  }
  quit(save = "no", status = 1)
}
```

- [ ] **Step 3: 跑核对**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/31_check_paper_numbers.R
```
Expected: `全部数字都能在产物里找到。`
若列出数字，逐个查明：是论文笔误（改论文），还是引了一个没有产物支撑的数字
（要么补脚本，要么删掉这个说法）。**不允许放着不管。**

- [ ] **Step 4: 全篇门禁**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && grep -nP '[\x{4e00}-\x{9fa5}]' paper/preprocessing-expressiveness.md | head && grep -c "CITATION NEEDED" paper/preprocessing-expressiveness.md && wc -w paper/preprocessing-expressiveness.md
```
Expected: 无中文；`[CITATION NEEDED]` 的条数已知；词数在 4000–6000 之间。

- [ ] **Step 5: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/ R/31_check_paper_numbers.R && git commit -m "论文 §10-11、参考文献，以及一个自动核对数字的门禁脚本"
```

---

## Self-Review

**Spec 覆盖检查：**

| Spec 要求 | 对应任务 |
|---|---|
| §3 十一节结构 | Task 6（1–3）、7（4–7）、8（8–9）、9（10–11） |
| §3.1 关键数字表 | Task 6 Step 1 强制回查；Task 9 自动核对 |
| §4 三张图 + 绘图规范 | Task 3、4、5；规范写进 Global Constraints |
| §5 明确不写的 | Task 9 §10 的效度威胁；赛后无榜单验证写进 Task 6 §2 |
| §6 两处弱点 | Task 7（§6 写成较弱实例）、Task 8（§9 为独立贡献） |
| §7 五条效度威胁 | Task 9 Step 1 |
| §8 交付物 | Task 1（30_lattice_hit.R）、3–5（29_paper_figures.R + 三图）、6–9（论文） |
| §9 fig1 陈旧副标题 | Task 2 |
| §10 参考文献白名单 | Task 9 Step 1 |

**Spec 之外新增的两项，理由：**

- **Task 1**：spec §8 已加入，因为命中率此前无脚本，论文不能引用不可复现的测量。
- **Task 9 的 `31_check_paper_numbers.R`**：本会话三次抓到数字不一致，人工核对已被证明不可靠。

**类型一致性：** `PAPER_THEME`、`ALGO_COLORS`、`FONT` 在 Task 3 定义，Task 4、5 复用，
名字一致；`output/lattice_hit.rds` 的列名在 Task 1 的 Interfaces 声明，Task 4 按该列名读取。

**占位符扫描：** 无 TBD/TODO。`[Author Name]`、`[CITATION NEEDED]` 是交付物中有意保留的占位，
已在 Global Constraints 说明不得代填。
