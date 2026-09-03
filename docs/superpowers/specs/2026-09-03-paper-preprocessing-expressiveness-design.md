# 学术论文 — 设计文档 (Design Spec)

- **日期**：2026-09-03
- **产出**：一篇英文课程论文，Markdown 稿 + 三张新图 + 绘图脚本
- **载体项目**：`XIRANZHANG-X/smartphone-addiction-prediction`
- **状态**：设计已经用户确认，待实施

---

## 0. 这份文档定什么

定这篇论文**写什么、不写什么、证据从哪来**。不定具体措辞。

全部数字来自本仓库脚本在冻结折叠上的产出，统一汇总在
[docs/实验报告.md](../../实验报告.md)。**论文不得引入任何该文档中没有的数字。**

---

## 1. 决策记录（用户已确认）

| 项 | 决定 |
|---|---|
| 主题 | 手机成瘾预测项目的方法学发现（不是 D 盘那个 NLP 项目） |
| 语言 | 英文 |
| 去处 | 课程论文 / 大作业 |
| 中心论断 | A 为主线，C 单独成节，B 的纪律进方法部分 |
| 格式 | 先写 Markdown，定稿后再转 |
| 图表 | 只做论文真正需要的 3 张新图，全英文标签 |
| 绘图规范 | 参照 `C:\Users\Lenovo\Desktop\MH6211_ANALYTICS_SOFTWARE_1\asignment\R代码整理\绘图规范与总结.md` |
| 作者 | 留占位符，不代填 |
| 引用 | 只写可核实的；拿不准的标 `[CITATION NEEDED]`；**不编造任何引用** |
| 讨论区 | 写进 Acknowledgements 与脚注，不隐瞒来源 |

---

## 2. 中心论断

> **一项预处理的价值，等于该变换有多难被下游模型自己构造出来。**

两个推论：

1. **同一变换在不同模型族上可以符号相反。** 「能不能自己构造出来」是模型
   **表达能力**的性质，不是数据的性质，所以它随模型族变。
2. **两个预处理步骤会互相破坏**——当其一的输出离开了另一其所要求的定义域。
   这一条是本项目的新发现，此前未见有人提出。

---

## 3. 章节安排与证据来源

每一节都必须指明证据脚本。**没有脚本支撑的说法不写进论文。**

| 节 | 内容 | 证据 |
|---|---|---|
| 1 | Introduction。问题设定；「哪种插补最好」是欠定问题；三条贡献 | — |
| 2 | Data and Task。69 万行、61% 缺失、完整时标签近乎确定；生成器的 0.01 格点与四项约束；MCAR 验证 | `R/02_eda.R`、`R/17_discussion_checks.R` |
| 3 | **Experimental Protocol**。冻结折叠；折内拟合；配对统计（Cohen's d + 逐折同号，而非 n=5 的 p 值）；安慰剂列；**分辨率下限是配对的性质**；**可复现性声明**（lightgbm 不可逐位复现，偏差 4e-5，低于所有下限） | `R/04_folds.R`、`R/03_features.R` 的 `prepare_fold()`、`R/18_new_features.R`、`R/23_resolution.R`、docs/实验报告.md §2.4 |
| 4 | The 4×4 Grid。14 格全量结果 | `R/run_grid_full.R` |
| 5 | Instance 1: Imputation。GBDT 上 L2>L3、线性模型上 L3>L2 | `R/09_repeated_cv.R`（n=15） |
| 6 | Instance 2: Derived Features。三项 vs 四项 | `R/20_feature_v2.R` |
| 7 | Instance 3: Exact-Value Encoding。8 倍差；one-hot 对照 | `R/21_te_by_family.R`、`R/24_onehot_lr.R` |
| 8 | **Destructive Interaction**。格点命中率；预注册预测被证实 | `R/run_grid_full.R`、docs/实验报告.md §10.1 |
| 9 | **Does Selection Transfer?** 五级阶梯；7 对交换的逐对下限 | `R/25_size_ladder.R`、`R/28_ladder_pairs.R`、`R/27_weight_transfer.R` |
| 10 | Discussion。统一表述；可操作建议；效度威胁 | — |
| 11 | Conclusion | — |

### 3.1 关键数字（论文中必须与此一致）

| 事实 | 数值 |
|---|---|
| 训练 / 测试行数 | 691,369 / 296,302 |
| 正例率 | 0.7094 |
| 至少缺一列的行 | 61.06% |
| 四项约束成立 | 421,427 行中 100.0000%，残差最小值 0.000（三项版 0.100） |
| MCAR 最大差 | 0.0042 |
| 最优单模型 | L1_xgboost 0.96784 ± 0.00053 |
| 集成（14 成员，秩空间逻辑回归） | 0.96807 |
| TE 收益 | xgboost +0.00424 / lightgbm +0.00475 / ranger +0.02039 / glmnet +0.03352 |
| lightgbm 的复现性 | 三次实测 0.96039 / 0.96043 / 0.96039，最大偏离 4e-5，低于所有配对的分辨率下限 |
| L3 vs L2（glmnet，n=15） | +0.00617，d 21.46，15/15 |
| L1 vs L2（xgboost，n=15） | +0.00039，d 3.42，15/15 |
| 格点命中率 | L2 100.00% / L3 0.03–0.19% / L4 99.98–100.00% |
| L4 vs L3（xgboost 全量） | 0.95443 对 0.94770 |
| 编码前 L3 vs L4（Tier A） | 0.95951 对 0.95791（方向相反） |
| 阶梯 Spearman | 0.903 / 0.927 / 0.964 / 0.988 |
| 选择遗憾 | 五级全部 0.00000 |
| 权重迁移代价上界 | +0.00003，符号 10/10 一致 |
| 分辨率下限 | 同库孪生对 0.000098；跨模型族 0.000564 |
| one-hot + 逻辑回归 | Tier A 0.95583；全量 0.95929 |

---

## 4. 图（三张，全英文）

| 图 | 内容 | 形式 |
|---|---|---|
| **1** | TE 收益按模型族 | 分组柱 + 95% CI 误差棒 + 逐折散点。纵轴跨一个数量级 |
| **2** | (a) 格点命中率按插补线；(b) L3/L4 在编码前后的 AUC 反转 | 双面板；(b) 用配对斜线图 |
| **3** | (a) 十个候选 AUC 随 n 变化，L3 三格高亮为唯一下降的；(b) Spearman ρ 与选择遗憾随 n | 双面板折线 |

**4×4 网格用表格，不画图。**

### 4.1 绘图规范（照抄 MH6211 那份）

- `ggsave(..., dpi = 300)`，尺寸用英寸
- 全局加粗：`element_text(face = "bold")`，标题/轴标题/刻度/图例均设
- 西文用 Times New Roman：`windowsFonts(Times = windowsFont("Times New Roman"))`
- 黑色加粗面板边框：`panel.border = element_rect(color = "black", fill = NA, linewidth = 2)`
- 加粗刻度线：`axis.ticks = element_line(color = "black", linewidth = 1)`
- 字号：轴标题 18–30、刻度 14–22、图例 14–18、标题 22–26
- 配色 Tableau 10 低饱和：`#4E79A7` `#F28E2B` `#E15759` `#76B7B2` `#59A14F`
- 颜色用**命名向量**传给 `scale_*_manual`，避免顺序错乱
- 网格线虚线浅灰：`linetype = "dashed", color = "grey70", linewidth = 0.6`
- 主题 `theme_bw`（面板边框场景）或 `theme_classic`（SCI 风）
- 参考线一律 `linetype = "dashed"`

---

## 5. 明确不写的

- **不写模型部署、业务落地、伦理讨论。** 这些与论断无关。
- **不写「我们拿了第几名」。** 榜单成绩只在 Data and Task 里作为一句背景。
- **不把赛后数字说成榜单成绩。** 比赛 8/31 已结束，赛后所有数字**没有榜单验证**，
  必须逐处标注。外推值必须写明是外推。
- **不编造引用。**
- **不引入 docs/实验报告.md 之外的数字。**

---

## 6. 两处已向用户声明的弱点

**其一，第 6 节是三个实例里最弱的。**
派生特征那一条的证据是「改对定义后 +0.00064、5/5 同号」，
没有另外两个实例那样的跨模型族符号反转。论文如实写成
「a weaker third instance」，不拔高。

**其二，第 9 节不是中心论断的实例。**
子样本迁移是一篇独立的方法学贡献。放进来的理由是它同时为中心论断
提供了一个额外证据：排名错误全部集中在 L3 那几格，
而那正是「两个预处理互相破坏」的作用点。
用户已确认保留为独立一节，不压缩进 Discussion。

---

## 7. 效度威胁（必须写进 §10）

- **单一数据集，且是合成的。** 0.01 格点与四项约束都是生成器的产物。
  逐取值编码的巨大收益很可能不能迁移到真实数据。
- **阶梯只含 10 个候选，不含 L4 四格。** 第 9 节的结论适用范围是这 10 个。
- **未验证超参数选择的迁移性。** 只验证了模型选择与集成配权。
- **未做 10 折对照。** 全项目统一 5 折（折叠契约）。
- **`24_onehot_lr.R` 的对照有一处不等价**：讨论区第 26 帖用 10 折且含交互，
  我们是 5 折不含交互，所以 0.95583 与他的 0.96005 不是同一口径的比较。

---

## 8. 交付物

| 文件 | 内容 |
|---|---|
| `paper/preprocessing-expressiveness.md` | 论文正文，作者处留 `[Author Name]` / `[Student ID]` / `[Affiliation]` |
| `R/29_paper_figures.R` | 三张图的绘图脚本，照 MH6211 规范 |
| `paper/figures/fig1_te_by_family.png` | 图 1 |
| `paper/figures/fig2_lattice_mechanism.png` | 图 2 |
| `paper/figures/fig3_size_ladder.png` | 图 3 |

---

## 9. 顺带要修的一处 bug

`R/12_figures.R` 第 106 行，`fig1_硬结构约束.png` 的副标题仍写着**三项**约束
（"每日屏幕时间 ≥ 社交媒体时间 + 游戏时间"）。四项更正没有同步到绘图代码，
这张图现在画的是一个已被推翻的结论。与本次一并修掉并重绘。

---

## 10. 参考文献白名单

只写以下这些（我有把握的），其余一律 `[CITATION NEEDED]` 留给用户补：

| 主题 | 文献 |
|---|---|
| XGBoost | Chen & Guestrin, KDD 2016 |
| LightGBM | Ke et al., NeurIPS 2017 |
| ranger | Wright & Ziegler, J. Stat. Softw. 2017 |
| glmnet | Friedman, Hastie & Tibshirani, J. Stat. Softw. 2010 |
| Random forests | Breiman, Mach. Learn. 2001 |
| Target encoding | Micci-Barreca, SIGKDD Explor. 2001 |
| PMM 插补 | Little, J. Bus. Econ. Stat. 1988 |
| 多重插补 / mice | van Buuren & Groothuis-Oudshoorn, J. Stat. Softw. 2011 |
| missForest | Stekhoven & Bühlmann, Bioinformatics 2012 |
| 缺失数据理论 | Little & Rubin, *Statistical Analysis with Missing Data* |
| 数据挖掘中的泄漏 | Kaufman, Rosset & Perlich, KDD 2012 |
| 效应量 | Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 1988 |
| ROC / AUC | Hanley & McNeil, Radiology 1982 |
| 自助法 | Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 |

竞赛讨论区的贡献写进 Acknowledgements：四项约束、逐取值编码、秩空间 stack
三项想法来自那里，我们做的是在自己的冻结折叠上重新测量。
