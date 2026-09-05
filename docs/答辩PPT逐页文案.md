# 课堂答辩 PPT · 逐页文案（2026-09-05 修订版）

12 分钟 / 16 页正文（P5、P8、P18 已删）/ 正文全英文。

本文件把每一页分成两块，**两者用途不同**：

- **页面上（slide）**：会出现在投影上的全部文字。这里只放**结论、数字、表格、图**；
  理由行用 `→` 开头（house style：事实黑色，理由深红）。
- **讲稿（speaker notes）**：讲者要说的话，**不上页面**。凡是对图/表的解释、
  机制的展开、为什么这样设计，全部进这一块。讲稿是"照着说也完整"的脚本。

数字全部来自 `docs/实验报告.md`（权威）与 `paper/preprocessing-expressiveness.md`
（论文，与答辩口径一致）。两者冲突时以实验报告为准，本文件已对齐。

> **⚠ 本轮修订改了什么（重要，做 PPT 时注意）：**
>
> 1. **P19 的统一论断表整体重做。** 旧版沿用三处旧的"无反证口径"数字
>    （−0.00105 / +0.01065 / +0.00010 / −0.00357）。其中 −0.00105 与 +0.01065
>    是编码前的旧口径，且 +0.01065 在当前网格里没有对应的测量；
>    −0.00357（线性模型上删派生特征）**项目没有可复现的测量**，论文 §6 明确
>    "派生特征这一实例不做跨模型族论断"。新版改用论文 §5–§7 的数字：
>    插补行用 n=15 配对检验（+0.01273 / +0.00617），派生特征行只报树上的
>    +0.00064（四项余量），glmnet 一格写"未测量"。
> 2. **分辨率下限改为一对新的实测值：0.000098（近乎孪生）与 0.000564
>    （跨模型族），相差近六倍**（实验报告 §10.5、论文 §3.4 Table 2）。
>    旧版写的 0.000076 / 0.000680 / 九倍是更早那一次（另一对模型）的实测，
>    口径已被替换。P20 与 P5 的备用数字同步改。
> 3. **P15 "排名偏离都发生在第 6~9 名"是错的**：前五名内有一次 2↔3 的交换
>    （L2_xgboost vs L1_lightgbm），它低于那一对的分辨率下限、判不清；
>    六次真正的排错全部涉及 L3 三格。新版按报告 9.4.4/9.5 的说法改。
> 4. P4 页码注明"四件事"，页面实际只有**三件**，统一为三件事。
> 5. P9 树家族的排序对 ranger 没有 L1，表头写法修正。
> 6. P17 "61% 行编码失效"按论文 §8 的严格说法改为**上界**。
> 7. 图表清单更新：取消 P7 的四级阶梯 SVG（已随该内容删除），只保留 P3 新图。

---

## 第一部分 · 实验基础设置（第 1–6 页）

### P1 — 封面

**页面上（slide）**

```
Preprocessing Beats Modelling
Predicting Smartphone Addiction · Kaggle Playground S6E8

Group 10 · Members A–I
Metric: ROC-AUC
Best single model 0.96784 | 14-member ensemble 0.96807
```

**讲稿**

一句话点题：这个项目最大的收益不来自模型，来自预处理——改数据表示的一步，
比我们做过的任何建模改动都值钱。
接着预告三分结构：先讲并列比较（插补×算法），再讲两个后续问题，最后收束到
统一论断。约 20 秒。

---

### P2 — 任务与指标

**页面上（slide）**

> **The Metric Decides What Counts as a Good Model**

Predict `addicted_label` from 12 daily-use and routine features.
**The classes are not balanced: 70.94% of the training rows are positive.**

**Accuracy counts rows, one per person.**
> Accuracy = (TP + TN) / (TP + TN + FP + FN)

Predict "addicted" for every row → Accuracy = 0.7094, the majority rate itself,
earned by a model that separates nobody.

**AUC counts pairs, one drawn from each class.**
> AUC = P( s(x⁺) > s(x⁻) ) = (1 / n⁺n⁻) · Σᵢ Σⱼ 1[s(x⁺ᵢ) > s(x⁻ⱼ)]

Every term compares one positive with one negative, so the class sizes divide out.

→ Under 70.94% positives, accuracy rewards predicting the majority; AUC cannot be
moved that way, because one class never appears except opposite the other.
→ The competition scores on AUC — and that is why this is the right metric for this data.

**讲稿**

两条公式各念一句就够，重点是"准确率数的是**行**，AUC 数的是**配对**"。
可以顺口带过一句工程细节：n⁺n⁻ ≈ 9.85×10¹⁰ 对，全量算 AUC 时这个乘积会整数
溢出（报告 10.4 记录过）。
"70.94% 的正例率"是本页唯一要听众记住的数字——它决定后面所有方法的走向。

---

### P3 — 缺失机制：MCAR 是检验出来的

**页面上（slide）**

> **The Missingness Is MCAR — We Tested It**

MCAR: whether a value is missing is independent of the data — of the label and of
the feature values.

691,369 training rows · 296,302 test rows · every column is missing 4.18%–19.38%
**61.06% of rows are missing at least one column.**

〔图：新画 SVG，本页主体〕三个检验竖排：每个检验一行，列四段
「question asked → how → result → verdict」：

| | question asked | how | result | verdict |
|---|---|---|---|---|
| **1** | Is missingness related to the **label**? | addiction rate of the missing vs the present group, all 12 columns | largest gap **0.0042** | **no** |
| **2** | Do the other columns' **values** predict a column being missing? | xgboost per column, on rows where the other 11 are complete, 5-fold | AUC **0.5012 – 0.5144** | **no** |
| **3** | Do the other columns' **missing flags** predict it? | same setup, flags only, no values | AUC **0.5674 – 0.7547** | **yes** |

**Missingness is independent of the data values — but the columns' missing indicators
are not independent of one another: it clusters by row.**

→ Test 3 is not a counter-example to MCAR. MCAR asks for independence from the
**data values**; it does not require the columns' missing flags to be independent.
Some rows are simply more complete than others.
→ Tests 1 and 2 are what decide the method: **nothing about the label lives in the
missingness pattern**, so the missing entries can only be recovered from the columns
that are present — hence "what to do with missing values" is one axis of the main
experiment.

**讲稿**

先一句话说清 MCAR 的定义，再走一遍三个检验。
检验 2 是最贴近 MCAR 定义的那一个，AUC 全部贴在 0.50——重点念这一行。
检验 3 的 "yes" 必须立刻接上一句"这并不反驳 MCAR"，否则听众会误以为自相矛盾。
可补的一句（被追问时）：任何只用已观测数据做的检验，都验不了「缺失是否与那个
**没有观测到的值本身**有关」（MNAR 盲区）——这不是我们特有的局限。

图的要求：三个区间（0.0042 / 0.5012–0.5144 / 0.5674–0.7547）的字面精确度是
全图意义所在，用原生 SVG 而不是生成图像；右侧竖轴标注 0.5 的位置，
让 0.50±0.01 和 0.50±0.13 的距离一眼可见。

---

### P4 — EDA 立住的三件事

**页面上（slide）**

> **What the Data Told Us Before Any Model Was Fitted**

版式：三行，每行左侧一句结论 + 一条理由行，右侧一张图。图占版面主体。

---

**1 · One column already separates the classes.**
1-hour bins of daily screen time: addiction rate **24.2% → 99.96%**.
〔图 A〕`reports/figures/fig2_标签单调性.png` 的 (b) 面板（真实 vs 打乱标签）

→ The flat red line is the noise floor: identical binning on shuffled labels spans
only **0.024** against the real **0.771** — the curve is not an artifact of the binning.

---

**2 · Gender, stress level and academic/work impact carry nothing.**
〔图 B〕新图：三列各取值的成瘾率，在固定屏幕时间分段内比较，三条线全平

→ EDA said noise; the ablation agreed: removing all three costs **−0.00003**
(Cohen's *d* −0.30, only 3/5 folds agreeing). Dropped.

---

**3 · One relation holds on every row.**
`screen time ≥ social + gaming + work/study`
〔图 C〕`reports/figures/fig1_硬结构约束.png` 按四项口径重出（全部点落在 y = x 上方）

→ The residual comes all the way down to **0.000**, so this is the true boundary,
not a looser bound implied by it. It becomes the derived feature "budget slack".

**讲稿**

三条各 15 秒。
第 1 条：要不要额外解释"0.771 vs 0.024"——打乱标签后极差塌掉，说明单调性来自
真实关联，不是分箱造的。
第 2 条：只念结论和 −0.00003，不念置换重要性。
第 3 条：要说"残差最小值 0.000"——这是图上看不出来的，也是这条约束值钱的原因。
（可补内幕：我们最初写成了三项，残差最小值停在 0.100，才发现四项才是真正边界。）

⚠ 三张图都要重做：
- 图 A：`fig2` 是中文标注的双面板，取 (b) 面板、标注改英文。
  图注写的是"13 个间隔中有 3 处小幅回落（最大 0.013）"，所以只写 "rises"，
  不要写 "monotonically"。
- 图 B：无现成图，需新出。`R/02_eda.R` 有该口径的计算，加一段绘图即可。
- 图 C：现成 `fig1` 画的是三项版且含中文，需按四项口径（421,427 行）重出。

---

### P5 — 已删（页面不再出现）

原「可比性的三条纪律」整页删除。三条内容的去向：

- **冻结折叠 + 折内拟合** → 并进 P7 实验设计，一行说清。
- **配对统计量的报法**（同折 delta / Cohen's d / 逐折同号）→ 不上页面，
  作为口头约定，第一次出现配对数字（P10）时顺口说明一次。
- **安慰剂列、分辨率下限** → 全部去掉，答辩不讲。被追问时用数字回答：
  同库近乎孪生的一对下限 **0.000098**，跨模型族 **0.000564**（实验报告 §10.5）。

⚠ 页码：本文件保留原编号，P5 留空。定稿后统一重排为 1–16。

---

### P6 — 特征工程

**页面上（slide）**

> **Feature Engineering in Two Steps**

**Step 1 — five derived features**

| feature | formula |
|---|---|
| budget slack | `screen − (social + gaming + work/study)` |
| weekend ratio | `weekend screen ÷ screen` |
| social share | `social ÷ screen` |
| gaming share | `gaming ÷ screen` |
| waking-time share | `screen ÷ (24 − sleep)` |

**Step 2 — per-value target encoding**

Replace each **exact value** of a column by the positive rate of that value, computed
on the training part of the fold:

> enc(v) = ( positives at v + prior × m ) / ( rows at v + m ),    m = 20

→ Every ratio and difference in Step 1 is computed **after** imputation — a missing
component would otherwise make the whole feature missing.
→ Why it fits this data: values sit on the generator's **0.01 lattice**, and the
second decimal correlates with the positive rate — so the exact value carries signal,
and that signal is **not monotone**. A tree split can say "greater than 6.37";
it cannot say "exactly 6.37". A lookup table can.

**讲稿**

两步的分工说清楚：第一步是"我们从数据结构里读出来的关系"，第二步是
"让模型能查表"。
格点这个观察来自竞赛讨论区，我们按纪律把它当线索、在自己的冻结折叠上复现
确认之后才建的编码（实验报告 §8）。这句口头说带一句，是"我们读了讨论区、
但没有照抄"这个加分点的落点。

⚠ 编码的收益**不在本页**，在 P19 的三次验证表里一起讲。
⚠ 用 **+0.00424**（17 特征 base 0.96138 → 0.96562，d 7.97，5/5 同号，
`R/18_new_features.R`）。**不要用 +0.00455**——后者把编码和"改对四项定义"
捆在一起报，是两件事的结果。P20 引用 +0.00455 时是作为"特征工程整步"。

---

## 第二部分 · 并列比较：4 种插补 × 4 种算法（P7–P13）

### P7 — 主实验与结果

**页面上（slide）**

> **Four Ways to Handle Missing Values × Four Algorithms**

**L1** keep the missing marker — the model learns which way to send it.
**L2** fill the column median.   **L3** regress on the other columns, fill the prediction.
**L4** random forest prediction, then draw one **observed** value from the 5 nearest
donors (PMM).

Only xgboost and lightgbm handle missing values natively, so L1 exists for those two
alone — **14 cells, not 16.**

| CV AUC | xgboost | lightgbm | ranger | glmnet |
|---|---|---|---|---|
| **L1** | **0.96784** | 0.96746 | — | — |
| **L2** | 0.96755 | 0.96726 | 0.96324 | 0.94898 |
| **L3** | 0.94770 | 0.94081 | 0.93903 | **0.95490** |
| **L4** | 0.95443 | 0.95410 | 0.95076 | 0.94559 |

Best single model **L1_xgboost 0.96784 ± 0.00053**, 1,264 boosting rounds.

→ All 14 cells run on the same 691,369 rows under one frozen 5-fold split, and every
transform that reads the label — imputer and encoder alike — is fitted inside the
training part of the fold. The 14 scores are directly comparable.

**讲稿**

先让评委看整张表，指出两个跨度的对比：**算法之间差约 0.03，插补策略之间
只差 0.001~0.01**。先选对算法，再谈插补。
再指出 L3 那三格的折间 sd 是其余各格的 4~7 倍（0.0021~0.0034 对 ~0.0005）——
这是个警示信号，P17 会收口解释它。
最后念一遍那条理由行：它挡的是"你们的 target encoding 是不是泄漏了"
这个必问的问题。
被问到 lightgbm 数字时可以说明：lightgbm 不可逐位复现，单次运行 ±4e-5，
低于我们做的每一对的分辨率下限（报告 §2.4）。
原计划"四级阶梯 SVG"已取消，本页不再需要新图。

---

### P8 — 已删（原「主结果」页并入 P7）

---

### P9 — 两张网怎么读

**页面上（slide）**

> **The Two Model Families Disagree**

Trees (xgboost · lightgbm · ranger):  **L1 > L2 > L4 > L3**  (ranger: no L1)
The linear model (glmnet):  **L3 > L2 > L4**

**1 · On the trees, doing nothing wins.** A boosted tree already learns a default
direction for a missing value at every split. Imputation turns "unknown" into a
definite number — and the tree then has to trust that number as evidence.

**2 · The linear model wants the opposite.** glmnet has no way to represent "unknown"
— every row must carry a number. L3's conditional-mean fill preserves the correlation
structure a linear model depends on; L4's random draw is noise around that same mean.

**3 · Open: why is L4 not better?** L4 fills in **real observed values** and is the
more faithful imputer, yet on every tree it sits below the plain median fill.

→ Points 1 and 2 are the same preprocessing step carrying **opposite signs on two
model families** — the first of three independent appearances of that pattern.
→ Our hypothesis for point 3 (unmeasured): what matters to a tree is not how
accurately a cell is filled, but whether the filled cells can still be told apart.

**讲稿**

三点各 15 秒。
第 3 点必须说成开放问题，不要说成结论。旁证只作口头：L1 与 L2 只差 0.0003
（标记显式 vs 隐式保留），到 L4 掉 0.013（标记消失）；且 L2 的中位数在编码表里
是单一取值，所有被填的行共享它，本身就是一个缺失标志。
这页是 P19 论断的第一块砖，说清楚"符号相反"四个字。

⚠ 项目**没有测过**第 3 点的假设。能证伪它的实验写在报告 §9 之后：在 20 万那一级
重跑 `L4_xgboost` 并打开 12 个缺失指示（`WITH_NA_INDICATORS <- TRUE;
FORCE_REBUILD <- TRUE`）。若 0.95221 明显向 L2 的 0.96516 靠拢则假设成立。
插补有缓存，代价分钟级。**跑了再决定第 3 点怎么写。** 不构成反驳的那条：
「缺失指示无用」的消融（+0.00003，3/5）跑在 L1 线上，那条线标记本来就原生。

---

### P10 — 稳健性

**页面上（slide）**

> **The Same Conclusion at n = 15**

The comparisons we wanted to draw conclusions from were re-run on **three further
fold splits** — 5 folds × 3 splits = **15 paired observations** instead of 5. Both
configurations always see the same rows, so the difference is measured on identical
data:

> dᵢ = AUC_A(fold i) − AUC_B(fold i),   mean Δ = (1/n) Σ dᵢ,   Cohen's *d* = mean(d) / sd(d)

| Comparison | mean Δ | Cohen's *d* | folds agreeing | p |
|---|---|---|---|---|
| L1 vs L2 (xgboost) | +0.00039 | 3.42 | **15/15** | < 0.00001 |
| L2 vs L3 (xgboost) | +0.01273 | 7.74 | **15/15** | < 0.00001 |
| L3 vs L2 (**glmnet**, opposite direction) | +0.00617 | 21.46 | **15/15** | < 0.00001 |

→ mean Δ says how big the gap is; Cohen's *d* says how big it is compared with its
own scatter — unitless, and therefore comparable across pairs a raw AUC difference
cannot be.
→ Row 3 is page 9's sign reversal measured directly: the same pair of imputation
lines, opposite winners on the two model families, both sides unanimous across
15 folds.

**讲稿**

先说清这一页在做什么——**同样的对比，换三套折重跑，把配对样本从 5 提到 15**。
重复出来的结果只用于统计检验，不进主结果表；主表一律用那套冻结折叠。
"同号"那一列是最省事的判据，三行都是 15/15。
n=5 时配对 t 检验不稳健，所以我们以 Cohen's d 和同号数为主判据，p 值只作参考——
这句话解释了为什么表里 p 排在最后一列。
不要念表，只说"我们把最重要的两个结论各测了 15 次，都是全票"。

⚠ 公式口径：这里的 Cohen's *d* 是**配对形式**（d_z），分母是差值自身的标准差，
不是两组的合并标准差（`R/09_ablation.R:171`：`coh <- mean(d)/sd(d)`）。
被问到时要说得出这一点。

---

### P11 — 特征研究

**页面上（slide）**

> **How Important Is a Feature? It Depends on What Else the Model Has.**

We removed `notifications_per_day` and re-ran the whole pipeline — twice, at two
moments in the project:

| | cost of removing it |
|---|---|
| before the per-value encodings | **−0.00825** |
| after the per-value encodings | **−0.00022** |

**Thirty-seven times smaller — and nothing about the column itself changed.**
`te_notifications_per_day` now carries the same information by a second route, so
deleting the raw column stopped meaning *removing the information*.

Permutation importance over all 25 features: **the top five are all encoded columns**,
the first is ten times the sixth (the highest-ranking raw column).

〔图〕`reports/figures/fig8_特征重要性.png`（需重出英文版）

→ Ablation asks whether a column is **indispensable**; permutation asks whether the
fitted model **used** it. The two answers come apart exactly when a second column
carries the same information — which is what the encodings created.
→ An ablation table is only meaningful together with the feature set it was measured
on. Quoted on its own it will be read as a property of the column, which it is not.

**讲稿**

重点是这个 37 倍，不是那两个绝对数字。
一句话概括：**「这一列重不重要」不是这一列的性质，是这一列和其余特征之间
关系的性质。** 它和 P19 是同一家——那里讲模型能不能自己造出来，这里讲信息
有没有第二条通路。
置换重要性不念，只念"前五全是编码列"。

---

### P12 — 超参数与低成本增益

**页面上（slide）**

> **What Tuning Actually Bought**

**Number of rounds — the one change that paid.** Early runs used fixed 600 rounds,
a number with nothing behind it. Letting early stopping choose (near 1,158):

| fixed 600 rounds | 0.95910 |
|---|---|
| **early stopping** | **0.96088** — **+0.00178** |

**Grid search — 18 configurations in two stages** (`R/10_tune.R`):
learning rate ∈ {0.03, 0.05, 0.1} × depth ∈ {4, 6, 8}, then `min_child_weight` × `colsample`.

| default: learning rate 0.05, depth 6 | 0.96529 |
|---|---|
| **best: learning rate 0.03, depth 4** | **0.96545** — **+0.00016** |

Stage two changed nothing. **Depth 4 > 6 > 8 at all three learning rates.**

→ Shallow wins because the label is close to a monotone function of a few variables
(page 4) — the search confirmed a property already visible in the data.
→ **Tuning was worth four times more before the encodings**: +0.00067 then, +0.00016
now. The encoding hands the model directly what it previously approximated with more
rounds and finer splits, so the margin left for tuning shrinks.
→ Re-running all four imputation lines with the tuned parameters leaves
**L1 > L2 > L3 unchanged** — the grid's conclusions do not depend on the
hyper-parameters.

**讲稿**

顺序很重要——**先说早停 +0.00178，再说搜参 +0.00016**。听众会自己得出
"调参没用"的结论，那正是我们要引导的：真正的收益在别处。
然后立刻给出那个"四倍"——调参不是天生没用，是**被编码吸收了**。这是本页真正
的发现，和 P11、P19 同一个家族。
最后一条稳健性是防守用的：被问"结论会不会只是没调好参"，答案是换了最优参数
排序不变。

⚠ 稳健性只报到 L1 > L2 > L3。同一次重跑里 L3 与 L4 相差 0.0000054，远低于任何
分辨率下限（最接近的一对也有 0.000098）——**这两个的先后不是排序，不能报**。
⚠ 「已定价但不设默认」三项（种子平均 +0.00017 / 3× 时间，学习率 0.05→0.02
+0.00013 / 2.2× 时间）放 P20 的"如果重来"里口头说，不上本页。

---

### P13 — 集成

**页面上（slide）**

> **Ensembling in Rank Space, +0.00023**

| Combination method | CV AUC | vs best single |
|---|---|---|
| **Rank-space logistic regression** | **0.96807** | **+0.00023** |
| Hill climbing | 0.96802 | +0.00018 |
| Logistic regression in probability space | 0.96720 | −0.00064 |
| Rank averaging | 0.96485 | −0.00299 |

14 members · rank correlation between members: min 0.819, median 0.929, max 0.997.

**Why rank space.** AUC reads only the ordering — members don't share a scale, so
averaging probabilities lets the wide member dominate. Ranking each member first
removes the scale and discards nothing AUC uses.

→ Rank space beats probability space by **0.00087**, the largest gap among the four
methods — a zero-cost change, applied to predictions we already had.
→ Rank averaging falls **below** the best single model: it treats all members alike,
and 8 of the 14 score under 0.9550.
→ The ceiling here is not the combination rule — one candidate dominates and most
members are near-duplicates. **Member homogeneity is the root cause**; page 19 says
what to select for instead.

**讲稿**

集成只涨 0.0002，坦率说出来，并把它接到 P19 的落点："为什么集成躺平——
成员长得太像。"
尺度的机制口头讲一遍（图上不留细节）：AUC 只读排序，而我们的成员没校准到
同一尺度——glmnet 的输出铺得很开，梯度提升树挤在很窄的区间。在概率空间
加权等于把各自的校准也一起加权了，摊得开的成员会占主导，不是因为它排得好，
是它的数字动得更大；换成秩以后这一层被中和掉。
最后一句指向 P19，下一个板块由此进去。

---

## 第三部分 · 后续实验（P14–P19）

### P14 — 两个后续问题

**页面上（slide）**

> **Two Questions the Grid Could Not Answer**

The main grid ran on the full data, but virtually every other comparison — ablation,
tuning, feature pricing — ran on a **200,000-row subsample**, and the full data was
used only for delivery. Two things were assumed rather than tested:

**Q1. Does a choice made on the subsample survive on the full data?**

**Q2. Do two preprocessing steps interact?**
Imputation and per-value encoding were designed and measured separately. Nothing
guaranteed they compose.

**讲稿**

这一页把叙事从"并列"切到"递进"，语气要明显换挡。这 20 秒值钱，念完两个问号
停一下。

---

### P15 — Q1 设计与结果

**页面上（slide）**

> **Selection Transfers Completely. Ranking Does Not.**

A nested ladder — 50k ⊂ 100k ⊂ 200k ⊂ 400k ⊂ 691,369 — with the same 10 candidates,
the same 25 features and the same folds at every level.

| rows | Spearman ρ | winner | **selection regret** |
|---|---|---|---|
| 50,000 | +0.903 | L1_xgboost ✓ | **0.00000** |
| 100,000 | +0.927 | L1_xgboost ✓ | **0.00000** |
| **200,000** ← what we used | **+0.964** | L1_xgboost ✓ | **0.00000** |
| 400,000 | +0.988 | L1_xgboost ✓ | **0.00000** |

The top-5 **set** is identical to the full-data set at every level.

→ **Selection regret** is what screening on this sample costs against picking on the
full data — and it is exactly zero everywhere: 50k (7.2% of the data) already finds
the full-data winner.
→ The ranking does not fully transfer, and the deviations are not noise:
**all six real misrankings involve an L3 cell**; the only top-5 inversion (ranks 2–3)
sits below that pair's own resolution floor. The next page says why.

〔图〕`paper/figures/fig3_size_ladder.png`（已英文版）

**讲稿（全部不上页面）**

**为什么嵌套。** 五级是包含关系，不是独立抽样。独立抽的话，"排名变了"分不清是
样本量的作用还是换了一批行的作用；嵌套之后样本量是唯一变化的量。
**每个池里的行保留原来的折号**，所以任意两级的验证集互相包含，AUC 可直接比。
这两点是这实验能成立的关键。

**为什么用「遗憾」这个指标。** ρ 和命中率只说排名对不对，真正要问的是
"用小样本筛选要付多少代价"。哪怕排名乱了，只要遗憾是 0.0001，就没有代价。

**为什么 10 个候选不是 14。** L4 的链式随机森林在全量上一格 234 分钟，
十几小时的算力换不来它对"排名能不能迁移"的额外答案。所以本页结论适用范围是
这 10 个候选；把 L4 放进去会多出一处迁移失败，机制与已有的相同。

**20 万那一级就是冻结的 `subsample_200k`**，不另抽——它同时充当"主网格在
新特征口径下的重跑"，一份算力两个用途。

---

### P17 — Q2：两个预处理互相破坏

**页面上（slide）**

> **Imputation Can Destroy the Encoding**

The encoding is keyed on **exact values**, and the data sits on the generator's
**0.01 lattice**.

Lookup hit rate for the cells an imputer filled in:

| line | what it fills in | hit rate |
|---|---|---|
| L2 median | an observed median — on the lattice | **100.00%** |
| **L3 regression** | an arbitrary real number — **off** the lattice | **0.045%** |
| L4 PMM | a value copied from a real donor — on the lattice | **99.999%** |

At least one column is missing in 61% of rows — an **upper bound** on how many rows
this failure can touch, since the eight encoded columns are a subset of the twelve
features.

**Verification — one variable changed at a time** (xgboost):

| condition | L3 | L4 | winner |
|---|---|---|---|
| 200k, **no** encoding | 0.95951 | 0.95791 | L3 by 0.00160 |
| 200k, **with** encoding | 0.95297 | 0.95221 | L3 by 0.00076 |
| **full**, with encoding | 0.94770 | **0.95443** | **L4 by 0.00673** |

> **With per-value encoding in the pipeline, an imputer must fill values back onto
> the lattice.**

→ This is orthogonal to which imputer is statistically more accurate. L3 estimates
better, and produces something the next step cannot use.
→ The encoding **halves** L3's lead; the reversal needs the full data as well.
Reporting only the last row would credit the encoding with a reversal that also
required the sample size.

〔图〕`paper/figures/fig2_lattice_mechanism.png`（已英文版）

**讲稿（全部不上页面）**

**这是全场最新的一页，慢讲。** 三个命中率念出来——100%、0.045%、99.999%。

**L4 那一行是事先写下的预测，然后才去测的。** PMM 抄的是真实观测值，所以它
必须留在格点上。先写预测再测量，这是这条机制可信的原因。

**为什么反转需要全量。** 数据越多，编码表对格点上的取值越准，"查得到"与
"查不到"两类行的差距就越大，所以 L3 的损失随样本量放大。样本量阶梯直接证实：
L3 三格是全表**唯一** AUC 随样本量下降的行（L3_xgboost 从 20 万的 0.95297 掉到
全量的 0.94770，同期 L4_xgboost 从 0.95221 升到 0.95443，两条线相向而行）。
这也是 P15 那批排名偏离集中在 L3 的原因。

**一个推论：小样本会低估这种破坏的程度。**

**L3 那三格的折间 sd 是其余各格的 4~7 倍**（0.0021~0.0034 对 ~0.0005），正是
"编码对哪些行失效"随折变化的直接表现——P7 埋的那个警示信号在这里收口。

---

### P18 — 已删（原「Q2 的验证」页并入 P17）

---

### P19 — 统一论断

**页面上（slide）**

> **One Sentence Covers All Three**

| preprocessing step | can the model build it itself? | gradient-boosted trees | linear model (glmnet) |
|---|---|---|---|
| Imputation (L2 vs L3) | GBDT represents "unknown" natively; glmnet cannot | L2 beats L3: **+0.01273** (n = 15, 15/15) | L3 beats L2: **+0.00617** (n = 15, 15/15) |
| Derived features | GBDT approximates a two-column ratio with staircase splits | ratios absorbed; the **four-term slack** survives **+0.00064** (5/5) | — (not measured) |
| Per-value lookup (TE) | GBDT approximates with thousands of splits; glmnet would need one parameter per value | **+0.00424** | **+0.03352** (≈ 8×) |

> **The value of a preprocessing step equals how hard the downstream model finds it
> to construct on its own.**

→ What the model can already build, we only duplicate as a redundant column. What it
cannot build is the only place new information can enter.
→ "Can it build it" is a property of the model's **expressiveness**, not of the
data — exactly why one step can carry opposite signs, or an eightfold difference in
value, across two model families.
→ A linear model *can* do per-value lookup — one-hot on exact values reaches 0.95929
— but only by spending one parameter per value. What it cannot do is the lookup
**without** paying for it.
→ And this is the answer to page 13: to make an ensemble worth something, select
members whose **preprocessing needs differ**, not members whose scores are high.

**讲稿**

全场落点。前面三部分在这里合流。
- 第一行：P10 的符号相反，直接引 n=15 的配对数字（与论文 §5 同源，Tier A 20 万、
  25 特征口径）。主网格（P7）上同向且更大——树那格 L2−L3 ≈ 0.020——但那同时
  包含着 P17 的格点破坏；两种口径数字方向一致，不必解释成矛盾。
- 第二行：坦白这是最弱的一个实例——没有跨模型族测量，只报"四项余量在编码之上
  仍有效，比值被吸收"。这正是论文 §6 直接声明的弱处，先自己说。
- 第三行：8 倍差，念出 0.00424 和 0.03352。
念出加框那句话之后停一拍。最后一句接 P13 的集成落点。

---

### P20 — 结论

**页面上（slide）**

> **Conclusions**

| | CV AUC |
|---|---|
| Best single model, L1_xgboost | **0.96784 ± 0.00053** |
| 14-member ensemble, rank-space logistic | **0.96807** |
| Leaderboard during the competition | 0.96627 |

**Three things worth carrying to the next problem**

1. The value of a preprocessing step equals how hard the downstream model finds it to
   construct itself — verified three times here, always in the same direction.
2. Whether a hand-built feature helps depends on **how many terms its shape needs** —
   a tree approximates a two-column ratio; it cannot approximate a four-term
   hyperplane at any depth.
3. "How large is significant" is a property of the pair being compared, not of the
   data — **0.000098** for near-twins, **0.000564** across model families (≈ 6×, same
   data).

**Limits.** 5-fold throughout, no 10-fold control. The two post-competition numbers
have no leaderboard validation. The size ladder covers 10 of the 14 candidates.
Transfer was verified for model selection and ensemble weights, not for
hyper-parameter selection.

→ Given the ordering again, we would do feature engineering **before** the imputation
study: the whole imputation study spans 0.00297; one feature-engineering step (the
encoding + the corrected definition) is worth 0.00455.

**讲稿**

留 10 秒给"边界"那一段——主动说没做什么，比被问出来好。
"如果重来"那条是收尾：先在特征工程上花算力，再研究插补；并口头补三个工程
取舍（种子平均 +0.00017 / 3× 时间、学习率 0.05→0.02 +0.00013 / 2.2× 时间、
0.02→0.01 再 +0.00003 / 再 2×）——已定价、不设默认。

---

## 需要新画的图（1 张）

**P3 的 MCAR 检验链**：三个检验竖排，每行「问什么 → 怎么测 → 结果 → 结论」
四段。检验 1、2 判 no（支持 MCAR），检验 3 判 yes 但用不同颜色标出
「这一条不反驳 MCAR」。右侧一条竖轴标出 AUC 0.5 的位置，让 0.5012–0.5144 和
0.5674–0.7547 的距离一眼可见。原生 SVG——三个数字区间的字面精确度是这张图的
全部意义，不能交给生成图像。

## 需要重做/复用的图（4 张）

| 页 | 文件 | 处理 |
|---|---|---|
| P4-A | `reports/figures/fig2_标签单调性.png` | 取 (b) 面板、英文标注 |
| P4-B | 新出（`R/02_eda.R` 加一段） | 三噪声列在固定屏幕分段内的成瘾率 |
| P4-C | `reports/figures/fig1_硬结构约束.png` | 按**四项**口径重出（421,427 行）、英文标注 |
| P11 | `reports/figures/fig8_特征重要性.png` | 英文标注 |
| P15 | `paper/figures/fig3_size_ladder.png` | 已英文版，直接用 |
| P17 | `paper/figures/fig2_lattice_mechanism.png` | 已英文版，直接用 |

## 数字口径备忘（做 PPT 时对照）

- 主网格（P7）与 P17 三条件表：**全量**，25 特征，冻结折叠。
- n=15 配对表（P10、P19 第一行）：**Tier A 20 万**，三套折划分，只用于统计检验。
- TE 收益（P19 第三行）：Tier A，17 特征 base → 25 特征，+0.00424/5 折同号。
- 编码的"整步收益"（P20 末条）：+0.00455 含"改对四项定义"，与 +0.00424 分开说。
- 分辨率下限：近乎孪生 0.000098 / 跨模型族 0.000564（实验报告 §10.5；
  ρ 低于约 0.95 时公式低估约 12%，用脚本输出的实测值）。
- lightgbm 数字均为单次运行，±4e-5；被问时说明，不解释为结论差异。