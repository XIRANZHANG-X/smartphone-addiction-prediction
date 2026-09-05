# Preprocessing-Expressiveness Paper — Revision Round 1 (Peer Review Response)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 `build/output/review_preprocessing-expressiveness_20260904.md`（5 位审稿人 + 跨审稿
意见汇总 + 13 条修订任务表）逐条落实修订。用户已确认：范围 = 全部 13 条；T1（中心论断未操作化）
的处理方向 = 降级措辞，不做新的「重构难度」代理指标实验。

**Architecture:** 沿用原计划已确立的纪律——**新分析先进 `docs/实验报告.md`，再进论文**；论文里
每个数字必须能追溯到某个脚本的产物。5 条修订需要新分析（T2/T6/T7/T10，以及 T12 的引用决定），
其余是文本/格式修订。分组为 10 个任务：先做需要新证据的（它们可能改变后面文本任务要写什么），
再做纯文本任务，最后做扫尾。

**Tech Stack:** 同原计划——R 4.6.1（`D:/R/R-4.6.1/bin/Rscript.exe`）、data.table、ggplot2。

**Base branch:** 继续在 `paper-preprocessing-expressiveness` 分支上工作（HEAD `6ffac24`，尚未
合并 main）——这是同一篇论文的修订，不开新分支。

## Global Constraints

- R 不在 PATH，一律用 `"D:/R/R-4.6.1/bin/Rscript.exe"` 调用。
- 工作目录必须是 `C:/Users/Lenovo/Desktop/project`；每条 Bash 命令自带 `cd`。
- **不得引入 `docs/实验报告.md` 中没有的数字。** 新测的数字必须先写进报告再进论文——这条纪律
  在这一轮修订里同样适用于 T2/T6/T7/T10 产出的每一个新数字。
- **不得编造引用。** 仍是 spec §10 白名单那 14 条；白名单外一律 `[CITATION NEEDED]`。
- 论文语言为英文；作者占位符 `[Author Name]` / `[Student ID]` / `[Affiliation]` 不代填。
- 写 R 脚本用 Write 工具，不用 bash heredoc。
- 每个任务结束时提交一次，commit message 结尾加：
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  ```
- **已核实的背景事实**（写任务简报时直接引用，不必重新查）：
  - 已装包版本：xgboost 3.2.1.1、lightgbm 4.7.0、ranger 0.18.0、glmnet 5.0、
    data.table 1.18.4、missRanger 2.6.1（L4/PMM 用）。R 4.6.1。
  - `R/lib_models.R` 与全部 `R/06_model_*.R` 里**没有任何显式的模型级随机种子**——
    只有 `output/folds.rds` 的折种子 20260821。T8 审稿意见成立，不是审稿人误判。
  - `docs/实验报告.md:102` 的「24.2% → 99.96%」与 `:104` 的「极差从 0.771 降到 0.024」
    很可能是两个不同的量：前者是「首箱到末箱」，后者是「全部分箱里的真实极差（最大−最小）」，
    数据有很多 1 小时分箱，极值不一定落在首末两箱。T13 要求的是**说清楚这是两个不同口径**，
    不是必然要改数字——除非任务 10 实际去查脚本发现其中一个是笔误。

---

### Task 1: MCAR 再验证 + 逐列缺失率表 + 训练/测试分布一致性引用

对应 T2（主要，统计审稿人+数据质量审稿人共同提出，Cc.2）、T12（数据质量审稿人）。

当前 §2.2「缺失完全随机（MCAR）」只验证了「缺失组 vs 非缺失组成瘾率最大差 0.0042」这一个
必要条件，不构成完整的 MCAR 判定。§8 的「插补对被填走的行是中性的」这条解读依赖这个判定，
所以要补强，不能只是措辞降级。

**Files:**
- Create: `R/32_mcar_check.R`
- Create (产物): `output/mcar_check.rds`
- Modify: `docs/实验报告.md`（§2.2 补充联合分布检验结果 + 12 列缺失率表；补充训练/测试
  分布一致性发现的交叉引用，指向已有的 `R/19_train_test_shift.R` / 1.4 节发现 12——
  **不要重新做一遍这个分析，引用已有结果即可**，这是用户已确认的处理方式）
- Modify: `paper/preprocessing-expressiveness.md`（§2.2）

**Interfaces:**
- Produces: `output/mcar_check.rds`，内容至少包含：12 列各自的缺失率、完整行与不完整行的
  联合特征分布比较结果（如卡方或多元检验的统计量与 p 值；具体检验方法由实现者判断，只要能
  回答「缺失机制是否独立于全部特征取值」这个问题，不只是独立于结局），以及原有的逐列
  成瘾率最大差 0.0042 复算（作为一致性检查）。

- [ ] **Step 1: 写脚本，跑，把结果先写进 `docs/实验报告.md` §2.2**

脚本要做两件原来没做的事：(a) 12 列缺失率表；(b) 完整行 vs 不完整行的联合特征分布比较——
不是逐列单变量检验，是要能回答「缺失机制是否真的与全部特征值独立」。可以用 Little's MCAR
检验（如果有现成 R 包，如 `naniar::mcar_test()` 或 `BaylorEdPsych`），也可以用「预测缺失
指示变量」的方式（拿全部其他列去预测某列是否缺失，AUC 接近 0.5 支持 MCAR，明显偏离则不是）——
实现者判断哪种更适合当前数据结构与已装包，写清楚用的是哪种方法、为什么、以及它检验的是
「缺失与结局无关」还是更强的「缺失与全部协变量独立」，避免重犯审稿人指出的口径混淆。

跑完后，把新数字（缺失率表、联合分布检验的统计量/p 值/结论）写进
`docs/实验报告.md` §2.2，并明确区分「缺失与结局不相关」（已有的 0.0042 那条）和
「MCAR（缺失与全部特征独立）」（这次新测的）两个不同的表述。

- [ ] **Step 2: 论文 §2.2 同步修订**

把新证据写进论文 §2.2：12 列缺失率表（或至少给出范围，别再只用单一的 61.06%）、
联合分布检验结果、以及区分「缺失与结局无关」vs「MCAR」两种表述。同时处理 T12：
在 §2（或 §8 讨论处，实现者判断哪里更自然）加一句，引用已有的训练/测试分布一致性发现
（差异几乎全部来自缺失模式），不重新做分析，只是把已有结论纳入论文论证——这一条支撑
论文「数据完成度是核心」的框架，审稿人认为值得提一句。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/32_mcar_check.R docs/实验报告.md paper/preprocessing-expressiveness.md && git commit -m "revision: MCAR 补强验证 + 逐列缺失率 + 引用训练测试分布一致性 (T2, T12)"
```

---

### Task 2: 分辨率下限定义统一 + 低 ρ 失准披露

对应 T5（统计审稿人）。

`§3.4` 给的公式算的是 `SD(差值)`，但表 2 里 `0.000098` 与 `0.000564` 其实是这个 SD 值乘以
约 1.96（95% CI 半宽）。这条修订会改变「分辨率下限」这个词接下来在 §9/Table 11 里的确切含义，
**必须在 Task 4（T7）之前做**。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（§3.4，Table 2）

**Interfaces:**
- Produces: 一个明确的「分辨率下限」定义（SD 还是 1.96×SD，二选一，选定后全文统一），
  后续任务（尤其 Task 4）必须使用这个统一定义。

- [ ] **Step 1: 核实 `R/23_resolution.R` 实际算的是哪个**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && grep -n "1.96\|qnorm\|resolution\|floor" R/23_resolution.R | head -20
```
确认表 2 的 `0.000098` / `0.000564` 究竟是 `SD(差值)` 本身还是 `1.96 × SD(差值)`。

- [ ] **Step 2: 改写 §3.4**

把公式的输出说清楚是哪一个量；如果表 2 数字确实是 1.96×SD（95% CI 半宽），就把「分辨率
下限」正式定义为这个 95% 区间半宽，公式那行加上 `× 1.96`。同时加一句披露：这个公式是差值
方差的一阶近似，ρ 越低失准越明显（对表 2 里 ρ=0.745 的跨模型族那一对尤其如此），建议对低 ρ
的配对改用 bootstrap 实测 SD——不需要真的重新跑 bootstrap（`R/23_resolution.R` 如果已经是
bootstrap 实现就不用改方法，只需要在文中说清楚这本来就是 bootstrap 实测值，不是解析公式的
一阶近似；如果确实是解析公式，才需要加失准披露）。哪种情况取决于 Step 1 的核实结果。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: §3.4 分辨率下限定义与表 2 统一，披露低 rho 失准 (T5)"
```

---

### Task 3: 「5/5 同号 + d>2」判据的零分布刻画

对应 T6（统计审稿人）。

现在的安慰剂列只是一次随机实现（+0.00003 / d 0.11 / 2/5），不足以刻画这个判据本身的
假阳性率。这是全文用来「宣布一个效应是真的」的核心判据，值得补一次多重抽样。

**Files:**
- Create: `R/33_criterion_null_dist.R`
- Create (产物): `output/criterion_null_dist.rds`
- Modify: `docs/实验报告.md`（§2.3 或安慰剂列所在小节，补充多次抽样结果）
- Modify: `paper/preprocessing-expressiveness.md`（§3.3）

**Interfaces:**
- Produces: `output/criterion_null_dist.rds`，多次（建议 ≥30 次，实现者可视算力调整，
  但至少要能给出一个有意义的假阳性率估计而不是单点）安慰剂列抽样（或对折标签做置换）
  下，「5/5 同号 + Cohen's d > 2」这条判据被触发的比例。

- [ ] **Step 1: 写脚本，跑**

在冻结折叠上，重复生成多个随机噪声列（缺失模式与某个真实列匹配，做法同现有安慰剂列），
每次都算「5/5 同号 + d>2」是否成立，统计触发比例——这就是这条判据在零假设下的假阳性率。
固定一个可复现的种子序列（不是留一个未固定的随机安慰剂）。

- [ ] **Step 2: 把结果写进 `docs/实验报告.md`，再改论文 §3.3**

论文 §3.3 补一句：多次抽样下，「5/5+d>2」判据的假阳性率是多少（新数字，先进报告再进论文）。
如果假阳性率本身很低（预期如此，因为标准很严），这是对判据的一次正面校准，不是削弱论文。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add R/33_criterion_null_dist.R docs/实验报告.md paper/preprocessing-expressiveness.md && git commit -m "revision: 5/5+d>2 判据的零分布与假阳性率刻画 (T6)"
```

---

### Task 4: 表 11「平局」判决的等价性论证

对应 T7（统计审稿人）。**依赖 Task 2**——必须先有统一的分辨率下限定义才能做这个论证。

`L2_xgboost vs L1_lightgbm` 那一对现在写的是「Tie——差值低于该对的下限」，但「不能拒绝差异」
不等于「没有差异」，需要正式的等价性论证支持这个措辞。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（§9，Table 11 及其判决措辞）
- 可能需要重跑或补充 `R/28_ladder_pairs.R` 的输出（若现有产物已包含 bootstrap 区间则不用重跑）

**Interfaces:**
- Consumes: `output/ladder_pairs.rds`（先检查是否已含这一对在全量数据上的 bootstrap 置信区间；
  没有的话才需要补测）

- [ ] **Step 1: 检查现有产物是否已支持等价性论证**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" -e 'x<-readRDS("output/ladder_pairs.rds"); str(x)'
```
看这个对象是否已经有 bootstrap 区间或类似的量。如果有，直接用；如果没有，对
`L2_xgboost vs L1_lightgbm` 这一对在全量数据上补一次 bootstrap（复用 `R/23_resolution.R`
或 `R/28_ladder_pairs.R` 里已有的 bootstrap 机制，不要重新发明）。

- [ ] **Step 2: 改写 Table 11 的判决措辞**

给出等价界（equivalence margin，用 Task 2 里统一好的分辨率下限定义作为界）内的覆盖率，或
全量数据上该对差值的 bootstrap 区间整体落在等价界以内这一事实，作为「平局」判决的正式支持；
把「Tie」改写为有等价性论证支撑的表述（如「在 [等价界] 以内不可分辨」），而不是简单地说
「低于下限所以是平局」。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: 表 11 平局判决补等价性论证 (T7)"
```

---

### Task 5: 第二位小数信号展示 + 命中率逐列范围

对应 T10（可复现性审稿人+数据质量审稿人共同提出，Cc.3）。

§2.3「第二位小数与正例率相关」这个承重声称目前只有断言，没有展示；表 8 的命中率只报汇总值，
掩盖了列间差异。

**Files:**
- Modify or Create: 追加到 `R/29_paper_figures.R`（若要出新图）或新建 `R/34_decimal_signal.R`
  （若只是产出数字/表格，不需要新图——实现者判断）
- Modify: `docs/实验报告.md`（补第二位小数信号的效应量 + 逐列命中率范围）
- Modify: `paper/preprocessing-expressiveness.md`（§2.3、§8 Table 8）

**Interfaces:**
- Consumes: `output/lattice_hit.rds`（已有，含逐列逐折命中率，Task 1 原计划里 `R/30_lattice_hit.R`
  的产物——`col` 与 `fold` 两列已经支持按列/按折汇总，不需要重新测）

- [ ] **Step 1: 第二位小数信号的效应量**

按第二位小数（或精确取值）分组，算成瘾率的差或相关强度（可以是简单的按第二位小数分组的
成瘾率极差，类似 §2.2 已经在用的分箱极差写法，保持口径一致）。写一个小脚本或在
`R/17_discussion_checks.R` 已有产物里找是否已经算过（检查 `output/` 下是否已有相关产物再决定
是否新写）。

- [ ] **Step 2: 表 8 补逐列命中率范围**

`output/lattice_hit.rds` 已经是逐列逐折的（Task 1 原计划里定义的列：`line`、`col`、`fold`、
`n_imputed`、`n_hit`、`hit_rate`），直接按 `col` 汇总出每条插补线的逐列命中率范围
（最小值到最大值），不需要重新测——这是现成产物的一次新汇总。

- [ ] **Step 3: 数字先进报告，再改论文**

把新汇总数字写进 `docs/实验报告.md`，再改论文 §2.3（第二位小数信号的效应量）与
§8 Table 8（逐列命中率范围，或至少给出最小值-最大值区间，不只是汇总值）。若判断值得配一张
小图/小表展示分组后的成瘾率曲线，可以加，但不强制——文字给出效应量数字也满足审稿要求。

- [ ] **Step 4: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add docs/实验报告.md paper/preprocessing-expressiveness.md && git commit -m "revision: 第二位小数信号效应量 + 命中率逐列范围 (T10)"
```

---

### Task 6: 代码与数据可用性声明 + 模型种子核查

对应 T8（可复现性审稿人，唯一的硬性阻塞项之一）。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（新增小节，建议放在 §3.5 之后，作为
  §3.6 或独立的 "Code and Data Availability" 段——实现者判断哪个位置读起来更自然）

**Interfaces:**
- 无新产物；但内容必须真实核实，不能照抄审稿人的措辞就当作已完成。

- [ ] **Step 1: 核实以下事实，不要假设**

```bash
cd /c/Users/Lenovo/Desktop/project && grep -rn "set.seed\|seed\s*=" R/06_model_*.R R/lib_models.R R/07_ensemble.R
```
已知结论（本次会话已核实，可直接引用）：xgboost 3.2.1.1、lightgbm 4.7.0、ranger 0.18.0、
glmnet 5.0、data.table 1.18.4、missRanger 2.6.1、R 4.6.1；**模型级随机种子目前没有在任何
脚本里显式设置**，只有折种子 20260821（`output/folds.rds`）。

- [ ] **Step 2: 写可用性声明**

内容必须包含：(a) 仓库信息——**不要编造一个 GitHub URL**，如果仓库确实没有公开地址，写
「代码见课程提交物」或类似说法，或留 `[REPO URL]` 占位符，和作者占位符处理方式一致，不要
代填一个不存在的地址；(b) R 版本与上面核实过的关键包版本；(c) 折种子 20260821；(d) **如实
说明模型级随机种子目前未固定**——不要为了满足审稿意见就声称种子已固定，这会是编造。如实
披露「lightgbm 因未设 `deterministic=TRUE` 而不可逐位复现」这条 §3.5 已经写了，这里可以
交叉引用，不用重复整段。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: 新增代码/数据可用性声明，如实披露模型种子未固定 (T8)"
```

---

### Task 7: 方法定义补全——L1-L4、target encoding 平滑与编码列清单

对应 T9（可复现性审稿人）。

L1-L4 插补线、TE 的 `m=20` 平滑、8 个被编码列的清单目前只在脚本里，§7/§8 的表格数字无法
从正文独立重建。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（建议在 §3 加一个新小节给出这些定义，
  §4/§7 分别只需一句回指，不需要重复整段——实现者判断具体放在 §3 还是 §4 更自然，
  但必须在 §4 第一次用到 L1-L4 之前给出定义）

**Interfaces:**
- 无新产物，纯 Methods 内容补全，数字必须与 `R/03_features.R`、`R/05_impute_L*.R` 的
  实际实现一致（不是凭记忆写，要去读脚本核实）。

- [ ] **Step 1: 从脚本里核实定义**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && sed -n '1,40p' R/05_impute_L3.R && grep -n "TE_COLS\|m = 20\|m=20" R/03_features.R
```
确认 L3 具体是哪种回归、L4 的 PMM 具体取几个近邻（脚本里 5 个近邻，本计划 Task 1 原文档已经
提到过，核实仍成立）、TE_COLS 的确切 8 个列名。

- [ ] **Step 2: 写进论文**

给出 L1（保留缺失标记）/L2（中位数）/L3（回归预测，具体到用的什么模型）/L4（PMM，missRanger，
5 个最近真实观测中随机抽一个）四条线的精确定义；target encoding 的平滑公式与 m=20；
8 个被编码列的清单。写完后检查：读者能否只凭正文里这段定义，重新推出表 8 的命中率口径
（分母=验证折原本缺失、分子=能在训练折编码表里查到）——这是审稿人具体提出的可复现性标准。

- [ ] **Step 3: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: 方法节补全 L1-L4、TE 平滑与编码列定义 (T9)"
```

---

### Task 8: §6 表达力论证改写（比值 vs 超平面的可逼近性）

对应 T3（方法学审稿人）。

现有表述「四项超平面……无论多深都无法逼近」在数学上不成立：轴对齐分裂树可以逼近任意仿射边界，
差别在于**达到同一精度所需的分裂数/样本量**，不是「能不能」。这条论证支撑预算余量特征
+0.00064 的增益解读，逻辑基础要修对，不能只是软化措辞。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（§6 第 3 段附近）

**Interfaces:** 无新产物，纯改写，但改写后的论证必须在逻辑上站得住——审稿人给的方向是
「在可用 n 下，四维边界所需的轴对齐分裂数量超出了模型容量，而两维比值仍在可负担范围内」，
这是把「能不能」换成「多贵」。

- [ ] **Step 1: 改写**

把"无论多深都无法逼近"这类绝对化表述改成基于逼近成本/样本量的说法：比值特征是过原点的
二维线性边界，树用有限的分裂就能逼近到高精度；四项残差是四维空间里的超平面，逼近到同等
精度所需的分裂数量随维度增长，在当前训练规模与树深度设置下实际上逼近不到——这是「容量 vs
可用 n」的问题，不是「可逼近性」本身的问题。改写时不要引入新的定量说法（比如具体的「所需
分裂数」数字）除非能从现有产物或简单核实中得到——如果只能定性论证，就定性论证，不要编造
一个听起来精确但没有测过的数字。

- [ ] **Step 2: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: §6 表达力论证改写为逼近成本视角 (T3)"
```

---

### Task 9: 中心论断与新颖性声明的措辞收敛（跨 Abstract/§1/§10.1/§11）

对应 T1（用户已确认：降级措辞，不做新实验）、T4（新颖性声明）、T11（摘要/引言超出 §10.3
边界，阻塞项）。三条都是「高层声称强度」问题，且都触及 Abstract/§1/§10/§11 里的相近段落，
合并成一个任务，避免多个任务分别改同一段落造成冲突。**建议在 Task 1-8 之后做**，这样收敛
措辞时能考虑到前面新增的证据（比如 Task 1 的 MCAR 补强、Task 6 的可用性声明）是否让某些
限定可以适当收紧或必须进一步放宽。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（Abstract、§1 全部、§10.1、§10.3、§11）

**Interfaces:** 无新产物，纯文本，但改动点分散在全文四处，且必须保持四处口径一致——
这正是 5 位审稿人（尤其 Cc.1）反复强调的问题：不能正文低调、摘要结论高调。

- [ ] **Step 1: T1 — 中心论断降级**

把「a preprocessing step is worth **exactly as much as** it is hard for the downstream
model to reconstruct on its own」这个强度降级为可辩护的说法，比如「...is worth **roughly
in proportion to**...」或「...tracks how hard...」——具体措辞由实现者选定，但必须做到：
(a) Abstract、§1 中心论断方框、§10.1、§11 四处统一用同一套强度用词，不要有的地方降级了
有的地方还是 "exactly as much"；(b) 明确第 1 条贡献是一个**解释性框架**，被三个实例支持，
而不是一条被独立操作化检验过的定律——可以直接说清楚论文实际展示的是「同一变换在不同模型
族上收益不同」这个事实模式，这个框架是对这个模式的一种解释。

- [ ] **Step 2: T11 — 摘要/引言收敛到 §10.3 的单数据集边界**

摘要里「the same transform can carry opposite sign」「two preprocessing steps can
destroy each other」现在是以一般性质陈述的。在摘要与 §1 里，要么在陈述这些结论的同一句里
带上「on this synthetic dataset」这类限定（不是只在第一句提一次数据集背景就完事），要么
明确指向 §10.3 的限制。目标是让审稿人读完摘要就知道这些结论的适用范围，不用等到 §10.3
才发现。

- [ ] **Step 3: T4 — 新颖性声明收敛**

§1 Contribution 2 与 §11 两处「此前未见有人报告」/"to our knowledge, not previously
reported" ——插补把值推出编码查找域、编码查不到回落全局均值，这类"未见类回退/域外值"失效
在应用机器学习里并不新，本文的贡献是把它在这个数据集上**量化**了。删除该声明，或降级为
「在我们检索的范围内，尚未见到把这一失效以逐取值编码命中率的形式量化」这类更谨慎的说法。

- [ ] **Step 4: 通读检查四处口径一致**

改完后，把 Abstract、§1 中心论断、§10.1、§11 四段拿出来放在一起读一遍，确认强度用词、
限定条件、新颖性声明的措辞都彼此一致，没有一处比另一处说得更满。

- [ ] **Step 5: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add paper/preprocessing-expressiveness.md && git commit -m "revision: 中心论断降级措辞 + 摘要收敛至单数据集边界 + 新颖性声明收敛 (T1, T4, T11)"
```

---

### Task 10: 术语/格式扫尾 + 全篇重新核对

对应 T13（写作与表述审稿人）+ 全篇复核。这是最后一个任务，兼做本轮修订的收尾门禁——
类似原计划 Task 9 的角色。

**Files:**
- Modify: `paper/preprocessing-expressiveness.md`（§6 Table 5、References、§2.3、首次出现
  TE 缩写处）

**Interfaces:**
- Consumes: `R/31_check_paper_numbers.R`（已有，本任务要重新跑一遍，因为前面几个任务往
  `docs/实验报告.md` 和论文里都加了新数字）

- [ ] **Step 1: `max_bin` 从「候选特征」改为明确标注为超参数对照**

Table 5（§6）把 `max_bin` 提到 2048 列为候选特征之一，但它是 xgboost 超参数不是派生特征。
改成在表格或紧邻文字里明确标注这是一个非特征的对照项（用于说明"编码能吸收什么"），不要
让它看起来和预算余量特征是同一类东西。

- [ ] **Step 2: 参考文献补年份**

Little & Rubin 那条目前是 `(n.d.)`——检查白名单原文（spec §10）是否其实提供了年份而之前
遗漏了；如果白名单本身没给年份，如实保留 `(n.d.)`（不编造年份），但去核实一下这本书公认的
出版年份是否可以从其他已用的白名单条目风格推断为合理补充——如果不确定就不要加，保持
`(n.d.)` 比编造一个错误年份更负责任。

- [ ] **Step 3: 「第二位小数与行的正例率相关」措辞澄清**

改成「与该精确取值的经验正例率相关」，避免「行的正例率」的歧义。

- [ ] **Step 4: TE 缩写首次定义**

确认 "exact-value target encoding" 在 Abstract 或 §7 第一次出现时给出 "(TE)" 缩写定义，
之后统一用缩写。

- [ ] **Step 5: 0.771 vs 0.758 的口径澄清**

核实 `docs/实验报告.md:102/104` 那两个数字——「24.2%→99.96%」（首箱到末箱，差约 0.758）
与「极差从 0.771 降到 0.024」（很可能是全部分箱里的真实极差，不一定等于首末两箱之差）。
去查产出这两个数字的脚本（`R/15_fig2_audit.R` 或 §1.4 相关），确认这确实是两个不同的量
而不是笔误。如果确认是两个不同口径，在论文 §2.2 提到这两个数字时加一句说明区别；如果
查出来其实是笔误，按脚本产出的真实数字改正（先改 `docs/实验报告.md`，再改论文）。

- [ ] **Step 6: 全篇门禁——重新跑一遍**

Run:
```bash
cd /c/Users/Lenovo/Desktop/project && "D:/R/R-4.6.1/bin/Rscript.exe" R/31_check_paper_numbers.R
```
Expected: 报告匹配情况；本轮修订新增了不少数字（MCAR 检验统计量、假阳性率、逐列命中率
范围等），如果这些新数字对应的产物没有被 `R/31_check_paper_numbers.R` 扫描到（比如存在
`output/mcar_check.rds` 但脚本没读这个文件），**照原计划 Task 9 的方式修一遍脚本的扫描
范围**，让它能验证这一轮新增的产物，而不是留着不管。

再跑：
```bash
cd /c/Users/Lenovo/Desktop/project && grep -nP '[\x{4e00}-\x{9fa5}]' paper/preprocessing-expressiveness.md; rg -c '[\x{4e00}-\x{9fff}]' paper/preprocessing-expressiveness.md; grep -c "CITATION NEEDED" paper/preprocessing-expressiveness.md; wc -w paper/preprocessing-expressiveness.md
```
Expected: 无中文；`[CITATION NEEDED]` 条数已知；字数（本轮会继续增长，不设硬上限，
但报告实际数字）。

- [ ] **Step 7: 提交**

```bash
cd /c/Users/Lenovo/Desktop/project && git add -A && git commit -m "revision: 术语/格式扫尾，重新核对全篇门禁 (T13)"
```

---

## Self-Review

**修订任务表覆盖检查：**

| 修订任务表 ID | 对应任务 |
|---|---|
| T1（用户已定：降级措辞） | Task 9 |
| T2 | Task 1 |
| T3 | Task 8 |
| T4 | Task 9 |
| T5 | Task 2 |
| T6 | Task 3 |
| T7 | Task 4（依赖 Task 2） |
| T8 | Task 6 |
| T9 | Task 7 |
| T10 | Task 5 |
| T11 | Task 9 |
| T12 | Task 1 |
| T13 | Task 10 |

**顺序理由：** 需要新分析的任务（1、3、5，以及依赖 Task 2 的 Task 4）排在前面，因为它们
产出的新证据可能影响后面文本任务（Task 9）怎么写限定条件；Task 2（分辨率下限定义）必须
先于 Task 4，因为 Task 4 要用 Task 2 统一好的定义。Task 9（措辞收敛）放在倒数第二，
因为收敛摘要/结论的语气时应该已经看到前面新增的全部证据。Task 10 收尾，兼任本轮的自动
门禁，角色对应原计划的 Task 9。

**类型一致性：** 「分辨率下限」的定义在 Task 2 里统一，Task 4 直接复用同一个定义，不
重新定义。`output/lattice_hit.rds` 的列结构（Task 1 原计划已定义）在 Task 5 里直接复用，
不重新测。

**占位符扫描：** Task 6 的可用性声明明确要求不编造仓库 URL，缺失时用占位符而非编造——
与原计划对作者占位符的处理原则一致。
