# 给 Chai 的接续说明

> 2026-09-03。内容有三部分：**① 你的提交去哪了、怎么恢复；② 你的哪几个脚本需要改；③ 一件跟你的工作直接相关的发现。**
>
> 先说结论：**你的三个提交一个都没丢**，它们完整地在远端分支 `eda/chai` 上。
> 但**你不要直接 `git pull`**——远端 `main` 的历史被改写过，直接拉会把两条历史搅在一起。
> 下面第一节给了安全的操作步骤。

---

## 一、发生了什么，以及你该怎么操作

### 1.1 发生了什么

你之前把这三个提交推到了 `main`：

```
412f738  新增基础层 EDA：特征分布、完整相关性矩阵、缺失联合结构
08c767d  新增三项 EDA 分析：插补约束合规性、train/test 分布一致性、跨模型共同错判行
4316887  新增 EDA 探索：谁打破了屏幕时间定的近乎确定规律
```

按原定流程，这些应当先开分支、review 之后再合进 `main`。所以它们被**原样搬到了
`eda/chai` 分支**，`main` 则改写成不含它们的历史。

**搬运顺序是先推分支、再改写 main**，所以任何时刻你的提交都至少存在于一处，没有丢失风险。
你现在可以直接验证：

```bash
git fetch origin
git log --oneline origin/eda/chai -4
```

应当看到你那三个提交，以及它们的父提交 `62132fb`。

### 1.2 你该怎么操作

**⚠ 第 3 步会丢弃未提交的本地改动。先跑 `git status` 确认工作区是干净的**，
不干净就先 `git stash` 或先提交。

```bash
# 0. 取回远端最新状态
git fetch origin

# 1. 给你当前的本地 main 打一个保险分支（不推远端，纯本地留底）
git branch backup-local-main

# 2. 确认你的三个提交完好 —— 看到 412f738 / 08c767d / 4316887 再往下走
git log --oneline origin/eda/chai -4

# 3. 把本地 main 对齐到远端（此时你的提交已在 eda/chai 和 backup-local-main 两处）
git checkout main
git reset --hard origin/main

# 4. 要继续做你的 EDA 时，切到你自己的分支
git checkout -b eda/chai origin/eda/chai
```

确认一切正常之后，`backup-local-main` 就可以删掉：`git branch -D backup-local-main`。

### 1.3 关于图片

你的提交 `4316887` 改动了 `fig1`~`fig5`、`fig7` 六张图。从体积变化看
（fig1 从 1.69 MB 变成 0.77 MB），这像是**在你机器上重新渲染**造成的字节差异，
而不是内容不同——同一份 `R/12_figures.R`，字体和图形设备不一样，出来的 PNG 就不一样。

`main` 上这八张图已于 2026-09-03 全部重新生成。**第 3 步的 `reset --hard` 会让
你本地这六张变成 `main` 的版本，这是预期行为，不用管。**

你自己的 `fig9a`/`fig9b`/`fig10`/`fig11a`/`fig11b` 是独有的，不冲突，仍在你的分支上。

> **已知问题，不用你修**：`main` 上 `fig1` 的副标题现在还写着**三项**约束
> （"每日屏幕时间 ≥ 社交媒体时间 + 游戏时间"），而正确的是四项。
> 这是我们这边的遗留，已经排进修复计划了。**你不要自己去改它**，否则又是一处冲突。

---

## 二、你需要改的地方

`main` 这两周有较大改动，对你五个脚本的影响**逐个**说明如下。

### 2.1 `R/18_constraint_faithfulness.R` —— 需要实质修改

**问题：你用的硬约束是三项的，而真正的约束是四项。**

你脚本第 10 行的注释和第 40 行的判定用的是：

```r
daily_screen_time_hours < social_media_hours + gaming_hours - 1e-6
```

生成器真正强制的是**四项**：

```
daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours
```

**判据不是"成立比例"，是"残差最小值"。** 两个版本在完整数据上都 100% 成立，
但四项版的残差最小值恰好触到 **0.000**，三项版停在 **0.100**——
说明三项版只是四项版蕴含的推论，四项版才是真正的边界。
核查脚本是 `R/17_discussion_checks.R`（`main` 上的那个）。

语义上的含义是：**`work_study_hours` 被算在屏幕时间里面**，也就是"在手机上做的工作/学习"。

**你的脚本问的是"插补对硬约束有多不老实"，那就必须拿真正的约束去问。**
用三项的话，你测到的违约率会系统性偏低——因为三项的约束更松。

改法：把第 40 行（以及脚本里其它用到该约束的地方）改成

```r
viol_cc <- cc[, mean(daily_screen_time_hours <
                     social_media_hours + gaming_hours + work_study_hours - 1e-6)]
```

并同步改掉第 10 行的注释。

### 2.2 `R/20_universally_hard_rows.R` —— 口径变了，结论要么加声明、要么重跑

你的脚本读的是：

```r
oof_files <- list.files("output/oof", pattern = "^oof_grid_.*\\.rds$", full.names = TRUE)
```

`oof_grid_*` 是 **Tier A、17 特征、加入逐取值编码之前**的那套预测。
它们没有被改动（还是 8 月 23 日那批），所以**你的脚本照样能跑、数字不变**。

但项目的现役口径已经变成 **25 特征**（12 原始 + 5 派生 + 8 个逐取值 target encoding），
全量最优单模型从 0.96465 提到 **0.96784**。所以你现在算出的"跨模型共同错判行"
描述的是一个**我们已经不再使用的配置**。

两条路，你选：

**（a）便宜：加一句口径声明。** 在脚本头部写明"本分析基于编码之前的 17 特征配置"。
结论作为历史记录仍然有效，不用重跑。

**（b）更好：换成现役口径。** 跑

```bash
"D:/R/R-4.6.1/bin/Rscript.exe" R/25_size_ladder.R 200k
```

约 25 分钟，产出 `output/ladder/oof_pool_200k_*.rds`——
**同样的 200,000 行、同样的折划分，但是 25 特征口径**，而且有 14 格（含 L4）。
然后把你的 pattern 改成：

```r
oof_files <- list.files("output/ladder", pattern = "^oof_pool_200k_.*\\.rds$",
                        full.names = TRUE)
# 注意去前缀也要跟着改
nm <- sub("^oof_pool_200k_", "", sub("\\.rds$", "", basename(f)))
```

我建议 (b)。你这个分析的价值在于"哪些行是所有模型都判错的"，
而这个问题在现役配置上问才有指导意义。

### 2.3 这三个脚本不受影响，可以照跑

| 脚本 | 依赖 | 结论 |
|---|---|---|
| `R/17_residual_exceptions.R` | 只读 `output/raw_train.rds` | **不受影响** |
| `R/19_train_test_shift.R` | 只读 `raw_train.rds` / `raw_test.rds` | **不受影响** |
| `R/21_foundational_eda.R` | 只读 `raw_train.rds` | **不受影响** |

### 2.4 脚本编号要改

你的 17~21 和 `main` 上的 17~21 **编号全部撞车**（文件名不同，所以 git 不会报冲突，
但合并之后编号就没有意义了）：

| 编号 | `main` 上的 | 你的 |
|---|---|---|
| 17 | `17_discussion_checks.R` | `17_residual_exceptions.R` |
| 18 | `18_new_features.R` | `18_constraint_faithfulness.R` |
| 19 | `19_adversarial.R` | `19_train_test_shift.R` |
| 20 | `20_feature_v2.R` | `20_universally_hard_rows.R` |
| 21 | `21_te_by_family.R` | `21_foundational_eda.R` |

`main` 上 00~28 已占用，29/30/31 已被后续工作预定。
**请把你的五个改成 32~36**，在提 PR 之前改好，合并时零摩擦：

```bash
git mv R/17_residual_exceptions.R      R/32_residual_exceptions.R
git mv R/18_constraint_faithfulness.R  R/33_constraint_faithfulness.R
git mv R/19_train_test_shift.R         R/34_train_test_shift.R
git mv R/20_universally_hard_rows.R    R/35_universally_hard_rows.R
git mv R/21_foundational_eda.R         R/36_foundational_eda.R
```

改完记得搜一遍别的文档里有没有引用旧文件名。

### 2.5 如果你以后要跑模型（现在不用管）

框架有三处改动，都是向后兼容的，但结果会变：

- `derive_features()` 改了两列：`other_screen` 现在是**四项**残差，
  `free_frac` 的分母改成 `24 - sleep`（原来把 `work_study` 减了两遍）。
  任何用到派生特征的旧数字都不能再用。
- `prepare_fold()` 多了一个可选参数 `cache_file`（有默认值，不传照常工作）。
- `R/06_framework.R` 多了两个环境变量 `POOL_FILE`、`IMPUTE_CACHE`，
  并且**在建模前会重设一次随机种子**。这对 L1/L2/L3 是空操作，但会改变 L4 的结果。

另外：**`output/` 整个在 `.gitignore` 里**（只有 `folds.rds`、`subsample_200k.rds`、
`results.md` 三个例外）。所以你从 git 拿不到任何实验产物，你机器上那份是你自己跑出来的。
`folds.rds` 和 `subsample_200k.rds` 是冻结契约，从头到尾没动过，全组一致。

---

## 三、两件跟你的工作直接相关的事

### 3.1 一处署名更正 —— 训练/测试缺失率的发现是你的

`main` 上 `R/02_eda.R` 的"发现 12"（训练集与测试集的逐列缺失率不同）
一度被记成本轮新增。核对之后确认：**你在 2026-08-26 的
`R/19_train_test_shift.R` 第三节就已经做了。** 已经在 `R/02_eda.R`
和 `docs/讨论区核查.md` 两处更正为你的发现。

本轮在你之上只补了两样：给出了二项标准误与 z 值（你用的是 0.5 个百分点的经验阈值），
以及区分了"缺失与标签无关"和"缺失与训练/测试身份相关"是**两个不同的问题**——
前者不成立、后者成立。

### 3.2 你那个"插补对硬约束有多不老实"的问题，有了一个意外的答案

这是本轮最有价值的一个发现，而**它和你 `18_constraint_faithfulness.R` 问的是同一件事的两面**，
值得你知道。

你问的是：插补填出来的值，违反硬约束到什么程度。

本轮问的是另一个方向：插补填出来的值，**下游还能不能用**。答案是——

数据被舍入在一个 **0.01 的格点**上，逐取值 target encoding 的编码表就是按精确取值建的。
而 L3（回归预测插补）填出来的是任意实数，**不在格点上，编码表里查不到**：

| 插补线 | 填的是什么 | 在编码表里查得到的比例 |
|---|---|---|
| L2（中位数） | 真实观测的中位数，在格点上 | **100.00%** |
| **L3（回归预测）** | 任意实数，不在格点上 | **0.03% ~ 0.19%** |
| L4（PMM） | 从真实样本里抽的实际取值，在格点上 | **99.98% ~ 100.00%** |

至少缺一列的行占 61%，所以编码对六成以上的行失效。后果是 L3 整条线在全量上崩掉
（L3_xgboost 0.94770，而 L1 是 0.96784）。

**一般化的说法：在有逐取值编码的流程里，插补必须填回格点上的值——
这跟"哪种插补更准"是正交的两件事。** L3 在统计意义上填得更准，但它填出的值下游用不了。

你那个"合规性"的角度和这个"可用性"的角度合起来，才是完整的
"插补会破坏下游什么"。**建议合成一条线来做**，你那份的价值在于它先问了这个方向。

> 完整记录见 `main` 上的 `docs/实验报告.md` 第 10.1 节（另有第 8 节讲逐取值编码本身）。
> 那份文档是现在的入口，1400 行，把全部实验结果与结论串成了一条线。

---

## 四、需要你做的事，按顺序

1. `git status` 确认工作区干净，然后按 §1.2 的四步恢复。
2. 把五个脚本改名成 32~36（§2.4）。
3. 改 `33_constraint_faithfulness.R` 的约束为四项（§2.1）。**这一条最要紧**，
   因为不改的话你测的是一个更松的约束，违约率会系统性偏低。
4. 决定 `35_universally_hard_rows.R` 走 (a) 还是 (b)（§2.2）。
5. 改完在 `eda/chai` 上提 PR。

有任何一步对不上，先别硬来，问一声。
