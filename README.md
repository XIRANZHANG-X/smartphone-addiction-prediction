# 智能手机成瘾预测

> 课程小组项目 · 10 人 · 交付 2026-09-06 · 汇报 2026-09-08

用 R 预测用户是否手机成瘾。评价指标 **ROC-AUC**，Kaggle 站内赛实时榜。

---

## 一句话说清这个项目在做什么

数据里 **61% 的行至少缺一个特征**，而标签在数据完整时几乎是确定的。所以这**不是一道建模题，是一道缺失数据题**——名次差异出现在小数点后第三位。

我们原本打赌「用数据的硬约束做精细插补」能赢：

```
每日屏幕时间 >= 社交媒体时间 + 游戏时间 + 工作学习时间   （421,427 行中 100% 成立）
```

**这个赌打输了**，而输的过程比赢更有价值。项目最终的核心结论是一个交互效应：

> **同一项预处理，在梯度提升树上可以是最差的，在线性模型上是决定性最优的。**
> **一项预处理的价值，等于该变换有多难被下游模型自己造出来。**
> 这条结论有三个独立实例（插补、派生比值、逐取值编码），方向完全一致；
> 最极端的一个：同一个编码在 glmnet 上值 +0.0335，在 xgboost 上只值 +0.0042。

完整的推理链（三次自我证伪 + 一次成功的反向预测，每步都有配对检验）见
[docs/项目说明.md](docs/项目说明.md) 第六节。

---

## 当前进度

**全部代码已完成并跑通，已两次提交 Kaggle。**
[方法学审查](docs/审查响应.md)的 13 条全部闭环（12 条修复 + 1 条附理由不采纳）。

| 阶段 | 本地 CV | Kaggle 榜单 |
|---|---|---|
| 初始基线（固定 600 轮） | 0.95910 | — |
| 加早停 | 0.96088 | — |
| 加超参数搜索 | 0.96145 | — |
| 最优单模型（全量，编码前） | 0.96465 | — |
| **九成员集成 —— 提交的就是它** | **0.96487** | **0.96627** |
| 赛后：更正两处定义 + 逐取值编码，最优单模型 | **0.96784** | 赛后，榜单已关闭 |
| 赛后：十四成员集成（秩空间 logistic） | **0.96807** | 同上 |

**本地验证已被榜单二次验证**：两次提交的 CV 与 LB 差值稳定在 +0.0014；
第二次提交前据此预测 0.96623，实际 0.96627，误差 4×10⁻⁵。

8 张图表见 `reports/figures/`。详细状态见 [docs/进度.md](docs/进度.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [**docs/实验报告.md**](docs/实验报告.md) | **组员从这里开始** —— 全部实验结果与结论的统一汇总，含子样本有效性验证 |
| [docs/项目总览.md](docs/项目总览.md) | 方法说明：工作流程、特征工程、每一步的理由，含全部图表 |
| [docs/进度.md](docs/进度.md) | 已完成 / 待完成 / 实验结果，项目状态的唯一真源 |
| [docs/审查响应.md](docs/审查响应.md) | 对方法学审查的逐条回应（含一条**不采纳**及其理由） |
| [docs/讨论区核查.md](docs/讨论区核查.md) | 竞赛讨论区 54 帖的做法核查：复现了什么、我们漏了什么、不采纳什么 |
| [output/results.md](output/results.md) | 全部实验数字，由 `R/11_report.R` 自动生成 |
| [docs/项目说明.md](docs/项目说明.md) | 技术细节：完整推理链、配对检验、方法学声明 |
| [docs/小组分工.md](docs/小组分工.md) | 10 人分工、每人的产出与验收标准 |
| [docs/时间线.md](docs/时间线.md) | 16 天计划、两个硬门禁 |
| [docs/superpowers/specs/](docs/superpowers/specs/) | 完整设计文档（含已被实测推翻的原始假设） |

---

## 快速开始

```r
# 1. 在 RStudio 里打开 smartphone-addiction-prediction.Rproj
# 2. 装包（只需一次）
source("R/00_setup.R")

# 3. 数据不在 git 里，需自行从 Kaggle 竞赛页下载并放入 data/raw/
#    train.csv  test.csv  sample_submission.csv

# 4. 生成共享产物（折叠契约，全组只跑一次）
source("R/01_load.R")
source("R/03_features.R")
source("R/04_folds.R")
```

### 实验产物：从 Releases 拿，不要重跑

`output/` 整个在 `.gitignore` 里（只有 `folds.rds`、`subsample_200k.rds`、
`results.md` 三个例外），所以 clone 之后是空的。跑一遍全部实验要七八个小时。

**[Releases](https://github.com/XIRANZHANG-X/smartphone-addiction-prediction/releases) → 下最新的 `artifacts-YYYY-MM-DD.zip`，解压后把 `output/` 覆盖到项目根目录。**

里面有全量 4×4 网格（14 格）、Tier A 网格、样本量阶梯四级、重复 CV
与全部专项结果。**不含**竞赛数据本身（`raw_train.rds` / `raw_test.rds` /
`features_raw.rds`）——那三个由上面第 3、4 步自己生成，几秒钟。

> 为什么不直接提交进 git：`output/` 有 819 MB，而 git 会永久保存每一个版本，
> 每次重跑网格都会再增加约 75–100 MB 且删不掉。Release 附件不进 git 历史。

---

## 代码架构

模型代码分三层，加一个功能只需改一处：

| 层 | 文件 | 职责 |
|---|---|---|
| 框架 | `R/06_framework.R` | 数据加载、Tier 选择、**防泄漏的 CV 循环**、存盘。共用，不要改 |
| 模型 | `R/lib_models.R` | 工厂 `make_xgb()` / `make_lgb()` / `make_ranger()` / `make_glmnet()`，早停只写一遍 |
| 配置 | `R/06_model_*.R` | **27 行**，只有三个配置变量和一行 `fit_predict <- make_xgb()` |

环境变量覆盖，不用为跑变体去改 14 个文件：

```bash
TIER=B        Rscript R/06_model_L1_xgboost.R   # 全量重训
REPEAT_ID=1   Rscript R/06_model_L1_xgboost.R   # 重复 CV
USE_DERIVED=0 Rscript R/06_model_L1_xgboost.R   # 关掉派生特征
USE_TE=0      Rscript R/06_model_L1_xgboost.R   # 关掉逐取值编码
POOL_FILE=output/pools/pool_100k.rds \
              Rscript R/06_model_L1_xgboost.R   # 指定行池（样本量阶梯用）
IMPUTE_CACHE=1 TIER=B \
              Rscript R/06_model_L4_xgboost.R   # 缓存复用插补结果（L4 专用）
```

`IMPUTE_CACHE=1` 是 L4 的必备项：四个算法在同一折上拿到的 PMM 插补逐位相同，
缓存后全量四格从 15.6 小时降到 4.2 小时。等价性见 `R/26_cache_check.R`。

一键跑完全部实验：`Rscript R/run_pipeline.R`

## 目录结构

```
├── R/                   全部代码
│   ├── 00_setup.R       装包 + 环境检查
│   ├── 01_load.R        读数据
│   ├── 02_eda.R         探索性分析
│   ├── 03_features.R    特征工程（全组唯一真源）
│   ├── 04_folds.R       ★ 折叠契约，冻结后不得重跑
│   ├── 05_impute_L*.R   四条插补线
│   ├── 06_model_*.R     ★ 每人一个文件，互不干扰
│   ├── 07_ensemble.R    集成
│   └── 08_submit.R      生成提交文件
├── data/raw/            原始数据（不进 git）
├── output/              folds.rds 等契约产物
├── reports/figures/     8 张图表（300 dpi，可直接用于汇报）
├── slides/              汇报幻灯片
├── submissions/         提交记录 + log.csv 分数台账
└── docs/                中文文档
```

---

## 协作规则（三条，请务必遵守）

**1. `output/folds.rds` 冻结后任何人不得重新生成。**
它是所有人分数可比的唯一基础。重跑一次，全组之前的实验结果全部作废。

**2. 每人只改自己的 `R/06_model_<你的名字>.R`，不动别人的文件。**
这样 10 个人同时 push 不会冲突，也不需要开分支。

**3. 插补必须在每一折内部拟合。**
在全训练集上拟合插补再套用会造成信息泄漏，CV 分数虚高。模板 `R/06_model_TEMPLATE.R` 已经把正确的循环结构写好了，照抄即可。

**4. 逐取值 target encoding 同样必须在每一折内部拟合。**
和插补是同一条纪律。在 CV 循环之外拟合编码器（哪怕用同一套折）也是泄漏——
训练部分的每个样本都会被含验证折标签的统计量编码。框架已处理，`USE_TE=0` 可关闭。

---

## 提交产物接口

每条模型线产出两个文件，形状固定：

| 文件 | 长度 | 说明 |
|---|---|---|
| `output/oof/oof_<名字>.rds` | 691,369 | 全量交叉验证预测 |
| `output/test/test_<名字>.rds` | 296,302 | 测试集预测 |

对比实验（200k 子集）另出 `output/oof/oof_grid_<名字>.rds`，长度 200,000。

`07_ensemble.R` 自动读取目录下所有 `oof_*.rds`——**加模型不需要改任何已有代码**。
