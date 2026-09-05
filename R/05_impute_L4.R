# =============================================================================
# 05_impute_L4.R —— 第四条线：随机插补（PMM）
#
# 负责人：D
#
# -----------------------------------------------------------------------------
# 这条线在四条线里的位置
# -----------------------------------------------------------------------------
# 四条线构成一个阶梯，只变化一个量：**插补对点估计的承诺程度**。
#
#   L1  完全不承诺      保留 NA，让 GBDT 自己学「遇到缺失往哪走」
#   L2  承诺一个常数    所有缺失都填同一个中位数
#   L3  承诺条件均值    用回归预测，取条件期望
#   L4  不承诺具体值    从真实观测值里随机抽一个「像你的人」的取值
#
# 这个阶梯是为了回答一个具体问题：L1 > L2 > L3 的实测结果显示，
# 插补越精细分数越低。一个可能的机制是「收缩」——
# 回归估计天然向条件均值靠拢，压缩了插补行的特征分布，
# 而标签对 screen_time 是陡峭单调的，一压缩就把这些行推向基准率。
#
# 如果这个机制是对的，那么把随机性加回去应该有救。
# PMM（predictive mean matching）正是这么做的：
# 先用随机森林预测一个值，然后不用这个预测值，
# 而是去找预测值最接近的 k 个**真实观测样本**，从中随机抽一个的实际取值填进去。
#
# 结果因此保有真实数据的分布形状，不会出现「所有插补值都挤在均值附近」。
#
# L4 > L3 → 收缩机制成立
# L4 ≈ L3 → 收缩不是主因，要另找解释
#
# -----------------------------------------------------------------------------
# 关于「多重插补」
# -----------------------------------------------------------------------------
# 本文件早期版本生成 m 份插补再取平均 —— 那是错的。
# 取平均会把 PMM 好不容易保住的随机性又抹平回条件均值，
# 等于绕一大圈回到 L3，测不出任何东西。
#
# 真正的多重插补是：m 份插补数据各训一个模型，平均它们的**预测**，
# 而不是平均插补值。那需要框架区支持「一条线出 m 个模型」，
# 是一个独立的扩展方向，留给主线跑通之后再做。
# =============================================================================

library(missRanger)

NUM_COLS_L4 <- c(
  "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
  "work_study_hours", "sleep_hours", "notifications_per_day",
  "app_opens_per_day", "weekend_screen_time"
)

CAT_COLS_L4 <- c("gender", "stress_level", "academic_work_impact")

# 算力预算（实测后设定）：
# 框架每折调用 apply_imputer 两次（训练折 + 验证折），5 折共 10 次。
# 每次都要在十几万行上跑链式随机森林，所以树数和迭代轮数都压得比较低。
# D 的机器如果强可以往上调，但要在报告里注明用的是哪一组参数。
NUM_TREES <- 50L   # 每棵随机森林的树数
MAX_ITER  <- 2L    # 链式插补的迭代轮数
PMM_K     <- 5L    # 从预测值最接近的 k 个真实观测里随机抽一个

#' 拟合插补器
#'
#' missRanger 是「拟合与应用一体」的设计，没有可以保存下来套用到新数据的
#' 模型对象。为了既保持接口一致又不泄漏，这里只记录训练折的特征，
#' 真正的插补在 apply 阶段做：把待插补的数据拼在训练折后面一起跑，
#' 结束后只取回前面那一段。
#'
#' 这样训练折的特征会参与插补（这是对的，插补器本来就该从训练数据学），
#' 而 addicted_label 从头到尾没进过 missRanger，不构成标签泄漏。
#'
#' @param dt data.table，训练折的数据
#' @return list
fit_imputer_L4 <- function(dt) {
  cols <- c(NUM_COLS_L4, CAT_COLS_L4)
  # seed 必须由框架的随机流派生，不能写死。
  # 框架在每折调用前都会 set.seed(SEED + k)，所以这里抽出来的 seed
  # 各折不同（保证折间独立），但整体流程仍然完全可复现。
  #
  # 早期版本在 apply 里硬编码 seed = 1000L，导致 5 折的 missRanger
  # 用了完全相同的随机种子 —— 折与折之间的插补结果存在非预期的相关性。
  list(line  = "L4",
       donor = dt[, ..cols],
       seed  = sample.int(.Machine$integer.max, 1L))
}

#' 应用插补器：单次随机插补
#'
#' @param imp fit_imputer_L4 的返回值
#' @param dt data.table，要处理的数据（会被原地修改）
#' @return 插补后的 dt
apply_imputer_L4 <- function(imp, dt) {
  cols <- c(NUM_COLS_L4, CAT_COLS_L4)

  n_target <- nrow(dt)
  combined <- rbind(dt[, ..cols], imp$donor)   # 待插补的在前，供体在后

  filled <- missRanger::missRanger(
    as.data.frame(combined),
    num.trees = NUM_TREES,
    maxiter   = MAX_ITER,
    pmm.k     = PMM_K,     # ★ 这一行是本条线的全部要点：
                           # 不用随机森林的预测值，而是从预测值最接近的
                           # PMM_K 个真实观测样本里随机抽一个的实际取值。
                           # 插补结果因此落在真实数据见过的取值上，
                           # 分布形状得以保留，不会向均值收缩。
    seed      = imp$seed,   # 由 fit_imputer_L4 从框架随机流派生，各折不同
    verbose   = 0
  )

  # 只取回前面那一段（供体部分丢弃）
  for (cc in NUM_COLS_L4) {
    set(dt, j = cc, value = as.numeric(filled[[cc]][seq_len(n_target)]))
  }
  for (cc in CAT_COLS_L4) {
    set(dt, j = cc, value = factor(as.character(filled[[cc]][seq_len(n_target)]),
                                   levels = levels(dt[[cc]])))
  }

  dt
}
