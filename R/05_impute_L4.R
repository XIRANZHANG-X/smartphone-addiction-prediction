# =============================================================================
# 05_impute_L4.R —— 第四条线：多重插补
#
# 负责人：D（建议分配给机器最好的人，这条线最吃算力）
#
# -----------------------------------------------------------------------------
# 单次插补有一个根本问题：它假装自己猜对了。
#
# 填进去一个数之后，后面所有步骤都把它当成观测值对待，
# 于是「我不确定这个人到底刷了几小时手机」这件事就被抹掉了。
#
# 多重插补的做法是：生成 m 份不同的完整数据集（每份的随机性不同），
# 各训一个模型，最后把 m 份预测平均。插补的不确定性因此被保留下来，
# 而且平均本身还有降方差的效果。
#
# 我们用 missRanger：基于随机森林的链式插补，能自动捕捉非线性关系
# 和变量间的交互，比线性方法更灵活。代价是慢。
#
# 注意：这条线是四条里唯一可以降级的。如果跑不动：
#   - 先把 M_IMPUTE 从 5 降到 3
#   - 还不行就降 NUM_TREES
#   - 再不行告诉组长，整条线可以砍掉，不影响主线
# =============================================================================

library(missRanger)

NUM_COLS_L4 <- c(
  "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
  "work_study_hours", "sleep_hours", "notifications_per_day",
  "app_opens_per_day", "weekend_screen_time"
)

CAT_COLS_L4 <- c("gender", "stress_level", "academic_work_impact")

M_IMPUTE  <- 5L    # 生成几份插补数据集
NUM_TREES <- 100L  # 每棵随机森林的树数。默认 500 太慢，100 精度够用。
MAX_ITER  <- 3L    # 链式插补的迭代轮数

#' 拟合插补器
#'
#' missRanger 的设计是「拟合与应用一体」——它没有可以保存下来套用到新数据的
#' 模型对象。为了保持接口一致并且不泄漏，这里只记录训练折的数据，
#' 真正的插补在 apply 阶段做：把待插补的数据拼在训练折后面一起跑，
#' 结束后只取回待插补的那部分。
#'
#' 这样训练折的信息会参与插补（这是对的，插补器本来就该从训练数据学），
#' 但验证折的标签从头到尾没参与，不构成泄漏。
#'
#' @param dt data.table，训练折的数据
#' @return list
fit_imputer_L4 <- function(dt) {
  cols <- c(NUM_COLS_L4, CAT_COLS_L4)
  list(line = "L4", donor = dt[, ..cols])
}

#' 应用插补器：生成 m 份插补并取平均
#'
#' 取平均而不是返回 m 份，是为了让本条线能套进统一的框架区。
#' 严格的多重插补应该是「m 份各训一个模型再平均预测」，
#' 而这里是「m 份插补取平均再训一个模型」——这是简化版。
#'
#' 两者的差别值得在报告里讨论：前者保留了模型层面的不确定性，
#' 后者只保留了插补层面的。如果时间允许，D 可以额外实现严格版本
#' 作为对比，那会是一个很好的技术深度加分项。
#'
#' @param imp fit_imputer_L4 的返回值
#' @param dt data.table，要处理的数据（会被原地修改）
#' @return 插补后的 dt
apply_imputer_L4 <- function(imp, dt) {
  cols <- c(NUM_COLS_L4, CAT_COLS_L4)

  n_target <- nrow(dt)
  combined <- rbind(dt[, ..cols], imp$donor)   # 待插补的在前，供体在后

  # 累加 m 份插补结果，最后除以 m
  acc <- NULL

  for (m in seq_len(M_IMPUTE)) {
    set.seed(1000L + m)   # 每份用不同的种子，这样 m 份才有差异
    filled <- missRanger::missRanger(
      as.data.frame(combined),
      num.trees   = NUM_TREES,
      maxiter     = MAX_ITER,
      pmm.k       = 3L,     # predictive mean matching：从真实观测值里取，
                            # 避免插补出训练数据里根本不存在的取值
      seed        = 1000L + m,
      verbose     = 0
    )
    head_part <- filled[seq_len(n_target), NUM_COLS_L4, drop = FALSE]

    if (is.null(acc)) {
      acc <- as.matrix(head_part)
      # 类别列不做平均（平均没有意义），直接用第一份的结果
      for (cc in CAT_COLS_L4) {
        set(dt, j = cc, value = filled[[cc]][seq_len(n_target)])
      }
    } else {
      acc <- acc + as.matrix(head_part)
    }
  }

  acc <- acc / M_IMPUTE
  for (cc in NUM_COLS_L4) {
    set(dt, j = cc, value = as.numeric(acc[, cc]))
  }

  dt
}
