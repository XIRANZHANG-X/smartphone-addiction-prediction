# =============================================================================
# 05_impute_L3.R —— 第三条线：约束条件插补 ★
#
# 负责人：C
#
# -----------------------------------------------------------------------------
# 这是全项目的核心方法。原理见 docs/项目说明.md 第五节。
# -----------------------------------------------------------------------------
#
# 普通插补只利用统计相关性：拿其他列回归预测这一列，填进去，完事。
# 我们在这之上加一层「约束投影」——插补结果不许违反数据本身的硬规律。
#
# 那条硬规律是（在 451,246 行中 100% 成立，不是统计相关，是结构性约束）：
#
#     每日屏幕时间 >= 社交媒体时间 + 游戏时间
#
# 因为屏幕时间就是由社交 + 游戏 + 其他用途组成的。
#
# 举例说明它为什么有用：某人社交 3 小时、游戏 2 小时、屏幕时间缺失。
# 普通回归可能预测出 4.2 小时 —— 但这是不可能的，因为 3 + 2 = 5 > 4.2。
# 我们把它抬到至少 5 小时。这一步排除了大量不可能的取值，精度自然提高。
#
# -----------------------------------------------------------------------------
# 算法
# -----------------------------------------------------------------------------
#   拟合阶段（只在训练折上跑）：
#     1. 记录每列的中位数，用于初始化
#     2. 为 9 个数值列各拟合一个线性模型：用其他 8 列预测这一列
#        （预测变量里的缺失先用中位数占位）
#
#   应用阶段：
#     1. 中位数初始化
#     2. 用拟合好的线性模型预测每一个缺失格
#     3. ★ 约束投影：只调整原本缺失的格子，观测到的值一个都不动
#     4. 重复 2–3 共 N_ITER 轮，让相互依赖的插补值收敛
#
# 第 3 步的「只调整原本缺失的格子」很关键。如果连观测值一起改，
# 就等于篡改数据，那是另一回事了。
# =============================================================================

NUM_COLS_L3 <- c(
  "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
  "work_study_hours", "sleep_hours", "notifications_per_day",
  "app_opens_per_day", "weekend_screen_time"
)

CAT_COLS_L3 <- c("gender", "stress_level", "academic_work_impact")

# 迭代轮数。
#
# 早期版本写 2 并注释「实测 2 轮已经收敛，再多只是浪费时间」——
# 那句话是**编的**，从来没有人量过。加上收敛诊断之后第一次运行就报警了。
#
# 实测的相对变化轨迹（按各列标准差归一化）：
#   第 1 轮 0.3800
#   第 2 轮 0.0433   <- 仍有 4.3% 个标准差在动，远未收敛
#   第 3 轮 0.0137
#   第 4 轮 0.0063   <- 才低于 1% 的收敛阈值
#   第 8 轮 0.0010
#
# 取 4 轮。再往上收益进入千分之一量级，不值得多花时间。
N_ITER_L3 <- 4L

# -----------------------------------------------------------------------------
# 约束投影
# -----------------------------------------------------------------------------
#' 把插补结果投影回可行域
#'
#' @param dt data.table，已经填过一轮的数据
#' @param na_mask list，每列的原始缺失位置（逻辑向量）。只有这些位置会被修改。
#' @return 原地修改的 dt
#'
.project_constraints_L3 <- function(dt, na_mask) {

  scr <- dt$daily_screen_time_hours
  soc <- dt$social_media_hours
  gam <- dt$gaming_hours
  wkd <- dt$weekend_screen_time

  # --- 全体非负 -------------------------------------------------------------
  # 线性模型完全可能预测出负的小时数，这显然无意义。
  for (cc in NUM_COLS_L3) {
    bad <- na_mask[[cc]] & (dt[[cc]] < 0)
    if (any(bad, na.rm = TRUE)) set(dt, i = which(bad), j = cc, value = 0)
  }
  scr <- dt$daily_screen_time_hours
  soc <- dt$social_media_hours
  gam <- dt$gaming_hours

  # --- 约束一：屏幕时间缺失时，抬到 >= 社交 + 游戏 --------------------------
  # 只在屏幕时间原本缺失、且社交和游戏都不缺的行上生效。
  m <- na_mask$daily_screen_time_hours &
       !na_mask$social_media_hours & !na_mask$gaming_hours
  if (any(m)) {
    lower <- soc[m] + gam[m]
    set(dt, i = which(m), j = "daily_screen_time_hours",
        value = pmax(scr[m], lower))
  }
  scr <- dt$daily_screen_time_hours

  # --- 约束二：社交时间缺失时，压到 <= 屏幕 - 游戏 --------------------------
  m <- na_mask$social_media_hours &
       !na_mask$daily_screen_time_hours & !na_mask$gaming_hours
  if (any(m)) {
    upper <- pmax(scr[m] - gam[m], 0)
    set(dt, i = which(m), j = "social_media_hours",
        value = pmin(soc[m], upper))
  }
  soc <- dt$social_media_hours

  # --- 约束三：游戏时间缺失时，压到 <= 屏幕 - 社交 --------------------------
  m <- na_mask$gaming_hours &
       !na_mask$daily_screen_time_hours & !na_mask$social_media_hours
  if (any(m)) {
    upper <- pmax(scr[m] - soc[m], 0)
    set(dt, i = which(m), j = "gaming_hours",
        value = pmin(gam[m], upper))
  }

  # --- 约束四：社交和游戏同时缺失时，按比例压缩 ----------------------------
  # 两个都是猜的，无法判断谁该让步，所以按各自预测值的比例等比缩放，
  # 让它们的和正好等于屏幕时间。
  m <- na_mask$social_media_hours & na_mask$gaming_hours &
       !na_mask$daily_screen_time_hours
  if (any(m)) {
    idx <- which(m)
    s <- dt$social_media_hours[idx]
    g <- dt$gaming_hours[idx]
    cap <- dt$daily_screen_time_hours[idx]
    tot <- s + g
    over <- tot > cap & tot > 0
    if (any(over)) {
      scale <- ifelse(over, cap / tot, 1)
      set(dt, i = idx, j = "social_media_hours", value = s * scale)
      set(dt, i = idx, j = "gaming_hours",       value = g * scale)
    }
  }

  # --- 约束五：周末屏幕时间的比值先验 ---------------------------------------
  # 实测 weekend / daily 的比值均值 1.33、标准差 0.50，相关 +0.80。
  # 把插补值限制在 [0.5, 3.0] 倍区间内，剪掉明显离谱的预测。
  m <- na_mask$weekend_screen_time & !na_mask$daily_screen_time_hours
  if (any(m)) {
    idx <- which(m)
    base <- dt$daily_screen_time_hours[idx]
    set(dt, i = idx, j = "weekend_screen_time",
        value = pmin(pmax(dt$weekend_screen_time[idx], 0.5 * base), 3.0 * base))
  }

  dt
}

# -----------------------------------------------------------------------------
# 拟合
# -----------------------------------------------------------------------------
#' @param dt data.table，训练折的数据
#' @return list，含中位数、众数、和 9 个线性模型
fit_imputer_L3 <- function(dt) {

  med <- vapply(NUM_COLS_L3, function(cc) {
    stats::median(dt[[cc]], na.rm = TRUE)
  }, numeric(1))

  mode_of <- function(x) {
    x <- x[!is.na(x)]
    tb <- table(x)
    names(tb)[which.max(tb)]
  }
  mod <- vapply(CAT_COLS_L3, function(cc) mode_of(dt[[cc]]), character(1))

  # 用中位数填一份工作副本，作为拟合回归时的预测变量。
  # 这样每个模型都能用上全部 8 个预测变量，不必按缺失模式分情况讨论。
  work <- dt[, ..NUM_COLS_L3]
  for (cc in NUM_COLS_L3) {
    idx <- which(is.na(work[[cc]]))
    if (length(idx)) set(work, i = idx, j = cc, value = med[[cc]])
  }

  # 为每一列拟合一个线性模型：只用该列「观测到」的行来训练，
  # 因为只有这些行知道正确答案。
  models <- list()
  for (target in NUM_COLS_L3) {
    obs <- which(!is.na(dt[[target]]))
    preds <- setdiff(NUM_COLS_L3, target)

    train_df <- as.data.frame(work[obs, ..preds])
    train_df[[target]] <- dt[[target]][obs]

    fml <- stats::as.formula(paste(target, "~ ."))
    models[[target]] <- stats::lm(fml, data = train_df)
  }

  list(line = "L3", median = med, mode = mod, models = models)
}

# -----------------------------------------------------------------------------
# 应用
# -----------------------------------------------------------------------------
#' @param imp fit_imputer_L3 的返回值
#' @param dt data.table，要处理的数据（会被原地修改）
#' @return 插补后的 dt
apply_imputer_L3 <- function(imp, dt) {

  # 记住原始缺失位置。之后所有调整都只作用于这些格子。
  na_mask <- lapply(NUM_COLS_L3, function(cc) is.na(dt[[cc]]))
  names(na_mask) <- NUM_COLS_L3

  # --- 类别列：众数填补（和 L2 一样，这不是本条线的重点） -------------------
  for (cc in CAT_COLS_L3) {
    idx <- which(is.na(dt[[cc]]))
    if (length(idx)) {
      set(dt, i = idx, j = cc,
          value = factor(imp$mode[[cc]], levels = levels(dt[[cc]])))
    }
  }

  # --- 第 0 步：中位数初始化 ------------------------------------------------
  for (cc in NUM_COLS_L3) {
    idx <- which(na_mask[[cc]])
    if (length(idx)) set(dt, i = idx, j = cc, value = imp$median[[cc]])
  }

  # --- 迭代：回归预测 + 约束投影 --------------------------------------------
  # 收敛诊断：记录每轮插补值相对上一轮的平均绝对变化量。
  # 注释里写「2 轮已收敛」不能靠拍脑袋，要能量出来（审查意见 3.4）。
  delta_trace <- numeric(N_ITER_L3)

  for (iter in seq_len(N_ITER_L3)) {
    prev <- as.matrix(dt[, ..NUM_COLS_L3])

    for (target in NUM_COLS_L3) {
      idx <- which(na_mask[[target]])
      if (!length(idx)) next

      preds <- setdiff(NUM_COLS_L3, target)
      newdata <- as.data.frame(dt[idx, ..preds])
      pred <- stats::predict(imp$models[[target]], newdata = newdata)
      set(dt, i = idx, j = target, value = as.numeric(pred))
    }
    .project_constraints_L3(dt, na_mask)

    now <- as.matrix(dt[, ..NUM_COLS_L3])
    # 只统计被插补的格子，观测值本来就不变
    changed <- do.call(cbind, na_mask)
    sc <- apply(prev, 2, function(z) stats::sd(z, na.rm = TRUE))
    sc[!is.finite(sc) | sc == 0] <- 1
    rel <- abs(now - prev) / rep(sc, each = nrow(now))
    delta_trace[iter] <- if (any(changed)) mean(rel[changed]) else 0
  }

  # 收敛判据：最后一轮的相对变化 < 1% 个标准差
  attr(dt, "l3_delta_trace") <- delta_trace
  if (delta_trace[N_ITER_L3] > 0.01) {
    warning(sprintf(
      "L3 插补可能未收敛：第 %d 轮相对变化仍有 %.4f（阈值 0.01）。考虑调大 N_ITER_L3。",
      N_ITER_L3, delta_trace[N_ITER_L3]))
  }

  dt
}

# -----------------------------------------------------------------------------
# 诊断工具：约束到底触发了多少次
# -----------------------------------------------------------------------------
#' 统计约束投影在多少比例的行上真正改变了取值
#'
#' 实测结论是 0.06% —— 也就是说线性回归预测出来的值本来就几乎总是满足约束，
#' 「约束投影」这一层基本是装饰品。这个函数让这个结论可以随时复现，
#' 而不是靠一句注释。见项目说明第六节。
#'
#' @param imp fit_imputer_L3 的返回值
#' @param dt 待插补数据
#' @return list(n_rows, n_clamped, pct)
diagnose_constraints_L3 <- function(imp, dt) {
  d1 <- apply_imputer_L3(imp, copy(dt))

  # 关掉约束再跑一遍，比较差异
  proj_backup <- .project_constraints_L3
  assign(".project_constraints_L3", function(dt, na_mask) dt, envir = globalenv())
  d2 <- apply_imputer_L3(imp, copy(dt))
  assign(".project_constraints_L3", proj_backup, envir = globalenv())

  m1 <- as.matrix(d1[, ..NUM_COLS_L3])
  m2 <- as.matrix(d2[, ..NUM_COLS_L3])
  diff_cell <- abs(m1 - m2) > 1e-8
  n_clamped <- sum(rowSums(diff_cell, na.rm = TRUE) > 0)

  list(n_rows = nrow(dt), n_clamped = n_clamped,
       pct = 100 * n_clamped / nrow(dt))
}
