# =============================================================================
# 05_impute_L2.R —— 第二条线：中位数 / 众数填补
#
# 负责人：B
#
# 最经典的填补方法：数值列填中位数，类别列填出现最多的那个值。
#
# 这条线的作用是当「对照组」。它几乎肯定不是最好的，但它是所有教科书上
# 的默认做法，所以必须跑，用来量化「更聪明的插补到底值多少 AUC」。
#
# 中位数必须在折内计算 —— 用全部训练数据算中位数再套到验证折上，
# 属于轻微但真实的泄漏。框架区已经保证了这一点，这里只要正确实现
# fit / apply 两个函数即可。
# =============================================================================

NUM_COLS_L2 <- c(
  "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
  "work_study_hours", "sleep_hours", "notifications_per_day",
  "app_opens_per_day", "weekend_screen_time"
)

CAT_COLS_L2 <- c("gender", "stress_level", "academic_work_impact")

#' 拟合插补器：记下每一列的中位数 / 众数
#'
#' @param dt data.table，训练折的数据
#' @return list，含每列的填补值
fit_imputer_L2 <- function(dt) {
  med <- vapply(NUM_COLS_L2, function(cc) {
    stats::median(dt[[cc]], na.rm = TRUE)
  }, numeric(1))

  mode_of <- function(x) {
    x <- x[!is.na(x)]
    tb <- table(x)
    names(tb)[which.max(tb)]
  }
  mod <- vapply(CAT_COLS_L2, function(cc) mode_of(dt[[cc]]), character(1))

  list(line = "L2", median = med, mode = mod)
}

#' 应用插补器
#'
#' @param imp fit_imputer_L2 的返回值
#' @param dt data.table，要处理的数据（会被原地修改）
#' @return 填补后的 dt
apply_imputer_L2 <- function(imp, dt) {
  for (cc in NUM_COLS_L2) {
    idx <- which(is.na(dt[[cc]]))
    if (length(idx)) set(dt, i = idx, j = cc, value = imp$median[[cc]])
  }
  for (cc in CAT_COLS_L2) {
    idx <- which(is.na(dt[[cc]]))
    # 保持 factor 类型不变：用 levels 里的对应值赋回去，不要退化成字符串
    if (length(idx)) {
      set(dt, i = idx, j = cc,
          value = factor(imp$mode[[cc]], levels = levels(dt[[cc]])))
    }
  }
  dt
}
