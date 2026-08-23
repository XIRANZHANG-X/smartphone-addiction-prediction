# =============================================================================
# 08_submit.R —— 生成 Kaggle 提交文件
#
# 用法：source("R/08_submit.R")
# 产出：submissions/submission_<日期>_<方式>.csv
#
# 默认用 07_ensemble.R 选出的最优集成。想提交某个单模型，
# 把 USE_SINGLE 改成那个模型名即可（调试或对照时有用）。
# =============================================================================

library(data.table)

# 设成模型名（如 "L3_xgboost"）可直接提交该单模型；设成 NULL 用集成结果
USE_SINGLE <- NULL

dir_sub <- "submissions"
dir.create(dir_sub, showWarnings = FALSE, recursive = TRUE)

test <- readRDS("output/raw_test.rds")

# ---- 取预测 -----------------------------------------------------------------
if (is.null(USE_SINGLE)) {
  f <- "output/ensemble_best.rds"
  if (!file.exists(f)) {
    stop("找不到 output/ensemble_best.rds，请先运行 source(\"R/07_ensemble.R\")")
  }
  ens    <- readRDS(f)
  pred   <- ens$test_pred
  tag    <- ens$method
  cv_auc <- ens$cv_auc
  cat(sprintf("使用集成结果：%s（CV %.5f，含 %d 个模型）\n",
              tag, cv_auc, length(ens$models)))
} else {
  f <- sprintf("output/test/test_%s.rds", USE_SINGLE)
  if (!file.exists(f)) stop("找不到 ", f)
  pred   <- readRDS(f)
  tag    <- USE_SINGLE
  cv_auc <- NA_real_
  cat(sprintf("使用单模型：%s\n", tag))
}

# ---- 校验 -------------------------------------------------------------------
# 提交文件出错是最冤的失分方式，所以每一条都检查。
stopifnot(
  "预测长度与测试集不一致" = length(pred) == nrow(test),
  "预测里有 NA"            = !anyNA(pred),
  "预测里有非有限值"       = all(is.finite(pred))
)

# ---- 归一化到 (0, 1) --------------------------------------------------------
# AUC 只看排序，理论上不需要归一化。但 rank 平均和爬山法输出的是秩，
# 数值上是 1..n 而不是概率。Kaggle 一般能接受，
# 保险起见统一压回 (0,1) 区间，同时也让文件更像「概率」。
rng <- range(pred)
if (rng[2] - rng[1] > 0) {
  pred <- (pred - rng[1]) / (rng[2] - rng[1])
}
# 压到开区间，避开 0 和 1 这两个端点
pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)

# ---- 写文件 -----------------------------------------------------------------
out <- data.table(id = test$id, addicted_label = pred)

stamp <- format(Sys.time(), "%Y%m%d_%H%M")
fname <- file.path(dir_sub, sprintf("submission_%s_%s.csv", stamp, tag))
fwrite(out, fname)

# ---- 报告 -------------------------------------------------------------------
cat(sprintf("\n已写出 %s\n", fname))
cat(sprintf("  行数      %s\n", format(nrow(out), big.mark = ",")))
cat(sprintf("  预测均值  %.4f（训练集正例率 0.7094）\n", mean(pred)))
cat(sprintf("  预测范围  [%.4f, %.4f]\n", min(pred), max(pred)))

cat("\n提交之后，请把 Public LB 分数填回 submissions/log.csv 对应行的 lb_public 列。\n")
cat("CV 和 LB 的对应关系是我们判断本地验证是否可信的唯一依据 —— 别偷懒不记。\n")

# 追加一行到台账
log_f <- file.path(dir_sub, "log.csv")
if (file.exists(log_f)) {
  # colClasses = "character"：fread 会把 date 列自动识别成 IDate，
  # 和我们要追加的字符型日期对不上，rbindlist 会把它填成空值 ——
  # 结果台账里新记录没有日期，而且不报错。全部按字符读最省事。
  log_dt <- fread(log_f, colClasses = "character")
  log_dt <- rbindlist(list(log_dt, data.table(
    date            = as.character(Sys.Date()),
    submission_file = basename(fname),
    method          = tag,
    cv_auc          = if (is.na(cv_auc)) "" else sprintf("%.5f", cv_auc),
    cv_sd           = "",
    lb_public       = "",
    notes           = ""
  )), use.names = TRUE, fill = TRUE)
  fwrite(log_dt, log_f)
  cat(sprintf("已在 %s 追加一行。\n", log_f))
}
