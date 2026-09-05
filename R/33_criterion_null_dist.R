# =============================================================================
# 33_criterion_null_dist.R —— 判据"5/5 折同号 + Cohen's d > 2"的经验假阳性率
#
# 用法：Rscript R/33_criterion_null_dist.R
# 产出：output/criterion_null_dist.rds + 屏幕/日志报告
#
# -----------------------------------------------------------------------------
# 背景（审稿意见，第 3 项任务）
# -----------------------------------------------------------------------------
# §3.3 用"5 折里 5 折同号 + Cohen's d > 2"作为判定一个效应"是真的"的标准，
# 而不是相信 n=5 的 p 值。R/18_new_features.R 已经报过一次安慰剂测量
# （随机噪声列，缺失模式与真实列一致）：+0.00003 AUC、Cohen's d 0.11、
# 2/5 折同号——显然没有触发判据。但那只是**一次**随机抽样。
#
# 审稿人的意思是：一次抽样说明不了这个判据在纯噪声上到底多容易"误触发"，
# 要看很多次独立抽样才能估出一个真正的假阳性率。本脚本就是把
# "加一列随机噪声，看是否触发判据"这个测量，用不同种子重复很多次，
# 报告触发比例。
#
# -----------------------------------------------------------------------------
# 复用策略：不重新发明 run_cfg / add_placebo，也不能整份 source
# -----------------------------------------------------------------------------
# R/18_new_features.R 在定义完 add_placebo()（约 129 行）与 run_cfg()
# （约 140 行）之后，**在顶层**（不在函数里）立刻开始用它们做一整轮测量
# （R$base <- run_cfg("base")、R$maxbin <- run_cfg(...) ……一直到
# saveRDS(R, "output/new_features.rds")）。如果直接
# source("R/18_new_features.R")，这一整轮会被重新跑一遍——不仅浪费
# 几十分钟，还会覆盖已经存在的 output/new_features.rds。
#
# 这里改用"解析成顶层表达式列表，逐条 eval，一旦刚跑完的是
# `run_cfg <- function(...)` 这条赋值语句就停"的办法：不管文件其余部分
# 长什么样，我们保证在第一条会真正触发模型拟合的语句（第一个
# `R$base <- run_cfg(...)` 调用）执行之前就已经跳出循环。
# add_placebo 的定义在 run_cfg 之前，所以在同一趟里一并拿到。
# =============================================================================

suppressMessages({library(data.table)})

# 目标重复次数。历史上（output/logs/18_new_features.log）同一条 run_cfg
# 在这台机器上一次 5 折大约 2.2 分钟；30 次重复 + 1 次 base 预计
# 70 分钟量级，在"合理时间"内。脚本会在每次重复后打印累计用时与外推的
# 总预计用时，如果实际比这慢很多，可以据此提前终止进程——已完成的重复
# 已经逐次落盘，不会丢失。
N_TARGET <- 30L
OUT_RDS  <- "output/criterion_null_dist.rds"

t_script0 <- Sys.time()
ts <- function() format(Sys.time(), "%H:%M:%S")

# -----------------------------------------------------------------------------
# 从 R/18_new_features.R 里只取 run_cfg / add_placebo，不触发它自身的测量
# -----------------------------------------------------------------------------
is_top_assign <- function(e, nm) {
  is.call(e) && length(e) >= 3 &&
    (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
    is.name(e[[2]]) && identical(as.character(e[[2]]), nm)
}

setup_env <- new.env()
src_exprs <- parse("R/18_new_features.R")
got_run_cfg <- FALSE
for (e in src_exprs) {
  eval(e, envir = setup_env)
  if (is_top_assign(e, "run_cfg")) { got_run_cfg <- TRUE; break }
}
if (!got_run_cfg) {
  stop("R/18_new_features.R 里没找到顶层语句 'run_cfg <- function(...)'。",
       "文件结构可能变了，本脚本复用 run_cfg/add_placebo 的方式需要重新核实，",
       "不能在不确定的情况下继续（可能会漏跑，也可能会意外触发原脚本自己的",
       "整轮测量）。")
}
stopifnot(is.function(setup_env$run_cfg), is.function(setup_env$add_placebo),
          is.numeric(setup_env$f_pool), is.numeric(setup_env$y_pool))

run_cfg     <- setup_env$run_cfg
add_placebo <- setup_env$add_placebo

cat(sprintf("[%s] 已从 R/18_new_features.R 取得 run_cfg/add_placebo；",
            ts()))
cat(sprintf("f_pool 长度 %d，未触发该文件自身的顶层测量。\n",
            length(setup_env$f_pool)))

# -----------------------------------------------------------------------------
# 判据的计算方式：与 §3.3 / docs/实验报告.md §2.3 的定义、以及
# R/18_new_features.R 的配对报告代码逐位一致，已用已存的
# output/new_features.rds 数值核对过（base vs placebo 复算得到
# +0.00003、d=0.11、2/5，与文档一致）：
#   - 差值 d = arm_auc - base_auc（同一折，逐折相减）；
#   - "同号数" = sum(d > 0)：这是有方向的——本判据只用来判定"新加的这一列
#     是不是一个真实的、consistent 的正向提升"，不是"5 折互相之间是否
#     同号"这种对称统计量。R/18_new_features.R 的配对报告代码里也是
#     写死的 sum(d > 0)，不是 max(sum(d>0), sum(d<0))；
#   - Cohen's d = mean(d) / sd(d)（有符号；当 sign_agree==5 时全部折
#     都是正的，mean(d) 必为正，所以 "d > 2" 与 "|d| > 2" 在这个判据下
#     其实等价——这里仍写 abs() 以贴合 dispatch 里 "|d| > 2" 的字面表述，
#     不影响任何一次判定结果）；
#   - 触发 = (sign_agree == 5) & (abs(cohens_d) > 2)。
eval_repeat <- function(arm_auc, base_auc) {
  d <- arm_auc - base_auc
  sign_agree <- sum(d > 0)
  cohens_d <- mean(d) / sd(d)
  list(mean_diff = mean(d), sd_diff = sd(d), cohens_d = cohens_d,
       sign_agree = sign_agree,
       triggered = (sign_agree == 5L) && (abs(cohens_d) > 2))
}

# -----------------------------------------------------------------------------
# 基准臂：不加任何列，只拟合一次，所有重复共用同一份折叠级 AUC
# -----------------------------------------------------------------------------
# 关键的成本优化：base 不依赖安慰剂种子，所以只需要跑一次 5 折，把折级 AUC
# 存下来，后面 N_TARGET 次重复只需要重新跑"base + placebo(seed_i)"这一臂，
# 相比"每次重复都重跑 base+arm 两臂"省下接近一半的算力。
cat(sprintf("[%s] 开始拟合 base（5 折，不加安慰剂列）……\n", ts()))
t0 <- Sys.time()
base_res <- run_cfg("base_for_null_dist")
base_auc <- base_res$auc
t_base_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("[%s] base 完成，用时 %.2f 分钟，%d 个特征。折级 AUC：%s\n",
            ts(), t_base_min, base_res$n_feat,
            paste(sprintf("%.5f", base_auc), collapse = ", ")))

# -----------------------------------------------------------------------------
# 重复 N_TARGET 次：base + placebo(seed_i)，每次只重新拟合这一臂
# -----------------------------------------------------------------------------
# 种子：训练部分用 1000+i，验证部分用 2000+i —— 与 add_placebo 在
# R/18_new_features.R 里已经用过的 1L/2L 不冲突，写法上也和它保持一致
# （build() 内对 a 用一个种子、对 b 用另一个种子）。
results <- vector("list", N_TARGET)
repeat_mins <- numeric(0)

save_progress <- function(done_upto) {
  saveRDS(list(base_auc = base_auc, base_minutes = t_base_min,
               n_completed = done_upto, n_target = N_TARGET,
               results = results[seq_len(done_upto)]),
          OUT_RDS)
}

cat(sprintf("[%s] 开始 %d 次重复（每次：base + placebo(seed_i)，5 折）……\n",
            ts(), N_TARGET))

for (i in seq_len(N_TARGET)) {
  seed_tr <- 1000L + i
  seed_va <- 2000L + i
  build_i <- local({
    s_tr <- seed_tr; s_va <- seed_va
    function(a, b, y) list(add_placebo(a, s_tr), add_placebo(b, s_va))
  })

  t0 <- Sys.time()
  arm_res <- run_cfg(sprintf("null_%02d", i), build_i)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  repeat_mins <- c(repeat_mins, mins)

  st <- eval_repeat(arm_res$auc, base_auc)
  results[[i]] <- list(i = i, seed_tr = seed_tr, seed_va = seed_va,
                        arm_auc = arm_res$auc, mean_diff = st$mean_diff,
                        sd_diff = st$sd_diff, cohens_d = st$cohens_d,
                        sign_agree = st$sign_agree, triggered = st$triggered,
                        minutes = mins)

  elapsed_total <- as.numeric(difftime(Sys.time(), t_script0, units = "mins"))
  avg_so_far <- mean(repeat_mins)
  projected_total <- t_base_min + N_TARGET * avg_so_far
  cat(sprintf(
    "[%s] 重复 %2d/%d  用时 %.2f 分钟  mean_diff %+.5f  d %+.3f  同号 %d/5  触发=%s | 累计 %.1f 分钟，逐次均值 %.2f 分钟，跑满 %d 次预计共 %.1f 分钟\n",
    ts(), i, N_TARGET, mins, st$mean_diff, st$cohens_d, st$sign_agree,
    st$triggered, elapsed_total, avg_so_far, N_TARGET, projected_total))

  save_progress(i)  # 每次重复后立即落盘，中途终止也不丢已完成的结果
}

total_min <- as.numeric(difftime(Sys.time(), t_script0, units = "mins"))
cat(sprintf("\n[%s] 全部 %d 次重复完成，总用时 %.1f 分钟（含 base）。\n",
            ts(), N_TARGET, total_min))

# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
triggered_vec <- vapply(results, function(r) r$triggered, logical(1))
d_vec         <- vapply(results, function(r) r$cohens_d, numeric(1))
sign_vec      <- vapply(results, function(r) r$sign_agree, integer(1))
n_triggered   <- sum(triggered_vec)
fpr           <- n_triggered / N_TARGET

cat("\n", strrep("=", 74), "\n 判据的经验假阳性率\n", strrep("=", 74), "\n", sep = "")
cat(sprintf("N = %d 次独立重复（每次都是一次新的随机噪声列 vs 同一份 base）\n", N_TARGET))
cat(sprintf("触发判据（5/5 同号 & |d| > 2）的次数：%d / %d\n", n_triggered, N_TARGET))
cat(sprintf("经验假阳性率：%.1f%%\n", 100 * fpr))
cat(sprintf("|Cohen's d| 分布：均值 %.3f，中位数 %.3f，范围 [%.3f, %.3f]\n",
            mean(abs(d_vec)), median(abs(d_vec)), min(abs(d_vec)), max(abs(d_vec))))
cat(sprintf("同号数（out of 5）分布：\n"))
print(table(sign_vec))

saveRDS(list(base_auc = base_auc, base_minutes = t_base_min,
             n_completed = N_TARGET, n_target = N_TARGET,
             results = results, n_triggered = n_triggered, fpr = fpr,
             total_minutes = total_min),
        OUT_RDS)
cat(sprintf("\n已保存 %s\n", OUT_RDS))
