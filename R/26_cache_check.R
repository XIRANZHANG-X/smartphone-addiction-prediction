# =============================================================================
# 26_cache_check.R —— 验证插补缓存与「现算」逐位等价
#
# 用法：Rscript R/26_cache_check.R
#
# -----------------------------------------------------------------------------
# 为什么需要这个脚本
# -----------------------------------------------------------------------------
# 2026-09-02 给 prepare_fold() 加了插补缓存，动机是 L4（missRanger 的
# 链式随机森林 + PMM）在全量上每格要约 134 分钟，而四个算法跑的是**同一份**
# 插补——重复了四次。缓存后链式随机森林从 40 次降到 10 次，8.9 小时变 2.6 小时。
#
# 这个改动只有在「复用是等价而非近似」时才成立，而它依赖两件事：
#
#   1. L4 的随机种子由框架在折循环开头的 set.seed(SEED + k) 派生，
#      不依赖于用哪个算法建模 —— 所以四个算法应当拿到逐位相同的插补结果；
#   2. 建模前重设一次种子 —— 否则「读缓存」路径没消耗 missRanger 那部分
#      随机数，模型拿到的随机流会和「现算」路径不同。
#
# 两件事都不是显然的，所以在投入几小时算力之前先把它们测出来。
#
# 本脚本做三次 prepare_fold 并两两比对：
#   (1) 不用缓存
#   (2) 用缓存，冷启动（现算并写盘）
#   (3) 用缓存，热启动（读盘）
# 三者的插补结果必须完全一致，同时报告随机流的分歧以证明第 2 点的必要性。
# =============================================================================

suppressMessages({library(data.table)})
source("R/03_features.R")
source("R/05_impute_L4.R")

SEED  <- 20260821L
K     <- 1L                                   # 只测第一折，够了
CACHE <- file.path(tempdir(), "cache_check_L4.rds")
if (file.exists(CACHE)) unlink(CACHE)

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
train <- feat[is_train == 1L]
y_all <- train$addicted_label

# 用 5 万那一级，L4 在它上面几分钟就跑完，足以暴露问题
pool  <- readRDS("output/pools/pool_50k.rds")
X <- train[pool]; y <- y_all[pool]; f <- folds[pool]
tr <- which(f != K); va <- which(f == K)

cat(sprintf("池 %s 行 | 训练 %s | 验证 %s\n\n",
            format(nrow(X), big.mark = ","),
            format(length(tr), big.mark = ","),
            format(length(va), big.mark = ",")))

#' 跑一次 prepare_fold，返回结果以及**结束时的随机流状态**
one <- function(label, cache_file) {
  set.seed(SEED + K)
  t0 <- Sys.time()
  r  <- prepare_fold(X[tr], y[tr], X[va],
                     fit_imputer_L4, apply_imputer_L4,
                     use_derived = TRUE, use_te = TRUE,
                     cache_file = cache_file)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  # 结束时从随机流里抽一个数，用来看两条路径的随机流是否已经分家
  probe <- runif(1)
  cat(sprintf("  %-14s %5.2f 分钟   随机流探针 %.10f\n", label, mins, probe))
  list(tr = r$tr, va = r$va, probe = probe, mins = mins)
}

cat("三次运行：\n")
a <- one("不用缓存",   NULL)
b <- one("缓存·冷",    CACHE)
d <- one("缓存·热",    CACHE)

# ---- 比对 -------------------------------------------------------------------
cat("\n结果比对：\n")
chk <- function(nm, x, y) {
  same <- isTRUE(all.equal(x, y, check.attributes = FALSE, tolerance = 0))
  cat(sprintf("  %-28s %s\n", nm, if (same) "一致" else "★ 不一致"))
  same
}
ok <- c(
  chk("不用缓存 vs 缓存·冷（训练）", a$tr, b$tr),
  chk("不用缓存 vs 缓存·冷（验证）", a$va, b$va),
  chk("缓存·冷 vs 缓存·热（训练）",  b$tr, d$tr),
  chk("缓存·冷 vs 缓存·热（验证）",  b$va, d$va)
)

cat("\n随机流：\n")
cat(sprintf("  不用缓存 %.10f\n  缓存·冷 %.10f\n  缓存·热 %.10f\n",
            a$probe, b$probe, d$probe))
if (abs(a$probe - d$probe) > 1e-12) {
  cat("  → 热启动的随机流与现算**不同**（少消耗了 missRanger 那部分）。\n")
  cat("    这正是 06_framework.R 在 fit_predict 之前重设一次种子的原因：\n")
  cat("    重设之后模型拿到的随机流与走哪条路径无关。\n")
} else {
  cat("  → 随机流相同（意外，说明 missRanger 没有消耗全局随机流）。\n")
}

cat(sprintf("\n耗时：现算 %.2f 分钟，读缓存 %.2f 分钟，省 %.0f%%\n",
            b$mins, d$mins, 100 * (1 - d$mins / max(b$mins, 1e-9))))

if (!all(ok)) stop("★ 缓存与现算不一致，不能使用缓存") else
  cat("\n结论：缓存与现算逐位等价，可以用于 L4 的四格复用。\n")
