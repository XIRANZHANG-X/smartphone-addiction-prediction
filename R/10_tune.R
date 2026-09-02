# =============================================================================
# 10_tune.R —— 超参数搜索（审查意见 1.2）
#
# 用法：Rscript R/10_tune.R [xgboost|lightgbm]
# 产出：output/tune_<algo>.rds、控制台报表
#
# -----------------------------------------------------------------------------
# 为什么必须做这件事
# -----------------------------------------------------------------------------
# 项目此前的全部结论都建立在一组从未搜索过的参数上。这有两个问题：
#
#   1. 分数留在桌上。早停一项就带来 +0.0018，超过四条插补线全部差距的一半，
#      说明这类「基础功课」的收益远大于我们花大力气做的插补研究。
#
#   2. 更要命的是结论的稳健性。「L1 > L2 > L3 > L4」只有在**所有插补线
#      对超参数同等敏感**时才成立。如果 L3 在另一组参数下反超，
#      当前结论就不成立。这不是杞人忧天 —— 插补后的数据没有 NA，
#      树的分裂行为完全不同，最优深度本来就可能不一样。
#
# 因此本脚本做两件事：
#   A. 在 L1 上做两阶段粗搜，找到一组像样的参数
#   B. 用找到的参数**重跑 L1/L2/L3/L4 的排序**，检验结论是否翻转
#
# -----------------------------------------------------------------------------
# 算力预算
# -----------------------------------------------------------------------------
# 搜索阶段用 3 折而不是 5 折，20 万行子样本。搜索的目的是**排序候选参数**，
# 不是给出可发表的 AUC 估计，3 折足够且省 40% 时间。
# 选出的参数会用完整 5 折重跑一遍，那个数字才进对比表。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

args  <- commandArgs(trailingOnly = TRUE)
ALGO  <- if (length(args)) args[1] else "xgboost"
SEED  <- 20260821L
N_FOLD_SEARCH <- 3L

stopifnot(ALGO %in% c("xgboost", "lightgbm"))

# ---- 数据 -------------------------------------------------------------------
feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

source("R/05_impute_L1.R")

#' 用给定参数跑 N_FOLD_SEARCH 折，返回平均 AUC
#' @param line 插补线，默认 L1
eval_params <- function(params, line = "L1", n_fold = N_FOLD_SEARCH) {
  source(sprintf("R/05_impute_%s.R", line), local = FALSE)
  fit_i   <- get(paste0("fit_imputer_",   line))
  apply_i <- get(paste0("apply_imputer_", line))

  fp <- if (ALGO == "xgboost") make_xgb(params) else make_lgb(params)

  a <- numeric(0); iters <- integer(0)
  for (k in seq_len(n_fold)) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    fold <- prepare_fold(X_pool[tr], y_pool[tr], X_pool[va], fit_i, apply_i)
    X_tr <- fold$tr; X_va <- fold$va
    p <- fp(X_tr, y_pool[tr], X_va)
    bi <- attr(p, "best_iteration"); if (!is.null(bi)) iters <- c(iters, bi)
    a <- c(a, as.numeric(auc(roc(y_pool[va], as.numeric(p), quiet = TRUE))))
  }
  list(auc = mean(a), sd = sd(a), iters = if (length(iters)) mean(iters) else NA)
}

run_stage <- function(label, cfgs) {
  cat(sprintf("\n===== %s（%d 组）=====\n", label, length(cfgs)))
  res <- list()
  for (i in seq_along(cfgs)) {
    nm <- names(cfgs)[i]
    cat(sprintf("  [%2d/%2d] %-42s ... ", i, length(cfgs), nm)); flush.console()
    t0 <- Sys.time()
    r <- eval_params(cfgs[[i]])
    mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    cat(sprintf("AUC %.5f  (%.0f 轮, %.1f 分钟)\n", r$auc, r$iters, mins))
    res[[nm]] <- c(r, list(params = cfgs[[i]]))
  }
  res
}

# ---- 阶段一：学习率 × 树深 --------------------------------------------------
# 这两个是梯度提升里交互最强、影响最大的一对。
stage1 <- list()
for (eta in c(0.03, 0.05, 0.10)) {
  for (md in c(4L, 6L, 8L)) {
    key <- sprintf("eta=%.2f depth=%d", eta, md)
    stage1[[key]] <- if (ALGO == "xgboost") {
      list(eta = eta, max_depth = md)
    } else {
      list(learning_rate = eta, num_leaves = as.integer(2^md - 1))
    }
  }
}
r1 <- run_stage("阶段一：学习率 × 树深", stage1)

best1_name <- names(r1)[which.max(vapply(r1, function(z) z$auc, 0))]
best1 <- r1[[best1_name]]$params
cat(sprintf("\n阶段一最优：%s  AUC %.5f\n", best1_name, max(vapply(r1, function(z) z$auc, 0))))

# ---- 阶段二：在最优点附近调正则化 -------------------------------------------
stage2 <- list()
if (ALGO == "xgboost") {
  for (mcw in c(1L, 10L, 50L)) {
    for (cs in c(0.6, 0.8, 1.0)) {
      key <- sprintf("%s mcw=%d colsample=%.1f", best1_name, mcw, cs)
      stage2[[key]] <- utils::modifyList(best1,
                        list(min_child_weight = mcw, colsample_bytree = cs))
    }
  }
} else {
  for (mdl in c(20L, 50L, 200L)) {
    for (ff in c(0.6, 0.8, 1.0)) {
      key <- sprintf("%s min_data=%d feat_frac=%.1f", best1_name, mdl, ff)
      stage2[[key]] <- utils::modifyList(best1,
                        list(min_data_in_leaf = mdl, feature_fraction = ff))
    }
  }
}
r2 <- run_stage("阶段二：正则化", stage2)

all_res <- c(r1, r2)
scores  <- vapply(all_res, function(z) z$auc, 0)
best_name <- names(scores)[which.max(scores)]
best_par  <- all_res[[best_name]]$params

# ---- 报表 -------------------------------------------------------------------
cat("\n======================================================\n")
cat("  搜索结果（按 AUC 排序，前 10）\n")
cat("======================================================\n")
ord <- order(-scores)
for (i in head(ord, 10)) {
  cat(sprintf("  %.5f  %s\n", scores[i], names(scores)[i]))
}
cat(sprintf("\n默认参数 AUC : %.5f\n",
            scores[grep("eta=0.05 depth=6$", names(scores))[1]]))
cat(sprintf("最优参数 AUC : %.5f  (%s)\n", max(scores), best_name))
cat(sprintf("调参增益     : %+.5f\n",
            max(scores) - scores[grep("eta=0.05 depth=6$", names(scores))[1]]))

# ---- 稳健性检查：最优参数下，四条插补线的排序是否翻转？----------------------
cat("\n======================================================\n")
cat("  稳健性检查：用最优参数重跑四条插补线\n")
cat("  （若排序与默认参数下不同，说明原结论依赖于参数选择）\n")
cat("======================================================\n")
robust <- list()
for (line in c("L1", "L2", "L3", "L4")) {
  cat(sprintf("  %s ... ", line)); flush.console()
  r <- eval_params(best_par, line = line)
  robust[[line]] <- r$auc
  cat(sprintf("AUC %.5f\n", r$auc))
}
cat("\n最优参数下的排序：",
    paste(names(sort(unlist(robust), decreasing = TRUE)), collapse = " > "), "\n")

saveRDS(list(algo = ALGO, all = all_res, best_name = best_name,
             best_params = best_par, robustness = robust),
        sprintf("output/tune_%s.rds", ALGO))
cat(sprintf("\n已保存 output/tune_%s.rds\n", ALGO))
