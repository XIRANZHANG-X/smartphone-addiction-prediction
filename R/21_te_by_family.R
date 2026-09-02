# =============================================================================
# 21_te_by_family.R —— target encoding 在四个模型族上的效果
#
# 用法：Rscript R/21_te_by_family.R
# 产出：output/te_by_family.rds
#
# -----------------------------------------------------------------------------
# 为什么单独测这个
# -----------------------------------------------------------------------------
# 本项目的核心发现是：**预处理的价值取决于下游模型自己能表示什么**。
#   - 插补：能原生表示「未知」的模型（GBDT）不需要好的猜测，
#           不能表示的（glmnet/ranger）才需要（发现 9.1）
#   - 派生特征：能自己造出比值的模型不需要你给，不能造的才需要
#           （17 vs 12 特征那次，glmnet −0.00357 而 GBDT ≈ 0）
#
# 逐取值 target encoding 是同一个问题的第三个实例，而且方向应当最极端：
# 一个线性模型无法在**原始尺度**上表示「取值 5.23 对应概率 0.71」这种非单调查找：
# 那需要一个非单调函数，而它只有一个斜率。树至少能用上千次分裂去逼近。
#
# ⚠ 事后更正（见 R/24_onehot_lr.R）：这句话的早期版本写的是「完全无法……
#   无论怎么调系数都造不出来」，那是**说过头了**。给线性模型每个取值
#   一个自由参数（one-hot）它就能做到 —— 全量上 0.95929，高于 TE 的 0.94805。
#   做不到的不是「查找」，是「**在不增加参数的前提下**查找」。
#
# 所以预期：TE 对 glmnet 的收益 >> 对 GBDT 的收益。
# 如果测出来是这样，这就是同一条结论的第三次独立确认。
#
# 口径与 R/06_framework.R 一致；每个族配它能用的最好插补线。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

SEED <- 20260821L
feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

# 四个族 + 它们能用的插补线（ranger/glmnet 不支持原生 NaN）
FAM <- list(
  xgboost  = list(line = "L1", fac = function() make_xgb()),
  lightgbm = list(line = "L1", fac = function() make_lgb()),
  ranger   = list(line = "L2", fac = function() make_ranger()),
  glmnet   = list(line = "L2", fac = function() make_glmnet())
)

run <- function(fam, use_te) {
  cfg <- FAM[[fam]]
  source(sprintf("R/05_impute_%s.R", cfg$line), local = FALSE)
  fit_i   <- get(paste0("fit_imputer_",   cfg$line))
  apply_i <- get(paste0("apply_imputer_", cfg$line))
  fp <- cfg$fac()

  aucs <- numeric(0)
  t0 <- Sys.time()
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    imp  <- fit_i(X_pool[tr])
    X_tr <- derive_features(apply_i(imp, copy(X_pool[tr])))
    X_va <- derive_features(apply_i(imp, copy(X_pool[va])))
    if (use_te) {
      e <- fit_target_encoder(X_tr, y_pool[tr])
      X_tr <- apply_target_encoder(e, X_tr)
      X_va <- apply_target_encoder(e, X_va)
    }
    p <- fp(X_tr, y_pool[tr], X_va)
    aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(y_pool[va], as.numeric(p),
                                                   quiet = TRUE))))
  }
  cat(sprintf("  %-9s %s  TE %s  AUC %.5f ± %.5f  (%.1f 分钟)\n",
              fam, cfg$line, if (use_te) "开" else "关",
              mean(aucs), sd(aucs),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  aucs
}

cat(sprintf("Tier A：%s 行，冻结 5 折\n\n", format(nrow(X_pool), big.mark = ",")))
R <- list()
for (fam in names(FAM)) {
  R[[fam]] <- list(off = run(fam, FALSE), on = run(fam, TRUE))
}

cat("\n", strrep("=", 76), "\n 逐取值 target encoding 的收益，按模型族\n",
    strrep("=", 76), "\n", sep = "")
cat(sprintf("%-10s %-6s %11s %11s %11s %8s %7s\n",
            "算法", "插补", "TE 关", "TE 开", "差值", "Cohen d", "同号"))
for (fam in names(R)) {
  d <- R[[fam]]$on - R[[fam]]$off
  cat(sprintf("%-10s %-6s %11.5f %11.5f %+11.5f %8.2f %5d/5\n",
              fam, FAM[[fam]]$line, mean(R[[fam]]$off), mean(R[[fam]]$on),
              mean(d), mean(d) / sd(d), sum(d > 0)))
}

cat("\n预期：线性模型在原始尺度上表示不了非单调查找，收益应当最大；\n")
cat("GBDT 能用上千次分裂逼近，收益应当最小。\n")

saveRDS(R, "output/te_by_family.rds")
cat("\n已保存 output/te_by_family.rds\n")
