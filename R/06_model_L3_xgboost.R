# =============================================================================
# 06_model_TEMPLATE.R —— 建模模板
#
# 【怎么用】
#   1. 复制本文件，改名为 R/06_model_<你的线号>_<模型名>.R
#      例如 C 做约束插补 + xgboost，就叫 06_model_L3_xgboost.R
#   2. 只改「配置区」和「模型区」两块（下面有醒目标记）
#   3. 「框架区」一个字都不要动 —— 它保证了你的分数和别人可比
#   4. 跑完检查 output/oof/ 下有没有生成你的文件，然后 push
#
# 【为什么框架区不能动】
#   插补模型必须在每一折内部拟合。如果在全部训练数据上拟合一次再套用，
#   验证集的信息会通过插补参数渗进训练，CV 分数会虚高。
#
#   更麻烦的是：虚高的幅度随插补方法的复杂度递增。也就是说，
#   越复杂的插补方法虚高越多 —— 而我们主推的正是最复杂的 L3、L4。
#   如果不控制这一点，整个对比实验会系统性地偏袒我们自己的方法，
#   最后得出「约束插补大胜」的假结论，而这个结论经不起任何人复查。
#
#   框架区已经把正确的顺序写死了，照抄就不会错。
# =============================================================================

library(data.table)
library(pROC)

# =============================================================================
#  一、配置区 —— 改这里
# =============================================================================

# 你的模型名。会成为输出文件名的一部分，用英文和下划线，不要空格。
MODEL_NAME <- "L3_xgboost"

# 跑哪一层？
#   "A" = Tier A 对比实验，20 万行子样本，用于报告里的方法对比表
#   "B" = Tier B 集成候选，全量 69 万行，只有 Tier A 里排名靠前的才跑这个
# 先跑 A，等组长通知了再跑 B。
TIER <- "A"

# 你这条线用哪套插补？取值对应 R/05_impute_L*.R 里定义的函数。
#   L1 = 不插补（保留 NA，交给 xgboost/lightgbm 原生处理）
#   L2 = 中位数/众数填补
#   L3 = 约束条件插补
#   L4 = missRanger 多重插补
IMPUTE_LINE <- "L3"

# 随机种子。不要改，改了结果就没法和别人对齐。
SEED <- 20260821L

# =============================================================================
#  二、模型区 —— 改这里
# =============================================================================
#  这个函数必须定义在框架区之前，因为框架区的循环会调用它。
# =============================================================================

#' 训练模型并预测
#'
#' @param X_tr data.table，训练特征（已插补、已算派生特征）
#' @param y_tr integer 向量，0/1 标签
#' @param X_va data.table，要预测的特征，列与 X_tr 完全一致
#' @return numeric 向量，长度等于 nrow(X_va)，取值是「成瘾」的概率
#'
fit_predict <- function(X_tr, y_tr, X_va) {

  # --- 选特征 ---------------------------------------------------------------
  # 排除非特征列。想做消融就在这里加减。
  drop_cols <- c("id", "addicted_label", "is_train")
  use_cols  <- setdiff(names(X_tr), drop_cols)

  # xgboost 只吃数值矩阵，类别列要转成整数编码。
  # train 和 test 的 factor levels 在 01_load.R 里已经统一过，所以安全。
  to_matrix <- function(dt) {
    m <- dt[, ..use_cols]
    for (cc in names(m)) {
      if (is.factor(m[[cc]])) set(m, j = cc, value = as.integer(m[[cc]]))
    }
    as.matrix(m)
  }

  # --- 训练 -----------------------------------------------------------------
  dtrain <- xgboost::xgb.DMatrix(to_matrix(X_tr), label = y_tr)

  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,
    tree_method      = "hist",
    nthread          = parallel::detectCores()
  )

  model <- xgboost::xgb.train(params, dtrain, nrounds = 600, verbose = 0)

  # --- 预测 -----------------------------------------------------------------
  predict(model, xgboost::xgb.DMatrix(to_matrix(X_va)))
}

# =============================================================================
#  三、框架区 —— 不要动
# =============================================================================

source("R/03_features.R")                              # 拿 derive_features()
source(sprintf("R/05_impute_%s.R", IMPUTE_LINE))       # 拿插补函数

fit_imputer   <- get(paste0("fit_imputer_",   IMPUTE_LINE))
apply_imputer <- get(paste0("apply_imputer_", IMPUTE_LINE))

# ---- 载入共享产物 -----------------------------------------------------------
feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")

train_all <- feat[is_train == 1L]
test_all  <- feat[is_train == 0L]
y_all     <- train_all$addicted_label

stopifnot(
  "features_raw 的训练行数与 folds 长度不一致" = nrow(train_all) == length(folds)
)

# ---- 按 Tier 选择行 ---------------------------------------------------------
if (TIER == "A") {
  row_idx <- readRDS("output/subsample_200k.rds")
} else if (TIER == "B") {
  row_idx <- seq_len(nrow(train_all))
} else {
  stop("TIER 只能是 \"A\" 或 \"B\"")
}

X_pool <- train_all[row_idx]
y_pool <- y_all[row_idx]
f_pool <- folds[row_idx]        # 关键：子样本继承原折标签，不重新划分

cat(sprintf("模型 %s | Tier %s | %s 行 | 插补 %s\n",
            MODEL_NAME, TIER, format(nrow(X_pool), big.mark = ","), IMPUTE_LINE))

# ---- 交叉验证主循环 ---------------------------------------------------------
# 每一折的顺序是固定的，这个顺序就是防泄漏的全部秘密：
#   1. 只在训练折上拟合插补器
#   2. 用它变换训练折和验证折
#   3. 插补之后才算派生特征（比值/差值特征在插补前算全是 NA）
#   4. 训练模型，预测验证折
oof <- rep(NA_real_, nrow(X_pool))
fold_auc <- numeric(0)

for (k in sort(unique(f_pool))) {
  cat(sprintf("  第 %d 折 ... ", k))
  set.seed(SEED + k)

  tr <- which(f_pool != k)
  va <- which(f_pool == k)

  # 1. 只用训练折拟合插补器
  imp <- fit_imputer(X_pool[tr])

  # 2. 变换两边
  X_tr <- apply_imputer(imp, copy(X_pool[tr]))
  X_va <- apply_imputer(imp, copy(X_pool[va]))

  # 3. 插补之后才算派生特征
  X_tr <- derive_features(X_tr)
  X_va <- derive_features(X_va)

  # 4. 训练 + 预测
  pred <- fit_predict(X_tr, y_pool[tr], X_va)

  oof[va] <- pred
  a <- as.numeric(pROC::auc(pROC::roc(y_pool[va], pred, quiet = TRUE)))
  fold_auc <- c(fold_auc, a)
  cat(sprintf("AUC %.5f\n", a))
}

cv_mean <- mean(fold_auc)
cv_sd   <- sd(fold_auc)
oof_auc <- as.numeric(pROC::auc(pROC::roc(y_pool, oof, quiet = TRUE)))

cat(sprintf("\n跨折 AUC  %.5f ± %.5f\n", cv_mean, cv_sd))
cat(sprintf("整体 OOF  %.5f\n", oof_auc))

# ---- 存盘 -------------------------------------------------------------------
dir.create("output/oof",  showWarnings = FALSE, recursive = TRUE)
dir.create("output/test", showWarnings = FALSE, recursive = TRUE)

if (TIER == "A") {
  # Tier A 只出 OOF，不参与集成，所以不需要 test 预测
  saveRDS(oof, sprintf("output/oof/oof_grid_%s.rds", MODEL_NAME))
  cat(sprintf("\n已保存 output/oof/oof_grid_%s.rds（长度 %s）\n",
              MODEL_NAME, format(length(oof), big.mark = ",")))
} else {
  saveRDS(oof, sprintf("output/oof/oof_%s.rds", MODEL_NAME))

  # test 预测：在全部训练数据上重新拟合插补器和模型
  cat("\n生成 test 预测 ... ")
  set.seed(SEED)
  imp    <- fit_imputer(X_pool)
  X_full <- derive_features(apply_imputer(imp, copy(X_pool)))
  X_test <- derive_features(apply_imputer(imp, copy(test_all)))
  pred_test <- fit_predict(X_full, y_pool, X_test)

  saveRDS(pred_test, sprintf("output/test/test_%s.rds", MODEL_NAME))
  cat("完成\n")
  cat(sprintf("已保存 oof_%s.rds（%s）和 test_%s.rds（%s）\n",
              MODEL_NAME, format(length(oof), big.mark = ","),
              MODEL_NAME, format(length(pred_test), big.mark = ",")))
}

cat("\n请把这一行填进 submissions/log.csv：\n")
cat(sprintf("  %s,,%s,%.5f,%.5f,,\n", Sys.Date(), MODEL_NAME, cv_mean, cv_sd))
