# =============================================================================
# lib_models.R —— 模型工厂
#
# 每个 make_*() 返回一个符合框架契约的 fit_predict(X_tr, y_tr, X_va) 函数。
#
# 为什么要有这个文件：
#   1. 消除 6 个 06_model_*.R 之间的代码重复（审查意见 2.4）
#   2. 让 09_ablation*.R 复用同一套模型定义，避免主模型改了参数、
#      消融脚本没跟着改，导致结果不可比（审查意见 3.3）
#   3. 早停逻辑只写一遍（审查意见 1.1）
#
# -----------------------------------------------------------------------------
# 关于早停与嵌套验证（审查意见 1.1 / 1.3）
# -----------------------------------------------------------------------------
# 原先所有 xgboost 都写死 nrounds = 600，那是拍脑袋定的：
#   - 无法判断 600 轮是过拟合还是欠拟合
#   - 每折的最优轮数不同，固定轮数把这个差异抹平了
#   - 训练轮数本身是一个超参数，用固定值等于用外层验证折隐式地选了它
#
# 现在的做法：在**训练折内部**再切 20% 出来做早停监控，
# 定下 best_iteration 之后，用这个轮数在**训练折全量**上重新训一遍。
#
# 这样外层验证折从头到尾没有参与任何模型选择，构成真正的嵌套交叉验证。
# 代价是每折要训两次，换来一个站得住脚的 CV 估计。
# =============================================================================

suppressMessages(library(data.table))

# -----------------------------------------------------------------------------
# 共用工具
# -----------------------------------------------------------------------------

#' 把 data.table 转成模型能吃的数值矩阵
#'
#' @param dt data.table
#' @param use_cols 要用的列名
#' @param onehot 是否对类别列做独热编码。xgboost/lightgbm 用整数编码会把
#'   无序类别当成有序数值（审查意见 3.5）。这三个类别特征已被消融证明是
#'   纯噪声，所以实际影响为零，但接口留着，需要时可以打开。
.to_matrix <- function(dt, use_cols, onehot = FALSE) {
  m <- dt[, ..use_cols]
  cat_cols <- names(m)[vapply(m, is.factor, logical(1))]

  if (onehot && length(cat_cols)) {
    for (cc in cat_cols) {
      lv <- levels(m[[cc]])
      for (l in lv) {
        set(m, j = paste0(cc, "_", l), value = as.integer(m[[cc]] == l))
      }
      set(m, j = cc, value = NULL)
    }
  } else {
    for (cc in cat_cols) set(m, j = cc, value = as.integer(m[[cc]]))
  }

  as.matrix(m)
}

#' 选出参与建模的列
.use_cols <- function(X, drop_extra = character(0)) {
  setdiff(names(X), c("id", "addicted_label", "is_train", drop_extra))
}

#' 从训练折里切一个内部验证集出来做早停
#'
#' 分层抽样，保证内部验证集的正例率与训练折一致。
.inner_split <- function(y, val_frac = 0.2) {
  va <- integer(0)
  for (cls in unique(y)) {
    idx <- which(y == cls)
    va <- c(va, sample(idx, floor(length(idx) * val_frac)))
  }
  list(tr = setdiff(seq_along(y), va), va = sort(va))
}

# -----------------------------------------------------------------------------
# xgboost
# -----------------------------------------------------------------------------
#' @param params 覆盖默认参数的 list
#' @param max_rounds 早停的上限。给足够大，让早停来决定实际轮数。
#' @param es_rounds 连续多少轮没提升就停
make_xgb <- function(params = list(), max_rounds = 3000L, es_rounds = 50L,
                     drop_extra = character(0), onehot = FALSE) {
  defaults <- list(
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
  p <- utils::modifyList(defaults, params)

  function(X_tr, y_tr, X_va) {
    uc <- .use_cols(X_tr, drop_extra)
    M_tr <- .to_matrix(X_tr, uc, onehot)
    M_va <- .to_matrix(X_va, uc, onehot)

    # --- 第一步：切内部验证集，用早停定轮数 ---
    sp <- .inner_split(y_tr)
    d_in  <- xgboost::xgb.DMatrix(M_tr[sp$tr, , drop = FALSE], label = y_tr[sp$tr])
    d_out <- xgboost::xgb.DMatrix(M_tr[sp$va, , drop = FALSE], label = y_tr[sp$va])

    probe <- xgboost::xgb.train(
      p, d_in, nrounds = max_rounds,
      evals = list(inner_val = d_out),
      early_stopping_rounds = es_rounds,
      verbose = 0
    )

    # ⚠ xgboost 3.x 的坑：booster 是外部指针包装，names() 只有 "ptr"，
    #   probe$best_iteration 恒为 NULL。必须走 xgb.attr()。
    #   而且它是 0-indexed —— best_iteration = 27 表示用了 28 轮
    #   （实际训练轮数 = 27 + es_rounds + 1）。所以要 +1。
    best <- suppressWarnings(
      as.integer(xgboost::xgb.attr(probe, "best_iteration"))
    )
    if (is.na(best) || best < 0) {
      # 不要静默兜底成某个常数 —— 那会把 API 变更伪装成一次正常的实验，
      # 分数悄悄变差而没有任何人发现。宁可炸掉。
      stop("无法从 xgboost 取到 best_iteration。",
           "很可能是 xgboost 版本变更导致 API 改变，请重新确认取值方式。",
           " 当前版本：", as.character(utils::packageVersion("xgboost")))
    }
    best <- best + 1L

    # --- 第二步：用定下的轮数，在训练折全量上重训 ---
    # 内部验证集只用来定轮数，不参与最终模型的评估，
    # 外层验证折全程没被碰过。
    d_full <- xgboost::xgb.DMatrix(M_tr, label = y_tr)
    model  <- xgboost::xgb.train(p, d_full, nrounds = best, verbose = 0)

    attr_out <- predict(model, xgboost::xgb.DMatrix(M_va))
    attr(attr_out, "best_iteration") <- best
    attr_out
  }
}

# -----------------------------------------------------------------------------
# lightgbm
# -----------------------------------------------------------------------------
make_lgb <- function(params = list(), max_rounds = 3000L, es_rounds = 50L,
                     drop_extra = character(0), onehot = FALSE) {
  defaults <- list(
    objective       = "binary",
    metric          = "auc",
    learning_rate   = 0.05,
    num_leaves      = 63L,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq    = 1L,
    min_data_in_leaf = 50L,
    num_threads     = parallel::detectCores(),
    verbosity       = -1L
  )
  p <- utils::modifyList(defaults, params)

  function(X_tr, y_tr, X_va) {
    uc <- .use_cols(X_tr, drop_extra)
    M_tr <- .to_matrix(X_tr, uc, onehot)
    M_va <- .to_matrix(X_va, uc, onehot)

    sp <- .inner_split(y_tr)
    d_in  <- lightgbm::lgb.Dataset(M_tr[sp$tr, , drop = FALSE], label = y_tr[sp$tr])
    d_out <- lightgbm::lgb.Dataset.create.valid(d_in, M_tr[sp$va, , drop = FALSE],
                                                label = y_tr[sp$va])

    probe <- lightgbm::lgb.train(
      params = p, data = d_in, nrounds = max_rounds,
      valids = list(inner_val = d_out),
      early_stopping_rounds = es_rounds,
      verbose = -1L
    )
    # lightgbm 的 best_iter 是 1-indexed，可以直接当 nrounds 用
    # （实测：best_iter = 24 时 current_iter() = 44 = 24 + es_rounds）。
    best <- probe$best_iter
    if (is.null(best) || !is.finite(best) || best < 1) {
      stop("无法从 lightgbm 取到 best_iter，可能是版本变更。当前版本：",
           as.character(utils::packageVersion("lightgbm")))
    }
    best <- as.integer(best)

    d_full <- lightgbm::lgb.Dataset(M_tr, label = y_tr)
    model  <- lightgbm::lgb.train(params = p, data = d_full,
                                  nrounds = best, verbose = -1L)

    out <- as.numeric(predict(model, M_va))
    attr(out, "best_iteration") <- best
    out
  }
}

# -----------------------------------------------------------------------------
# ranger（随机森林）
# -----------------------------------------------------------------------------
# 随机森林不需要早停 —— 树数越多方差越小，不会过拟合，只是收益递减。
# 树数在这里是算力预算而非需要调的超参数。
make_ranger <- function(num_trees = 200L, mtry = NULL, min_node_size = 50L,
                        drop_extra = character(0)) {
  function(X_tr, y_tr, X_va) {
    uc <- .use_cols(X_tr, drop_extra)
    d_tr <- as.data.frame(.to_matrix(X_tr, uc, onehot = FALSE))
    d_va <- as.data.frame(.to_matrix(X_va, uc, onehot = FALSE))

    # ranger 不接受 NA，所以这条路径只对做过插补的线有意义。
    if (anyNA(d_tr) || anyNA(d_va)) {
      stop("ranger 不能处理 NA。本条线必须先做插补（L1 原生 NaN 线不适用）。")
    }
    d_tr$.y <- factor(y_tr, levels = c(0, 1))

    m <- ranger::ranger(
      dependent.variable.name = ".y",
      data = d_tr,
      num.trees = num_trees,
      mtry = if (is.null(mtry)) max(1L, floor(sqrt(length(uc)))) else mtry,
      min.node.size = min_node_size,
      probability = TRUE,
      num.threads = parallel::detectCores(),
      verbose = FALSE
    )
    as.numeric(predict(m, d_va)$predictions[, "1"])
  }
}

# -----------------------------------------------------------------------------
# glmnet
# -----------------------------------------------------------------------------
#' @param alpha 弹性网混合系数。0 = ridge，1 = lasso，0.5 = 各一半。
#'   审查意见 2.5 指出原先固定 0.5 缺乏依据 —— 现在由 09_alpha_scan.R
#'   做敏感性分析，结论写进报告。
make_glmnet <- function(alpha = 0.5, nfolds = 3L, drop_extra = character(0)) {
  function(X_tr, y_tr, X_va) {
    uc <- .use_cols(X_tr, drop_extra)
    M_tr <- .to_matrix(X_tr, uc, onehot = TRUE)   # 线性模型必须独热
    M_va <- .to_matrix(X_va, uc, onehot = TRUE)

    # 派生特征里的比值在分母为 0 时是 NA，插补不覆盖这些格子。
    # glmnet 不接受 NA / Inf，统一按 0 处理。
    M_tr[!is.finite(M_tr)] <- 0
    M_va[!is.finite(M_va)] <- 0

    cvfit <- glmnet::cv.glmnet(M_tr, y_tr, family = "binomial",
                               alpha = alpha, nfolds = nfolds,
                               type.measure = "auc")
    # predict.cv.glmnet 是 S3 方法，不是导出对象，不能用 glmnet:: 取。
    as.numeric(predict(cvfit, newx = M_va, s = "lambda.min", type = "response"))
  }
}
