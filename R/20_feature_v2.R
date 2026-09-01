# =============================================================================
# 20_feature_v2.R —— 第二轮：拆开重叠 + 改对算错的列 + 更安全的 TE
#
# 用法：Rscript R/20_feature_v2.R
# 产出：output/feature_v2.rds
#
# -----------------------------------------------------------------------------
# 第一轮（R/18）留下的三个问题
# -----------------------------------------------------------------------------
# 1. all(+0.00445) < 各项之和(+0.00592) —— 它们**互相替代**，必须拆开定价。
#    讨论区第 12 帖测过其中一对：max_bin 与取值编码是替代关系，
#    "只有不到 60% 的总和存活下来"。我们这里要自己拆。
#
# 2. 第一轮的 resid4 是**追加**一列，而不是把算错的 other_screen **改对**。
#    追加会稀释 feature fraction，改对不会 —— 是两个不同的问题。
#    同理 free_frac 的分母把 work 减了两遍（见发现 8 的更正）。
#
# 3. TE 在训练折内部是**样本内**拟合的：编码见过它自己被应用到的那些行的标签。
#    验证折是干净的（编码只用训练折的 y），所以第一轮 +0.00391 的**测量是诚实的**；
#    但模型在训练侧看到的是一个"知道答案"的编码，可能过度依赖它。
#    标准做法是在训练折内部再做一层折外编码。这里两个版本都测。
#
# 口径与 R/06_framework.R、R/18 完全一致。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

SEED <- 20260821L
TIME_COLS <- c("daily_screen_time_hours", "social_media_hours",
               "weekend_screen_time", "gaming_hours",
               "work_study_hours", "sleep_hours")
TE_COLS   <- c(TIME_COLS, "notifications_per_day", "app_opens_per_day")
TE_SMOOTH <- 20
TE_INNER  <- 5L      # 训练折内部再切几折做折外编码

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

safe_div <- function(num, den) ifelse(is.na(num) | is.na(den) | den <= 0,
                                      NA_real_, num / den)

# -----------------------------------------------------------------------------
# 派生特征的两个口径
# -----------------------------------------------------------------------------
# v1 = 现状：other_screen 只减两项、free_frac 分母减两遍 work
derive_v1 <- derive_features

# v2 = 按发现 1、8 的更正改对（**替换**，不是追加）
derive_v2 <- function(dt) {
  dt <- derive_features(dt)
  dt[, other_screen := daily_screen_time_hours -
                       (social_media_hours + gaming_hours + work_study_hours)]
  dt[, free_frac := safe_div(daily_screen_time_hours, 24 - sleep_hours)]
  dt[]
}

add_decimal <- function(dt) {
  for (cc in TIME_COLS) {
    iv <- as.integer(round(dt[[cc]] * 100))
    set(dt, j = paste0("d1_", cc), value = (iv %/% 10L) %% 10L)
    set(dt, j = paste0("d2_", cc), value =  iv %% 10L)
  }
  dt[]
}

# -----------------------------------------------------------------------------
# 逐取值 target encoding：两个版本
# -----------------------------------------------------------------------------
.te_table <- function(v, y, prior) {
  d <- data.table(v = v, y = y)[!is.na(v)]
  tb <- d[, .(s = sum(y), n = .N), by = v]
  tb[, enc := (s + prior * TE_SMOOTH) / (n + TE_SMOOTH)]
  tb[, .(v, enc)]
}

#' 简单版：在整个训练折上拟合，应用到训练折与验证折
#' 验证折是干净的（只用训练折的 y），但训练侧是样本内的。
te_simple <- function(X_tr, y_tr, X_va) {
  prior <- mean(y_tr)
  for (cc in TE_COLS) {
    m <- .te_table(X_tr[[cc]], y_tr, prior)
    for (D in list(X_tr, X_va)) {
      e <- m[data.table(v = D[[cc]]), on = "v", x.enc]
      set(D, j = paste0("te_", cc), value = fifelse(is.na(e), prior, e))
    }
  }
  list(X_tr, X_va)
}

#' 嵌套版：训练折内部再切 TE_INNER 折，训练行只拿**折外**的编码；
#' 验证折用整个训练折拟合的编码（它本来就在外面）。
#' 这样训练侧看到的编码不再"知道自己的答案"。
te_nested <- function(X_tr, y_tr, X_va) {
  prior <- mean(y_tr)
  n <- nrow(X_tr)
  g <- integer(n)
  for (cls in c(0L, 1L)) {
    ii <- which(y_tr == cls); g[sample(ii)] <- rep_len(seq_len(TE_INNER), length(ii))
  }
  for (cc in TE_COLS) {
    out <- rep(NA_real_, n)
    for (k in seq_len(TE_INNER)) {
      inn <- which(g != k); oth <- which(g == k)
      m <- .te_table(X_tr[[cc]][inn], y_tr[inn], prior)
      out[oth] <- m[data.table(v = X_tr[[cc]][oth]), on = "v", x.enc]
    }
    set(X_tr, j = paste0("te_", cc), value = fifelse(is.na(out), prior, out))
    m_full <- .te_table(X_tr[[cc]], y_tr, prior)
    e <- m_full[data.table(v = X_va[[cc]]), on = "v", x.enc]
    set(X_va, j = paste0("te_", cc), value = fifelse(is.na(e), prior, e))
  }
  list(X_tr, X_va)
}

# -----------------------------------------------------------------------------
run_cfg <- function(name, dv = derive_v1, dec = FALSE, te = NULL,
                    params = list()) {
  fp <- make_xgb(params)
  aucs <- numeric(0); nf <- NA_integer_; it <- integer(0)
  t0 <- Sys.time()
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)
    X_tr <- dv(copy(X_pool[tr])); X_va <- dv(copy(X_pool[va]))
    if (dec) { X_tr <- add_decimal(X_tr); X_va <- add_decimal(X_va) }
    if (!is.null(te)) { b <- te(X_tr, y_pool[tr], X_va); X_tr <- b[[1]]; X_va <- b[[2]] }
    if (is.na(nf)) nf <- length(setdiff(names(X_tr),
                                        c("id", "addicted_label", "is_train")))
    p <- fp(X_tr, y_pool[tr], X_va)
    bb <- attr(p, "best_iteration"); if (!is.null(bb)) it <- c(it, bb)
    aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(y_pool[va], as.numeric(p),
                                                   quiet = TRUE))))
  }
  cat(sprintf("  %-22s %2d 特征  AUC %.5f ± %.5f  (%.1f 分钟, %.0f 轮)\n",
              name, nf, mean(aucs), sd(aucs),
              as.numeric(difftime(Sys.time(), t0, units = "mins")), mean(it)))
  list(name = name, auc = aucs, n_feat = nf, iters = it)
}

MB <- list(max_bin = 2048L)
cat(sprintf("Tier A：%s 行，冻结 5 折\n\n", format(nrow(X_pool), big.mark = ",")))
R <- list()

cat("---- A 组：把算错的列改对（不含 TE）----\n")
R$v1  <- run_cfg("v1 现状")
R$v2  <- run_cfg("v2 改对定义", derive_v2)

cat("\n---- B 组：TE 的两个版本 ----\n")
R$te_s <- run_cfg("TE 简单版",   derive_v1, te = te_simple)
R$te_n <- run_cfg("TE 嵌套版",   derive_v1, te = te_nested)

cat("\n---- C 组：TE 在场时，其余三项还值多少（拆重叠）----\n")
R$te_mb  <- run_cfg("TE + maxbin",       derive_v1, te = te_simple, params = MB)
R$te_dec <- run_cfg("TE + decimal",      derive_v1, te = te_simple, dec = TRUE)
R$te_v2  <- run_cfg("TE + 改对定义",     derive_v2, te = te_simple)
R$full   <- run_cfg("TE + 全部",         derive_v2, te = te_simple, dec = TRUE,
                    params = MB)

# -----------------------------------------------------------------------------
cat("\n", strrep("=", 78), "\n 配对检验\n", strrep("=", 78), "\n", sep = "")
rep_vs <- function(basename, keys) {
  b <- R[[basename]]$auc
  cat(sprintf("\n对 %s（%.5f）：\n", R[[basename]]$name, mean(b)))
  cat(sprintf("  %-22s %7s %10s %11s %8s %7s\n",
              "配置", "特征数", "AUC", "差值", "Cohen d", "同号"))
  for (nm in keys) {
    d <- R[[nm]]$auc - b
    cat(sprintf("  %-22s %7d %10.5f %+11.5f %8.2f %5d/5\n",
                R[[nm]]$name, R[[nm]]$n_feat, mean(R[[nm]]$auc), mean(d),
                mean(d) / sd(d), sum(d > 0)))
  }
}
rep_vs("v1",   c("v2", "te_s", "te_n"))
rep_vs("te_s", c("te_mb", "te_dec", "te_v2", "full"))

saveRDS(R, "output/feature_v2.rds")
cat("\n已保存 output/feature_v2.rds\n")
