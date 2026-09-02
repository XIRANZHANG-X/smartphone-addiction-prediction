# =============================================================================
# 18_new_features.R —— 把讨论区的做法放到我们的冻结折叠上测量
#
# 用法：Rscript R/18_new_features.R
# 产出：output/new_features.rds + 屏幕报告
#
# -----------------------------------------------------------------------------
# 口径：与 R/06_framework.R 完全一致
# -----------------------------------------------------------------------------
#   同一份 folds.rds、同一份 subsample_200k.rds、同一个 make_xgb() 工厂、
#   同样的 set.seed(SEED + k)、同样的 L1 线（不插补，NaN 交给 xgboost）。
#   唯一变化的是特征集与（一个变体里的）max_bin。
#
# 这样得到的数字可以和 output/feat17/meta_grid_L1_xgboost_f17.rds 里的
# 0.96107 直接配对比较。
#
# -----------------------------------------------------------------------------
# 为什么 target encoding 不能走框架的插补钩子
# -----------------------------------------------------------------------------
# 框架的 fit_imputer(X_tr) 拿不到 y。而 target encoding 必须看 y，
# 且**只能看训练折的 y**。讨论区第 34 帖有一次现成的教训：
# 有人在 CV 循环**之外**交叉拟合编码器，broccoli beef 推导出
# 「X₋ₖ 里每个样本都是用含 yₖ 的统计量编码的」，作者复查后承认，
# 改进 CV 循环内部之后公榜大跌 —— 说明原来的 CV 是虚高的。
#
# 所以这里自己写循环，把编码器的拟合放在 tr 索引内部。
# =============================================================================

suppressMessages({library(data.table); library(pROC)})
source("R/lib_models.R")
source("R/03_features.R")

SEED <- 20260821L
TIME_COLS <- c("daily_screen_time_hours", "social_media_hours",
               "weekend_screen_time", "gaming_hours",
               "work_study_hours", "sleep_hours")
TE_COLS   <- c("daily_screen_time_hours", "social_media_hours",
               "weekend_screen_time", "gaming_hours",
               "work_study_hours", "sleep_hours",
               "notifications_per_day", "app_opens_per_day")
TE_SMOOTH <- 20    # 平滑强度：稀疏取值向全局均值收缩

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

cat(sprintf("Tier A：%s 行，冻结 5 折\n", format(nrow(X_pool), big.mark = ",")))

# -----------------------------------------------------------------------------
# 候选特征的构造函数
# -----------------------------------------------------------------------------

#' 发现 1 的完整版：四项预算残差
#'
#' 我们原本的 other_screen 只减了 social + gaming。讨论区（ryota517 最先发布，
#' Georgy Mamarin / Dariush Afshar 反复引用）用的是四项。
#' R/17_discussion_checks.R 已在 421,427 行上确认四项约束同样 100% 成立，
#' 且最小残差恰为 0.000（三项版最小 0.100）——说明四项才是生成器真正强制的那条。
#'
#' ⚠ **2026-09-03 起这一臂的含义变了。** 本脚本的 base 用的是
#' derive_features() 的**当前**定义，而四项更正已于 2026-09-01 合并进去，
#' 所以 base 里的 other_screen 已经就是四项残差 —— 这里再 add_resid4()
#' 等于**追加一列完全重复的列**。重跑得到 +0.00002、2/5 同号，
#' 与 placebo 的 +0.00003、2/5 无法区分。
#'
#' 这个数字仍然有意义，只是意义换了：它现在测的是**"给模型一列冗余副本
#' 值多少"**，答案是「和一列随机数一样多，也就是不值什么」。
#' 「改对定义值多少」那个测量在 R/20_feature_v2.R 里（v1 vs v2 两个显式口径）。
add_resid4 <- function(dt) {
  dt[, resid4 := daily_screen_time_hours -
                 (social_media_hours + gaming_hours + work_study_hours)]
  dt[]
}

#' 生成器的舍入格点：小数位
#'
#' 取值被写到 0.01 的格点上。第一位小数让正例率摆动 0.0852（daily）、
#' 0.1047（weekend），而这与手机使用行为无关。
#' 关键在于 target encoding **捡不起它**：编码把每个精确取值分开处理，
#' 无法把「所有以 .2 结尾的」汇集起来。
add_decimal <- function(dt) {
  for (cc in TIME_COLS) {
    iv <- as.integer(round(dt[[cc]] * 100))
    set(dt, j = paste0("d1_", cc), value = (iv %/% 10L) %% 10L)
    set(dt, j = paste0("d2_", cc), value =  iv %% 10L)
  }
  dt[]
}

#' 逐取值 target encoding（**只在训练折上拟合**）
#'
#' 讨论区把它称作本数据集最大的单一杠杆（+0.0027~0.0032）。机制是：
#' 这份数据被舍入到一个格点上，按精确取值编码等于把这个格点捡起来，
#' 而按量级分裂做不到 —— 精确取值的目标率是**非单调**的。
#'
#' 平滑：(sum_y + prior*m) / (n + m)，m = TE_SMOOTH。
#' 训练折里没出现过的取值 → 回落到训练折的全局均值。
fit_te <- function(X, y) {
  prior <- mean(y)
  lapply(setNames(TE_COLS, TE_COLS), function(cc) {
    d <- data.table(v = X[[cc]], y = y)[!is.na(v)]
    tb <- d[, .(s = sum(y), n = .N), by = v]
    tb[, enc := (s + prior * TE_SMOOTH) / (n + TE_SMOOTH)]
    list(map = tb[, .(v, enc)], prior = prior)
  })
}
apply_te <- function(enc, dt) {
  for (cc in names(enc)) {
    m <- enc[[cc]]$map
    set(dt, j = paste0("te_", cc),
        value = m[data.table(v = dt[[cc]]), on = "v", x.enc])
    # 未见取值与 NA 都回落到先验；对树模型来说 NA 也可以留着，
    # 但回落到先验能让「未见过」和「本来就缺失」共享同一个含义。
    set(dt, j = paste0("te_", cc),
        value = fifelse(is.na(dt[[paste0("te_", cc)]]),
                        enc[[cc]]$prior, dt[[paste0("te_", cc)]]))
  }
  dt[]
}

#' 安慰剂列（讨论区第 40 帖 Georgy Mamarin）
#'
#' 随机值，与真实列同样的缺失模式。任何真实特征都必须比它跑得远。
#' 它只花一列，就能告诉你「你那个 +0.0003 到底是一个特征，还是一个折」。
add_placebo <- function(dt, seed) {
  set.seed(seed)
  v <- runif(nrow(dt))
  v[is.na(dt$daily_screen_time_hours)] <- NA_real_   # 复制真实列的缺失模式
  set(dt, j = "placebo", value = v)
  dt[]
}

# -----------------------------------------------------------------------------
# 一次交叉验证 = 一个配置
# -----------------------------------------------------------------------------
run_cfg <- function(name, build = function(Xtr, Xva, ytr) list(Xtr, Xva),
                    params = list()) {
  fp <- make_xgb(params)
  aucs <- numeric(0); nfeat <- NA_integer_; iters <- integer(0)
  t0 <- Sys.time()
  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)

    # L1 线：不插补。派生特征照常算（口径与 06_framework.R 一致）
    X_tr <- derive_features(copy(X_pool[tr]))
    X_va <- derive_features(copy(X_pool[va]))

    # 新特征在这里加。注意 build 只能看到 y_pool[tr]。
    both <- build(X_tr, X_va, y_pool[tr])
    X_tr <- both[[1]]; X_va <- both[[2]]

    if (is.na(nfeat))
      nfeat <- length(setdiff(names(X_tr), c("id", "addicted_label", "is_train")))

    p <- fp(X_tr, y_pool[tr], X_va)
    bi <- attr(p, "best_iteration"); if (!is.null(bi)) iters <- c(iters, bi)
    aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(y_pool[va], as.numeric(p),
                                                   quiet = TRUE))))
  }
  el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("  %-14s %2d 特征  AUC %.5f ± %.5f  (%.1f 分钟, 平均 %.0f 轮)\n",
              name, nfeat, mean(aucs), sd(aucs), el, mean(iters)))
  list(name = name, auc = aucs, n_feat = nfeat, iters = iters, minutes = el)
}

# -----------------------------------------------------------------------------
# 实验清单
# -----------------------------------------------------------------------------
cat("\n---- 逐项测量（每项约 3 分钟）----\n")
R <- list()

R$base <- run_cfg("base")

R$maxbin <- run_cfg("maxbin2048", params = list(max_bin = 2048L))

R$resid4 <- run_cfg("resid4", function(a, b, y) list(add_resid4(a), add_resid4(b)))

R$decimal <- run_cfg("decimal", function(a, b, y) list(add_decimal(a), add_decimal(b)))

R$te <- run_cfg("te", function(a, b, y) {
  e <- fit_te(a, y); list(apply_te(e, a), apply_te(e, b))
})

R$placebo <- run_cfg("placebo", function(a, b, y)
  list(add_placebo(a, 1L), add_placebo(b, 2L)))

# 组合：把上面测出为正的都叠上（先全叠，事后再按结果拆）
R$all <- run_cfg("all", function(a, b, y) {
  a <- add_decimal(add_resid4(a)); b <- add_decimal(add_resid4(b))
  e <- fit_te(a, y); list(apply_te(e, a), apply_te(e, b))
}, params = list(max_bin = 2048L))

# -----------------------------------------------------------------------------
# 配对报告
# -----------------------------------------------------------------------------
cat("\n", strrep("=", 74), "\n 配对检验（全部对 base，同一套折）\n", strrep("=", 74), "\n", sep = "")
cat(sprintf("%-14s %7s %10s %11s %8s %7s %9s\n",
            "配置", "特征数", "AUC", "vs base", "Cohen d", "同号", "p(t)"))
b <- R$base$auc
for (nm in names(R)) {
  d <- R[[nm]]$auc - b
  if (nm == "base") {
    cat(sprintf("%-14s %7d %10.5f %11s %8s %7s %9s\n",
                R[[nm]]$name, R[[nm]]$n_feat, mean(b), "—", "—", "—", "—"))
    next
  }
  cat(sprintf("%-14s %7d %10.5f %+11.5f %8.2f %5d/5 %9.4f\n",
              R[[nm]]$name, R[[nm]]$n_feat, mean(R[[nm]]$auc), mean(d),
              mean(d) / sd(d), sum(d > 0), t.test(d)$p.value))
}

cat("\n判读规则（讨论区第 40 帖）：任何真实特征都必须跑得比 placebo 远。\n")
cat("n=5 的 p 值不稳健，以 Cohen's d 与同号数为准。\n")

saveRDS(R, "output/new_features.rds")
cat("\n已保存 output/new_features.rds\n")
