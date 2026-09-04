# =============================================================================
# 32_mcar_check.R —— MCAR 再验证：逐列缺失率 + 缺失可预测性（联合独立性检验）
#
# 背景（对应审稿意见 T2，统计审稿人 + 数据质量审稿人共同提出，Major）：
#   论文 §2.2 原先只验证了"缺失是否与结局（addicted_label）无关"——12 列中
#   缺失组与非缺失组成瘾率最大差 0.0042（R/02_eda.R 发现 6）。审稿人原话：
#
#     "MCAR 是关于缺失机制与全部特征取值（含未观测值）相互独立性的性质，
#      结局率平衡只是必要条件。"
#
#   换句话说，"缺失与结局无关"和"缺失完全随机（MCAR：缺失与全部协变量
#   都无关）"是两个不同强度的命题，前者不能推出后者。本脚本补上后者的
#   检验。两个命题的复算结果都存进 output/mcar_check.rds，供
#   docs/实验报告.md 与论文 §2.2 引用时把两句话分开写，不再混用。
#
# 方法选择：
#   naniar::mcar_test()、BaylorEdPsych::LittleMCAR()、norm 都未安装
#  （已确认；不新增依赖——环境可能没有网络，且为一次分析新增依赖不成比例）。
#   mice（已装，3.19.0）不带 Little's 检验。
#
#   改用"缺失可预测性"诊断：对每一个有缺失的列 c，训练一个分类器，
#   用**其余 11 列**的取值预测 is.na(c)（1/0），5 折 CV 算 AUC。
#     AUC ≈ 0.5     —— 缺失无法从其他特征的取值预测，与 MCAR 一致
#     AUC 明显 > 0.5 —— 缺失和其他特征的取值有关，是 MAR（而非 MCAR）的证据
#   这直接回答审稿人的问题——"缺失机制是否独立于全部特征取值"，
#   而不只是独立于结局。
#
#   ⚠ 预测变量不含 addicted_label。原因：把结局也塞进预测变量，会把
#   "缺失与结局有关"和"缺失与协变量有关"这两个问题重新搅在一起，
#   正是审稿人指出的口径混淆。这里只用其余 11 个原始特征，
#   干净地只测协变量独立性；结局独立性由第二部分单独测。
#
# 为什么用 xgboost 而不是 glm()：
#   1. 其余 11 列各自也有约 4%~19% 的缺失（见第一部分）。glm() 的默认
#      na.action 会做逐行 complete-case 剔除——要求同一行里其余 11 列
#      全部观测到，在本数据上会丢掉大量行；而且这种"以其余列全观测为
#      条件"的子集选择，如果缺失模式本身列间有关联，可能引入
#      selection bias——不是我们想测的东西。
#      xgboost 原生处理 NA（为每次分裂学习缺失值的默认分支方向），
#      可以在全部行上用到其余列已观测到的信息，不必为了凑
#      complete-case 而系统性丢行。
#   2. xgboost 能捕捉非线性/交互关系，比线性 logit 更有把握探测出
#      "缺失与其他特征相关"这个信号（如果它存在）——检验力更强，
#      AUC 落在 0.5 附近时，"支持 MCAR"这个结论才更站得住脚
#     （用一个弱分类器测不出关系，不能反过来说明关系不存在）。
#   3. 项目里已有同一形状的诊断——R/19_adversarial.R 的对抗验证，
#      用其余列预测"这一行来自训练集还是测试集"这个二元指示。
#      这里沿用同一套机器：make_xgb()（R/lib_models.R）、同样的参数
#     （eta=0.1, max_depth=6, max_rounds=2000），复用 output/folds.rds
#      冻结折叠，与全项目的折叠契约（实验报告 2.1 节）和度量纪律保持
#      一致，而不是另起一套临时方法。
#
# 计算规模：逐列分类器用 Tier A 20 万行子样本（output/subsample_200k.rds），
#   与项目"Tier A 用于对比、全量用于交付"的既有分工一致（实验报告 2.5
#   节）——这是一次诊断性质的方法学检验，不是交付数字，跑全量 69 万行
#   在这里没有额外的方法学收益，只有更长的等待时间。12 列缺失率与
#   0.0042 的复算仍然用全量训练集（与 R/02_eda.R 口径一致，逐行可核对）。
#
# 用法：Rscript R/32_mcar_check.R
# 产出：output/mcar_check.rds + 屏幕报告
# =============================================================================

suppressMessages({
  library(data.table)
  library(pROC)
})
source("R/lib_models.R")

SEED <- 20260821L
set.seed(SEED)

hr <- function(x) cat("\n", strrep("=", 74), "\n ", x, "\n", strrep("=", 74), "\n", sep = "")

dir_out <- "output"
train <- readRDS(file.path(dir_out, "raw_train.rds"))
folds <- readRDS(file.path(dir_out, "folds.rds"))
sub   <- readRDS(file.path(dir_out, "subsample_200k.rds"))
stopifnot(length(folds) == nrow(train))

y <- train$addicted_label

feat_cols <- c(
  "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
  "work_study_hours", "sleep_hours", "notifications_per_day",
  "app_opens_per_day", "weekend_screen_time",
  "gender", "stress_level", "academic_work_impact"
)
stopifnot(all(feat_cols %in% names(train)))

# =============================================================================
hr("第一部分：12 列各自的缺失率（全量训练集，691,369 行）")
# =============================================================================
miss_tab <- rbindlist(lapply(feat_cols, function(cc) {
  data.table(col = cc, n_miss = sum(is.na(train[[cc]])),
             rate = mean(is.na(train[[cc]])))
}))[order(-rate)]
print(miss_tab, row.names = FALSE)

cols_with_missing <- miss_tab[rate > 0, col]
cat(sprintf("\n%d / %d 列存在缺失；缺失率范围 %.2f%% ~ %.2f%%。\n",
            length(cols_with_missing), length(feat_cols),
            100 * min(miss_tab$rate), 100 * max(miss_tab$rate)))
cat("（此前论文只引用过 61.06%，那是「至少缺一列的行占比」，不是任何单列的\n")
cat("缺失率——两者不应混用，这也是本次补充逐列缺失率表的原因之一。）\n")

# =============================================================================
hr("第二部分（一致性检查）：复算逐列缺失组 vs 非缺失组成瘾率最大差")
# =============================================================================
# 与 R/02_eda.R 发现 6 逐行同口径（同一份 raw_train.rds，同样的 12 列），
# 独立复算而非照抄论文里的数字。
res_outcome <- rbindlist(lapply(feat_cols, function(cc) {
  m <- is.na(train[[cc]])
  data.table(col = cc, n_miss = sum(m),
             rate_miss = mean(y[m]), rate_obs = mean(y[!m]),
             delta = mean(y[m]) - mean(y[!m]))
}))[order(-abs(delta))]
print(res_outcome, row.names = FALSE)

max_outcome_delta <- max(abs(res_outcome$delta))
cat(sprintf("\n复算最大差值 %.4f（论文/报告此前引用的数字：0.0042）。\n", max_outcome_delta))
cat("这一条测的是「缺失是否与结局无关」，是 MCAR 的必要条件，不是 MCAR 本身——\n")
cat("第三部分测更强的那个命题：缺失是否与其余协变量的取值无关。\n")

# =============================================================================
hr("第三部分：逐列缺失可预测性 —— 用其余 11 列预测该列是否缺失")
# =============================================================================
cat("方法：xgboost（eta=0.1, max_depth=6, max_rounds=2000, 早停），\n")
cat("5 折 CV，复用 output/folds.rds 冻结折叠；Tier A 20 万行子样本。\n")
cat("预测变量：其余 11 列原始取值，保留其自身缺失（NA），交给 xgboost 原生处理；\n")
cat("不含 addicted_label（结局独立性已在第二部分测过，这里只测协变量独立性）。\n\n")

X_pool <- train[sub]
f_pool <- folds[sub]
cat(sprintf("Tier A 子样本：%s 行\n", format(nrow(X_pool), big.mark = ",")))

# 抽样前后逐列缺失率的对照——确认 Tier A 没有系统性扭曲缺失率
cat("\n子样本 vs 全量缺失率对照（应接近一致）：\n")
chk <- rbindlist(lapply(feat_cols, function(cc) {
  data.table(col = cc,
             rate_full = mean(is.na(train[[cc]])),
             rate_tierA = mean(is.na(X_pool[[cc]])))
}))
chk[, diff_pp := round(100 * (rate_tierA - rate_full), 2)]
print(chk, row.names = FALSE)

fp <- make_xgb(list(eta = 0.1, max_depth = 6), max_rounds = 2000L)
n_fold <- length(unique(f_pool))

predict_missingness_auc <- function(target_col) {
  other_cols <- setdiff(feat_cols, target_col)
  X <- X_pool[, ..other_cols]
  yy <- as.integer(is.na(X_pool[[target_col]]))

  aucs <- numeric(n_fold)
  for (k in seq_len(n_fold)) {
    set.seed(SEED + k)
    tr_idx <- which(f_pool != k)
    va_idx <- which(f_pool == k)
    p <- fp(X[tr_idx], yy[tr_idx], X[va_idx])
    aucs[k] <- as.numeric(pROC::auc(pROC::roc(yy[va_idx], as.numeric(p), quiet = TRUE)))
  }
  list(auc_mean = mean(aucs), auc_sd = sd(aucs), auc_per_fold = aucs,
       n_missing = sum(yy), rate_missing = mean(yy))
}

pred_res <- list()
for (cc in cols_with_missing) {
  cat(sprintf("\n拟合中：%s ...\n", cc))
  r <- predict_missingness_auc(cc)
  pred_res[[cc]] <- r
  cat(sprintf("  %-24s AUC 均值 %.4f（sd %.4f，5 折：%s）\n",
              cc, r$auc_mean, r$auc_sd,
              paste(sprintf("%.4f", r$auc_per_fold), collapse = " / ")))
}

pred_tab <- rbindlist(lapply(names(pred_res), function(cc) {
  r <- pred_res[[cc]]
  data.table(col = cc, n_missing = r$n_missing, rate_missing = r$rate_missing,
             auc_mean = r$auc_mean, auc_sd = r$auc_sd,
             auc_fold1 = r$auc_per_fold[1], auc_fold2 = r$auc_per_fold[2],
             auc_fold3 = r$auc_per_fold[3], auc_fold4 = r$auc_per_fold[4],
             auc_fold5 = r$auc_per_fold[5])
}))[order(-auc_mean)]

# 判读阈值（本脚本自定，非统计惯例的硬边界，说明依据见旁注）：
#   AUC 换算成 Cohen's d 的近似公式 d = sqrt(2) * qnorm(AUC)（Rice & Harris 2005）。
#   AUC 0.55 -> d ~ 0.18（小效应下限附近），AUC 0.60 -> d ~ 0.36（小-中效应）。
#   取 0.55 / 0.60 只是给"多大算有实质可预测性"一个可复现的分界，
#   报告里同时给出精确数值，读者可以用自己的标准重新判断。
pred_tab[, verdict := fifelse(auc_mean >= 0.60, "明显可预测（不支持该列 MCAR）",
                       fifelse(auc_mean >= 0.55, "弱可预测（临界，值得关注）",
                                                  "与随机猜测无法区分（支持该列 MCAR）"))]

cat("\n汇总（按 AUC 降序）：\n")
print(pred_tab[, .(col, n_missing, rate_missing = round(rate_missing, 4),
                    auc_mean = round(auc_mean, 4), auc_sd = round(auc_sd, 4), verdict)],
      row.names = FALSE)

cat(sprintf("\nAUC 范围：%.4f ~ %.4f（12 列均值 %.4f）。\n",
            min(pred_tab$auc_mean), max(pred_tab$auc_mean), mean(pred_tab$auc_mean)))
cat(sprintf("%d / %d 列 AUC >= 0.55；%d / %d 列 AUC >= 0.60。\n",
            sum(pred_tab$auc_mean >= 0.55), nrow(pred_tab),
            sum(pred_tab$auc_mean >= 0.60), nrow(pred_tab)))

# =============================================================================
hr("第三部分附加：拆解可预测性的来源 —— 其余列的取值，还是其余列缺不缺？")
# =============================================================================
# 第三部分的结果不是"接近 0.5"，是全部 12 列都明显偏高（0.5656~0.7560）——
# 这是一个意料之外、比预想强得多的信号，必须先弄清楚它从哪来，
# 才能诚实地写进报告，不能就这样把"支持 MCAR"翻成"支持 MAR"交差。
#
# 问题在于：第三部分给 xgboost 的预测变量是"其余 11 列的原始取值，
# 保留各自的 NA，交给 xgboost 原生处理"。这混进了两种不同的信号：
#   (a) 其余列**取值本身**——这才是审稿人问的"缺失是否与协变量取值有关"；
#   (b) 其余列**是否缺失**——xgboost 对缺失值学习默认分裂方向，等于隐式
#       把"这一列缺不缺"也当成了一个可用信号。如果缺失倾向于按行成群出现
#      （这一行要么大部分都填了、要么大部分都没填），(b) 单独就能撑起
#       一个远高于 0.5 的 AUC，但那**不直接等于**违反 MCAR——MCAR 的定义
#       是缺失独立于数据取值，不要求各列的缺失指示互相独立。
#
# 拆成两个纯净变体，各自单独训练（其余设置不变：xgboost、5 折、
# 同一套冻结折叠）：
#   flags_only  只给其余 11 列的缺失指示（0/1），不给任何取值
#               —— 纯 (b)，"缺失是否按行成群"
#   values_cc   只用其余 11 列**全部观测到**的行，喂真实取值，不含任何 NA
#               —— 纯 (a)，"取值本身是否预测缺失"，代价是样本量下降
cat("目的：把第三部分的 AUC 拆成「其余列缺不缺」和「其余列的取值」两个来源。\n")
cat("flags_only：其余 11 列的缺失指示（0/1）；values_cc：其余 11 列全观测的行，真实取值。\n\n")

predict_auc_generic <- function(X, yy, f) {
  nf <- length(unique(f))
  aucs <- numeric(nf)
  for (k in seq_len(nf)) {
    set.seed(SEED + k)
    tr_idx <- which(f != k); va_idx <- which(f == k)
    p <- fp(X[tr_idx], yy[tr_idx], X[va_idx])
    aucs[k] <- as.numeric(pROC::auc(pROC::roc(yy[va_idx], as.numeric(p), quiet = TRUE)))
  }
  list(auc_mean = mean(aucs), auc_sd = sd(aucs), auc_per_fold = aucs)
}

decompose_res <- list()
for (cc in cols_with_missing) {
  other_cols <- setdiff(feat_cols, cc)
  Xf <- X_pool[, ..other_cols]
  yy_full <- as.integer(is.na(X_pool[[cc]]))

  # --- flags_only：其余列的缺失指示 ---
  Xflags <- copy(Xf)
  for (oc in other_cols) set(Xflags, j = oc, value = as.integer(is.na(Xf[[oc]])))
  r_flags <- predict_auc_generic(Xflags, yy_full, f_pool)

  # --- values_cc：其余列全部观测到的行，喂真实数值 ---
  cc_idx <- which(complete.cases(Xf))
  r_values <- predict_auc_generic(Xf[cc_idx], yy_full[cc_idx], f_pool[cc_idx])

  cat(sprintf("%-24s 混合 %.4f | flags_only %.4f | values_cc %.4f（n=%s，该子集缺失率 %.4f）\n",
              cc, pred_res[[cc]]$auc_mean, r_flags$auc_mean, r_values$auc_mean,
              format(length(cc_idx), big.mark = ","), mean(yy_full[cc_idx])))

  decompose_res[[cc]] <- list(flags = r_flags, values_cc = r_values,
                               n_cc = length(cc_idx),
                               rate_missing_cc = mean(yy_full[cc_idx]))
}

decompose_tab <- rbindlist(lapply(names(decompose_res), function(cc) {
  r <- decompose_res[[cc]]
  data.table(col = cc,
             auc_mixed        = pred_res[[cc]]$auc_mean,
             auc_flags_only   = r$flags$auc_mean, auc_flags_sd = r$flags$auc_sd,
             auc_values_cc    = r$values_cc$auc_mean, auc_values_cc_sd = r$values_cc$auc_sd,
             n_cc = r$n_cc, rate_missing_cc = r$rate_missing_cc)
}))[order(-auc_mixed)]

# 与 R/19_adversarial.R 同款的"保留比例"读法：flags_only / values_cc
# 相对于混合信号（减去 0.5 这个无信息基线）各解释了多少。
decompose_tab[, pct_from_flags  := round(100 * (auc_flags_only - 0.5) / (auc_mixed - 0.5), 1)]
decompose_tab[, pct_from_values := round(100 * (auc_values_cc  - 0.5) / (auc_mixed - 0.5), 1)]

cat("\n汇总（按混合 AUC 降序；pct_from_* 是各自相对混合信号解释了多少）：\n")
print(decompose_tab[, .(col, auc_mixed = round(auc_mixed, 4),
                         auc_flags_only = round(auc_flags_only, 4), pct_from_flags,
                         auc_values_cc = round(auc_values_cc, 4), pct_from_values,
                         n_cc)], row.names = FALSE)

cat(sprintf("\nflags_only 中位数 AUC %.4f；values_cc 中位数 AUC %.4f。\n",
            median(decompose_tab$auc_flags_only), median(decompose_tab$auc_values_cc)))
cat("判读：两组 pct_from_* 都远高于 0 意味着两种信号都在起作用，不是单一机制；\n")
cat("values_cc 明显高于 0.5 就足以独立确认「缺失与协变量取值有关」这个更强的结论，\n")
cat("不必依赖 flags_only 的解读。\n")

# =============================================================================
hr("保存")
# =============================================================================
out <- list(
  meta = list(script = "R/32_mcar_check.R", seed = SEED,
              n_train_full = nrow(train), n_tier_a = nrow(X_pool),
              date = as.character(Sys.Date())),
  missingness_rate        = miss_tab,
  outcome_delta_check     = res_outcome,
  max_outcome_delta       = max_outcome_delta,
  tierA_missingness_check = chk,
  predictability          = pred_tab,
  predictability_decomposed = decompose_tab,
  method_note = paste0(
    "缺失可预测性：xgboost 5 折 CV AUC，用其余 11 列原始取值（含各自的 NA，",
    "交给 xgboost 原生处理）预测目标列是否缺失；不含 addicted_label。",
    "Tier A 20 万行子样本，folds.rds 冻结折叠。",
    "AUC ~ 0.5 支持该列 MCAR；明显偏高是 MAR（缺失与协变量取值有关）的证据。",
    " predictability_decomposed 把该 AUC 拆成 flags_only（其余列是否缺失）",
    "与 values_cc（其余列全观测子集上的真实取值）两个来源，",
    "分开验证「缺失按行成群」与「缺失与取值有关」两件不同的事。"
  )
)
saveRDS(out, file.path(dir_out, "mcar_check.rds"))
cat("\n已保存 output/mcar_check.rds\n")
