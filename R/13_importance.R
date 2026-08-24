# =============================================================================
# 13_importance.R —— 特征重要性
#
# 用法：Rscript R/13_importance.R
# 产出：output/importance.rds、reports/figures/fig8_特征重要性.png
#
# -----------------------------------------------------------------------------
# 为什么同时算两种重要性
# -----------------------------------------------------------------------------
# 「特征重要性」不是一个唯一定义的量，不同算法给出的答案可以差很远。
# 本脚本算两种，并把它们和消融实验的结论放在一起对照：
#
#   1. 增益重要性（gain）—— xgboost 内建。统计每个特征在所有分裂点上
#      带来的损失下降总和。计算几乎免费，但有个已知缺陷：
#      **它偏爱取值多的连续特征** —— 连续特征能被反复切分，
#      每次切分都累计一点增益，而二元特征最多只贡献一次。
#
#   2. 置换重要性（permutation）—— 把某一列的取值随机打乱，
#      重新预测，看 AUC 掉多少。掉得越多说明这一列越重要。
#      它直接衡量「模型的预测有多依赖这一列」，不偏袒任何特征类型，
#      代价是每个特征都要重新预测一遍。
#
# 消融实验（09_ablation.R）衡量的是第三件事：**把特征从训练里彻底删掉**，
# 重新训练之后掉多少分。三者回答的问题不同：
#
#   gain        这一列在树里被用得多不多
#   permutation 已训好的模型有多依赖这一列
#   消融        这一列对「能训出多好的模型」有多大贡献
#
# 三者一致时结论最可靠；不一致的地方，恰恰是最值得讲的地方。
# =============================================================================

suppressMessages({
  library(data.table); library(ggplot2); library(scales)
  library(showtext);   library(sysfonts)
})
source("R/lib_models.R")
source("R/03_features.R")
source("R/05_impute_L1.R")

SEED <- 20260821L

# ---- 中文字体（与 12_figures.R 保持一致） -----------------------------------
font_path <- "C:/Windows/Fonts/simhei.ttf"
if (file.exists(font_path)) { font_add("SimHei", font_path); CN <- "SimHei" } else CN <- "sans"
showtext_auto(); showtext_opts(dpi = 300)

BLUE <- "#4E79A7"; ORANGE <- "#F28E2B"; RED <- "#E15759"; TEAL <- "#76B7B2"

# ---- 中文特征名 -------------------------------------------------------------
CN_NAME <- c(
  daily_screen_time_hours = "每日屏幕时间",
  social_media_hours      = "社交媒体时间",
  gaming_hours            = "游戏时间",
  work_study_hours        = "工作学习时间",
  sleep_hours             = "睡眠时间",
  notifications_per_day   = "每日通知数",
  app_opens_per_day       = "每日应用打开次数",
  weekend_screen_time     = "周末屏幕时间",
  age                     = "年龄",
  gender                  = "性别",
  stress_level            = "压力水平",
  academic_work_impact    = "学业工作影响",
  other_screen            = "其他屏幕时间（派生）",
  weekend_ratio           = "周末/日常比值（派生）",
  social_share            = "社交占比（派生）",
  gaming_share            = "游戏占比（派生）",
  free_frac               = "空闲时间占比（派生）",
  screen_social           = "屏幕+社交（派生）",
  n_missing               = "缺失个数"
)
pretty_name <- function(x) {
  out <- CN_NAME[x]
  out[is.na(out)] <- ifelse(grepl("^is_na_", x[is.na(out)]),
                            paste0("缺失指示：", CN_NAME[sub("^is_na_", "", x[is.na(out)])]),
                            x[is.na(out)])
  out[is.na(out)] <- x[is.na(out)]
  as.character(out)
}

# ---- 数据：用 Tier A 子样本，L1 线（表现最好的那条） ------------------------
cat("准备数据 ...\n")
feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")
train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

# 用第 1 折作训练/验证划分：训练折建模，验证折算置换重要性
set.seed(SEED + 1L)
tr <- which(f_pool != 1L)
va <- which(f_pool == 1L)

imp  <- fit_imputer_L1(X_pool[tr])
X_tr <- derive_features(apply_imputer_L1(imp, copy(X_pool[tr])))
X_va <- derive_features(apply_imputer_L1(imp, copy(X_pool[va])))
y_tr <- y_pool[tr]; y_va <- y_pool[va]

use_cols <- setdiff(names(X_tr), c("id", "addicted_label", "is_train"))
cat(sprintf("训练 %s 行，验证 %s 行，特征 %d 个\n",
            format(length(tr), big.mark = ","),
            format(length(va), big.mark = ","), length(use_cols)))

# ---- 训练一个模型（与网格同参数、同早停逻辑） -------------------------------
cat("训练模型 ...\n")
to_m <- function(dt) {
  m <- dt[, ..use_cols]
  for (cc in names(m)) if (is.factor(m[[cc]])) set(m, j = cc, value = as.integer(m[[cc]]))
  as.matrix(m)
}
M_tr <- to_m(X_tr); M_va <- to_m(X_va)

p <- list(objective = "binary:logistic", eval_metric = "auc", eta = 0.05,
          max_depth = 6, subsample = 0.8, colsample_bytree = 0.8,
          min_child_weight = 10, tree_method = "hist",
          nthread = parallel::detectCores())

sp <- .inner_split(y_tr)
probe <- xgboost::xgb.train(
  p, xgboost::xgb.DMatrix(M_tr[sp$tr, ], label = y_tr[sp$tr]),
  nrounds = 10000L,
  evals = list(v = xgboost::xgb.DMatrix(M_tr[sp$va, ], label = y_tr[sp$va])),
  early_stopping_rounds = 50L, verbose = 0)
best <- as.integer(xgboost::xgb.attr(probe, "best_iteration")) + 1L
model <- xgboost::xgb.train(p, xgboost::xgb.DMatrix(M_tr, label = y_tr),
                            nrounds = best, verbose = 0)
cat(sprintf("早停选定 %d 轮\n", best))

fast_auc <- function(y, pr) {
  r <- rank(pr, ties.method = "average")
  n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(length(y)) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
base_auc <- fast_auc(y_va, predict(model, xgboost::xgb.DMatrix(M_va)))
cat(sprintf("验证折基准 AUC %.5f\n\n", base_auc))

# ---- 1. 增益重要性 ----------------------------------------------------------
cat("计算增益重要性 ...\n")
gain <- as.data.table(xgboost::xgb.importance(model = model))
setnames(gain, "Feature", "feature")
gain <- gain[, .(feature, gain = Gain)]

# ---- 2. 置换重要性 ----------------------------------------------------------
# 每个特征打乱 3 次取平均，降低随机性带来的波动。
cat("计算置换重要性（每个特征打乱 3 次）...\n")
set.seed(SEED)
N_REP <- 3L
perm <- rbindlist(lapply(seq_along(use_cols), function(j) {
  drops <- numeric(N_REP)
  for (r in seq_len(N_REP)) {
    Mp <- M_va
    Mp[, j] <- Mp[sample.int(nrow(Mp)), j]
    drops[r] <- base_auc - fast_auc(y_va, predict(model, xgboost::xgb.DMatrix(Mp)))
  }
  if (j %% 6 == 0) cat(sprintf("  %d/%d\n", j, length(use_cols)))
  data.table(feature = use_cols[j], perm_drop = mean(drops), perm_sd = sd(drops))
}))

# ---- 合并 -------------------------------------------------------------------
res <- merge(perm, gain, by = "feature", all.x = TRUE)
res[is.na(gain), gain := 0]
res[, name_cn := pretty_name(feature)]
setorder(res, -perm_drop)

cat("\n================= 特征重要性（按置换重要性排序）=================\n")
cat(sprintf("%-26s %12s %10s\n", "特征", "置换重要性", "增益占比"))
for (i in seq_len(nrow(res))) {
  cat(sprintf("%-26s %12.5f %9.2f%%\n",
              res$name_cn[i], res$perm_drop[i], 100 * res$gain[i]))
}

saveRDS(list(base_auc = base_auc, table = res), "output/importance.rds")

# ---- 图 8 -------------------------------------------------------------------
cat("\n绘图 ...\n")
top <- head(res, 14)
top[, name_cn := factor(name_cn, levels = rev(name_cn))]
top[, grp := fifelse(perm_drop > 0.01, "核心驱动",
              fifelse(perm_drop > 0.001, "有实质贡献", "可忽略"))]
top[, grp := factor(grp, levels = c("核心驱动", "有实质贡献", "可忽略"))]

p8 <- ggplot(top, aes(x = name_cn, y = perm_drop, fill = grp)) +
  geom_col(width = 0.68, color = "black", linewidth = 0.9) +
  geom_errorbar(aes(ymin = perm_drop - perm_sd, ymax = perm_drop + perm_sd),
                width = 0.25, linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.4f", perm_drop)), hjust = -0.18,
            family = CN, fontface = "bold", size = 4.4) +
  scale_fill_manual(values = c("核心驱动" = RED, "有实质贡献" = ORANGE,
                               "可忽略" = TEAL), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
  coord_flip() +
  labs(
    title    = "图 8  置换重要性：打乱这一列，AUC 掉多少",
    subtitle = paste0("在验证折上把某一列随机打乱后重新预测，掉分越多说明模型越依赖它。",
                      "\n误差棒为 3 次打乱的标准差；基准 AUC ", sprintf("%.5f", base_auc)),
    x = NULL, y = "AUC 下降量"
  ) +
  theme_bw(base_size = 15) +
  theme(
    text             = element_text(family = CN, face = "bold"),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 2),
    axis.title       = element_text(face = "bold", size = 19),
    axis.text        = element_text(face = "bold", size = 14, color = "black"),
    axis.ticks       = element_line(color = "black", linewidth = 1),
    panel.grid.major = element_line(linetype = "dashed", color = "grey70",
                                    linewidth = 0.6),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 21),
    plot.subtitle    = element_text(size = 13, color = "grey30"),
    legend.text      = element_text(face = "bold"),
    legend.position  = "top"
  )

dir.create("reports/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("reports/figures/fig8_特征重要性.png", p8,
       width = 11, height = 8, dpi = 300, bg = "white")
cat("已保存 reports/figures/fig8_特征重要性.png\n")

# ---- 三种口径的一致性检查 ---------------------------------------------------
if (file.exists("output/ablation.rds")) {
  ab <- readRDS("output/ablation.rds")
  base_ab <- mean(ab[["L1_full"]]$auc)
  pair <- list(notifications_per_day = "L1_no_notif",
               app_opens_per_day     = "L1_no_opens",
               age                   = "L1_no_age")
  cat("\n=========== 三种口径对照 ===========\n")
  cat(sprintf("%-20s %12s %12s %12s\n", "特征", "增益占比", "置换重要性", "消融掉分"))
  for (f in names(pair)) {
    if (is.null(ab[[pair[[f]]]])) next
    cat(sprintf("%-20s %11.2f%% %12.5f %12.5f\n",
                CN_NAME[f], 100 * res[feature == f, gain],
                res[feature == f, perm_drop],
                base_ab - mean(ab[[pair[[f]]]]$auc)))
  }
  cat("\n三者方向一致时结论最可靠。增益口径偏爱取值多的连续特征，\n")
  cat("置换与消融不受此影响，因此以后两者为准。\n")
}
