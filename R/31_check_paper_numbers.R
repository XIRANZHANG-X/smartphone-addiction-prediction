# =============================================================================
# 31_check_paper_numbers.R —— 论文里的每个 AUC 数字，都必须能在产物里找到
#
# 用法：Rscript R/31_check_paper_numbers.R
#
# 本会话已经三次抓到文档与产物的数字不一致（lightgbm 的 +0.00475、
# 十成员对十四成员的集成分、L3/L4 的混淆比较）。人工核对不可靠，改成自动的。
#
# 做法：把论文里形如 0.9xxxx 的数字全抽出来，与产物里能算出的全部 AUC
# 比对（容差 5e-6，只为吸收打印舍入）。找不到的逐个列出，由人判断
# 是笔误还是一个本来就没有产物支撑的数字 —— 两种都必须处理。
# =============================================================================

suppressMessages({library(data.table)})

PAPER <- "paper/preprocessing-expressiveness.md"
TOL   <- 5e-6

stopifnot("找不到论文" = file.exists(PAPER))

# ---- 1. 收集产物里的全部 AUC -----------------------------------------------
known <- numeric(0)
add <- function(v) known <<- c(known, as.numeric(v[!is.na(v)]))

metas <- c(list.files("output/oof",    "^meta_.*\\.rds$", full.names = TRUE),
           list.files("output/ladder", "^meta_.*\\.rds$", full.names = TRUE),
           list.files("output/repeat", "\\.rds$",         full.names = TRUE),
           list.files("output/archive_pre_te/tierA_grid",
                      "^meta_.*\\.rds$", full.names = TRUE))
for (f in metas) {
  m <- readRDS(f)
  add(m$cv_mean); add(m$oof_auc); add(m$fold_auc)
}

if (file.exists("output/te_by_family.rds")) {
  te <- readRDS("output/te_by_family.rds")
  # 论文表 6 报的是跨折均值，不是单折值，两个都收
  for (a in names(te)) {
    add(te[[a]]$on);       add(te[[a]]$off)
    add(mean(te[[a]]$on)); add(mean(te[[a]]$off))
  }
}
if (file.exists("output/size_ladder.rds")) {
  lad <- readRDS("output/size_ladder.rds")
  cols <- setdiff(names(lad$wide), "model")
  for (cc in cols) add(lad$wide[[cc]])
}
if (file.exists("output/ensemble_best.rds")) {
  en <- readRDS("output/ensemble_best.rds")
  add(unlist(en$all_scores)); add(en$cv_auc); add(en$single_aucs)
}
# one-hot 对照（表 7）：Tier A 在 onehot_lr.rds，Full 在 onehot_full.rds；
# 论文报的同样是跨折均值，不是单折值
for (f in c("output/onehot_lr.rds", "output/onehot_full.rds")) {
  if (!file.exists(f)) next
  oh <- readRDS(f)
  for (nm in names(oh)) {
    if (is.numeric(oh[[nm]])) { add(oh[[nm]]); add(mean(oh[[nm]])) }
  }
}
for (f in c("output/feature_v2.rds", "output/cheap_wins.rds",
            "output/alpha_scan.rds",
            "output/ablation.rds", "output/tune_xgboost.rds")) {
  if (!file.exists(f)) next
  x <- readRDS(f)
  add(unlist(rapply(x, function(z) if (is.numeric(z)) z else NA_real_,
                    how = "unlist")))
}
known <- unique(round(known, 8))
cat(sprintf("产物里收集到 %d 个不同的数值\n", length(known)))

# ---- 2. 抽论文里的数字 -------------------------------------------------------
txt <- paste(readLines(PAPER, encoding = "UTF-8", warn = FALSE), collapse = "\n")
found <- unique(as.numeric(regmatches(txt,
           gregexpr("0\\.9[0-9]{4}", txt))[[1]]))
cat(sprintf("论文里出现 %d 个形如 0.9xxxx 的数字\n\n", length(found)))

# ---- 3. 比对 -----------------------------------------------------------------
missing <- found[vapply(found, function(v) !any(abs(known - v) < TOL), logical(1))]

if (!length(missing)) {
  cat("全部数字都能在产物里找到。\n")
} else {
  cat("★ 以下数字在产物里找不到，逐个查明是笔误还是缺证据：\n")
  for (v in sort(missing)) {
    ctx <- regmatches(txt, gregexpr(sprintf(".{0,70}%s.{0,40}",
                      format(v, nsmall = 5)), txt))[[1]]
    cat(sprintf("\n  %.5f\n", v))
    for (c1 in head(ctx, 2)) cat("    ...", gsub("\n", " ", c1), "...\n")
  }
  quit(save = "no", status = 1)
}
