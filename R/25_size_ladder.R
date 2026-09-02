# =============================================================================
# 25_size_ladder.R —— 样本量阶梯：在部分训练集上做的模型选择，能不能迁移到全量
#
# 用法：
#   Rscript R/25_size_ladder.R           建池 + 跑全部 rung + 出分析
#   Rscript R/25_size_ladder.R --analyze 只重算分析（结果已存在时）
#   Rscript R/25_size_ladder.R 100k      只跑指定 rung
#
# 产出：output/pools/pool_<rung>.rds        行下标
#       output/ladder/meta_<rung>_<模型>.rds
#       output/size_ladder.rds             汇总与三个指标
#
# -----------------------------------------------------------------------------
# 这个实验回答什么
# -----------------------------------------------------------------------------
# 全项目的对比实验都跑在 20 万子样本（Tier A）上，只有最后交付才用全量。
# 这么做的理由是算力：全量一格要几分钟到十几分钟，而对比要跑几十格。
# 但它引出一个必须回答的问题：
#
#   **在 20 万上选出来的最优模型，就是全量上的最优模型吗？**
#
# 注意问的不是「20 万训练出的模型和全量训练出的模型一样好吗」——
# 那个答案显然是否（数据多总是更好），也不是我们关心的。
# 我们关心的是**选择**能不能迁移：如果两边选出的是同一个模型，
# 那么用小样本做筛选就是合法的，省下的算力没有代价。
#
# -----------------------------------------------------------------------------
# 设计
# -----------------------------------------------------------------------------
# 阶梯 n ∈ {5 万, 10 万, 20 万, 40 万, 69.1 万}，候选是 10 个非 L4 格
# （4 条插补线 × 4 个算法里，去掉需要 8 小时的 L4 那四格，
#  以及 L1 只有原生支持 NaN 的 xgboost / lightgbm 能跑）。
#
# 三个设计要点：
#
# 1. **池是嵌套的**：5 万 ⊂ 10 万 ⊂ 20 万 ⊂ 40 万 ⊂ 全量。
#    如果各规模独立抽样，"排名变了"就分不清是样本量的作用还是换了一批行的作用。
#    嵌套之后规模是唯一变化的量。
#
# 2. **20 万那一级就是冻结的 subsample_200k**，不另抽。
#    这样这一级同时充当「Tier A 网格带编码重跑」，一份算力两个用途。
#
# 3. **折号沿用冻结的 folds.rds**。每个池里的行保留它原来的折号，
#    所以任意两级之间的验证集是互相包含的，AUC 的比较不受折划分变化干扰。
#
# -----------------------------------------------------------------------------
# 报告三个指标，而不是只看 AUC
# -----------------------------------------------------------------------------
#   Spearman ρ      该规模下 10 个格的排名，与全量排名的秩相关
#   top-1 命中      该规模选出的冠军，是不是全量的冠军
#   选择遗憾        按该规模选出的模型，其**全量** AUC 与全量最优的差
#
# 第三个才是决策上真正关心的量：哪怕排名乱了，只要遗憾是 0.0001，
# 用小样本筛选也没有实际代价。
# =============================================================================

suppressMessages({library(data.table)})

SEED   <- 20260821L
RUNGS  <- c("50k", "100k", "200k", "400k")   # 全量那一级复用 output/oof/meta_*
NSIZE  <- c("50k" = 50000L, "100k" = 100000L, "200k" = 200000L, "400k" = 400000L)

CELLS <- c("L1_xgboost", "L1_lightgbm",
           "L2_xgboost", "L2_lightgbm", "L2_ranger", "L2_glmnet",
           "L3_xgboost", "L3_lightgbm", "L3_ranger", "L3_glmnet")

args    <- commandArgs(trailingOnly = TRUE)
analyze <- "--analyze" %in% args
picked  <- setdiff(args, "--analyze")
rungs   <- if (length(picked)) picked else RUNGS

dir.create("output/pools",  showWarnings = FALSE, recursive = TRUE)
dir.create("output/ladder", showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 一、建嵌套的池
# -----------------------------------------------------------------------------
#' 从一个已有的行集合里按 (折 × 类别) 分层抽出指定规模的子集
#'
#' @param pool  可抽的行下标（相对 train_all）
#' @param target 目标行数
#' @param folds,y 全量的折号与标签
strat_take <- function(pool, target, folds, y) {
  out <- integer(0)
  for (k in sort(unique(folds[pool]))) {
    for (cls in c(0L, 1L)) {
      cell <- pool[folds[pool] == k & y[pool] == cls]
      # as.numeric 不能省：两个整数相乘会溢出 32 位变 NA
      n_take <- round(as.numeric(target) * length(cell) / length(pool))
      n_take <- min(n_take, length(cell))
      out <- c(out, sample(cell, n_take))
    }
  }
  sort(out)
}

build_pools <- function() {
  feat   <- readRDS("output/features_raw.rds")
  folds  <- readRDS("output/folds.rds")
  train  <- feat[is_train == 1L]
  y      <- train$addicted_label
  sub200 <- readRDS("output/subsample_200k.rds")
  all_idx <- seq_len(nrow(train))

  set.seed(SEED + 2026L)

  # 20 万：直接用冻结的那份，不重抽
  p200 <- sub200
  # 10 万 ⊂ 20 万；5 万 ⊂ 10 万
  p100 <- strat_take(p200, NSIZE[["100k"]], folds, y)
  p050 <- strat_take(p100, NSIZE[["50k"]],  folds, y)
  # 40 万 = 20 万 ∪ 从补集里再分层抽 20 万
  rest <- setdiff(all_idx, p200)
  p400 <- sort(c(p200, strat_take(rest, NSIZE[["400k"]] - length(p200), folds, y)))

  pools <- list("50k" = p050, "100k" = p100, "200k" = p200, "400k" = p400)

  # ---- 检查：嵌套、分层、折分布 --------------------------------------------
  stopifnot("50k 不是 100k 的子集"  = all(p050 %in% p100),
            "100k 不是 200k 的子集" = all(p100 %in% p200),
            "200k 不是 400k 的子集" = all(p200 %in% p400))

  cat("嵌套池：\n")
  for (nm in names(pools)) {
    ii <- pools[[nm]]
    f  <- table(folds[ii])
    cat(sprintf("  %-5s %s 行  正类率 %.4f  折间行数极差 %s\n",
                nm, format(length(ii), big.mark = ","), mean(y[ii]),
                format(diff(range(f)), big.mark = ",")))
    saveRDS(ii, sprintf("output/pools/pool_%s.rds", nm))
  }
  cat(sprintf("  全量  %s 行  正类率 %.4f\n\n",
              format(length(all_idx), big.mark = ","), mean(y)))
  invisible(pools)
}

# -----------------------------------------------------------------------------
# 二、跑
# -----------------------------------------------------------------------------
#' Windows 上 system2(env=) 是坏的（R 会前置 Unix 的 env 命令），
#' 所以用 Sys.setenv + 继承。与 run_grid_full.R 里的同名函数一致。
run_with_env <- function(rscript, args, vars = character(0)) {
  old <- vapply(names(vars), function(k) Sys.getenv(k, unset = NA_character_),
                character(1))
  do.call(Sys.setenv, as.list(vars))
  on.exit({
    for (k in names(vars))
      if (is.na(old[[k]])) Sys.unsetenv(k)
      else do.call(Sys.setenv, setNames(list(old[[k]]), k))
  }, add = TRUE)
  system2(rscript, args, stdout = "", stderr = "")
}

run_rung <- function(rung) {
  rscript <- file.path(R.home("bin"), "Rscript")
  pool_f  <- sprintf("output/pools/pool_%s.rds", rung)
  tag     <- sprintf("pool_%s", rung)
  cat(sprintf("\n%s\n  rung %s（%s 行）\n%s\n",
              strrep("=", 66), rung,
              format(length(readRDS(pool_f)), big.mark = ","), strrep("=", 66)))

  for (i in seq_along(CELLS)) {
    cell <- CELLS[i]
    meta <- sprintf("output/ladder/meta_%s_%s.rds", tag, cell)
    if (file.exists(meta) && !nzchar(Sys.getenv("FORCE"))) {
      m <- readRDS(meta)
      cat(sprintf("  [%2d/%d] %-14s 已有 AUC %.5f，跳过\n",
                  i, length(CELLS), cell, m$cv_mean)); next
    }
    cat(sprintf("  [%2d/%d] %-14s ... ", i, length(CELLS), cell))
    t0 <- Sys.time()
    st <- run_with_env(rscript, sprintf("R/06_model_%s.R", cell),
                       c(TIER = "B", POOL_FILE = pool_f, QUIET = "1"))
    mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    if (st == 0 && file.exists(meta)) {
      m <- readRDS(meta)
      cat(sprintf("AUC %.5f ± %.5f  (%.1f 分钟)\n", m$cv_mean, m$cv_sd, mins))
    } else {
      cat(sprintf("失败（退出码 %s，%.1f 分钟）\n", st, mins))
    }
  }
}

# -----------------------------------------------------------------------------
# 三、分析
# -----------------------------------------------------------------------------
collect <- function() {
  rows <- list()
  for (rung in RUNGS) {
    tag <- sprintf("pool_%s", rung)
    for (cell in CELLS) {
      f <- sprintf("output/ladder/meta_%s_%s.rds", tag, cell)
      if (!file.exists(f)) next
      m <- readRDS(f)
      rows[[length(rows) + 1L]] <- data.table(
        rung = rung, n = m$n_pool, model = cell,
        auc = m$cv_mean, sd = m$cv_sd,
        fold_auc = list(m$fold_auc), minutes = m$minutes)
    }
  }
  # 全量那一级复用主网格的结果
  for (cell in CELLS) {
    f <- sprintf("output/oof/meta_%s.rds", cell)
    if (!file.exists(f)) next
    m <- readRDS(f)
    rows[[length(rows) + 1L]] <- data.table(
      rung = "full", n = 691369L, model = cell,
      auc = m$cv_mean, sd = m$cv_sd,
      fold_auc = list(m$fold_auc), minutes = m$minutes)
  }
  rbindlist(rows)
}

analyze_ladder <- function(tb) {
  full <- tb[rung == "full"]
  if (!nrow(full)) stop("缺全量结果，先跑 R/run_grid_full.R --no-l4")
  setkey(full, model)
  best_full     <- full[which.max(auc)]
  auc_full      <- setNames(full$auc, full$model)

  cat(sprintf("\n全量最优：%s，AUC %.5f\n\n", best_full$model, best_full$auc))

  out <- list()
  # 循环变量不能叫 rung —— tb 里有同名的列，data.table 的 i 表达式会
  # 优先取列，条件恒真，每一级都会拿到整张表。
  for (rg in c(RUNGS, "full")) {
    d <- tb[rung == rg]
    common <- intersect(d$model, names(auc_full))
    if (length(common) < 3L) next
    d <- d[model %in% common]
    rho  <- suppressWarnings(cor(d$auc, auc_full[d$model], method = "spearman"))
    tau  <- suppressWarnings(cor(d$auc, auc_full[d$model], method = "kendall"))
    pick <- d$model[which.max(d$auc)]
    # 选择遗憾：按这一级选出的模型，它在**全量**上的分数离全量最优差多少
    regret <- as.numeric(max(auc_full[common]) - auc_full[[pick]])
    out[[rg]] <- data.table(
      rung = rg, n = d$n[1], n_cell = nrow(d),
      spearman = rho, kendall = tau, pick = pick,
      hit = pick == best_full$model, regret = regret,
      auc_of_pick_here = max(d$auc), auc_of_pick_full = auc_full[[pick]])
  }
  res <- rbindlist(out)

  cat("规模   行数     格数  Spearman  Kendall  该级冠军        命中  选择遗憾\n")
  cat(strrep("-", 78), "\n")
  for (i in seq_len(nrow(res)))
    cat(sprintf("%-6s %-9s %2d    %+.3f    %+.3f   %-14s  %-4s  %.5f\n",
                res$rung[i], format(res$n[i], big.mark = ","), res$n_cell[i],
                res$spearman[i], res$kendall[i], res$pick[i],
                if (res$hit[i]) "是" else "否", res$regret[i]))

  cat("\n逐格 AUC（行=模型，列=规模）：\n")
  w <- dcast(tb, model ~ rung, value.var = "auc")
  cols <- intersect(c("50k", "100k", "200k", "400k", "full"), names(w))
  setcolorder(w, c("model", cols))
  w <- w[order(-get(tail(cols, 1)))]
  cat(sprintf("%-14s %s\n", "模型", paste(sprintf("%9s", cols), collapse = "")))
  for (i in seq_len(nrow(w)))
    cat(sprintf("%-14s %s\n", w$model[i],
                paste(sprintf("%9.5f", unlist(w[i, ..cols])), collapse = "")))

  # ---- 排名对照：ρ < 1 到底是谁造成的 --------------------------------------
  # 只报一个相关系数是不够的 —— 我们要知道排名的分歧集中在哪几个候选上。
  # 如果分歧集中在某一类模型上，那它就不是随机噪声，而是一个可解释的机制。
  rk <- copy(w)
  for (cc in cols) set(rk, j = cc, value = frank(-w[[cc]], ties.method = "min"))
  cat("\n名次对照（1 = 该规模下最好）：\n")
  cat(sprintf("%-14s %s %10s\n", "模型", paste(sprintf("%7s", cols), collapse = ""),
              "最大偏离"))
  drift <- integer(nrow(rk))
  for (i in seq_len(nrow(rk))) {
    v <- unlist(rk[i, ..cols])
    drift[i] <- max(abs(v - v[length(v)]))
    cat(sprintf("%-14s %s %10d\n", rk$model[i],
                paste(sprintf("%7d", v), collapse = ""), drift[i]))
  }
  rk[, max_drift := drift]

  # top-k 集合一致性：比名次更贴近"我们实际怎么用它"
  cat("\ntop-k 集合与全量的一致性：\n")
  full_ord <- w$model                                   # 已按全量降序
  topk <- rbindlist(lapply(cols, function(cc) {
    o <- w$model[order(-w[[cc]])]
    data.table(rung = cc,
               top1 = as.integer(identical(o[1], full_ord[1])),
               top3 = length(intersect(o[1:3], full_ord[1:3])),
               top5 = length(intersect(o[1:5], full_ord[1:5])))
  }))
  for (i in seq_len(nrow(topk)))
    cat(sprintf("  %-6s top-1 %s   top-3 命中 %d/3   top-5 命中 %d/5\n",
                topk$rung[i], if (topk$top1[i] == 1L) "是" else "否",
                topk$top3[i], topk$top5[i]))

  list(summary = res, wide = w, ranks = rk, topk = topk, raw = tb)
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
if (!analyze) {
  if (!all(file.exists(sprintf("output/pools/pool_%s.rds", RUNGS)))) build_pools()
  for (r in rungs) run_rung(r)
}

tb  <- collect()
res <- analyze_ladder(tb)
saveRDS(res, "output/size_ladder.rds")
cat("\n已保存 output/size_ladder.rds\n")
