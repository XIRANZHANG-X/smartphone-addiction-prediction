# =============================================================================
# 30_lattice_hit.R —— 插补填出的值，有多少能在编码表里查到
#
# 用法：Rscript R/30_lattice_hit.R
# 产出：output/lattice_hit.rds
#
# -----------------------------------------------------------------------------
# 为什么需要这个脚本
# -----------------------------------------------------------------------------
# 「两个预处理互相破坏」这条发现的机制证据是命中率：L3 用回归预测填补，
# 填出的值是任意实数，不在生成器的 0.01 格点上，逐取值编码查不到，
# 只能回落到全局均值。而至少缺一列的行占 61%。
#
# 这三个数字此前是临时算出来的，没有脚本。论文要引用它们，就必须能重算。
#
# -----------------------------------------------------------------------------
# 口径
# -----------------------------------------------------------------------------
# 对每一折、每一条插补线、每一个被编码的列：
#   分母 = 该列在**验证折**中原本缺失、因而被插补器填过的行数
#   分子 = 这些行填出的值，能在**训练折拟合出的编码表**里查到的行数
# 「查到」的定义与 apply_target_encoder() 完全一致：值出现在 enc$maps[[col]]$v 中。
#
# L1 不在此列 —— 它根本不填，缺失就留着缺失。
# =============================================================================

suppressMessages({library(data.table)})
source("R/03_features.R")

SEED <- 20260821L
LINES <- c("L2", "L3", "L4")

feat  <- readRDS("output/features_raw.rds")
folds <- readRDS("output/folds.rds")
sub   <- readRDS("output/subsample_200k.rds")

train_all <- feat[is_train == 1L]
X_pool <- train_all[sub]
y_pool <- train_all$addicted_label[sub]
f_pool <- folds[sub]

cat(sprintf("池 %s 行，%d 折，%d 个编码列\n\n",
            format(nrow(X_pool), big.mark = ","),
            length(unique(f_pool)), length(TE_COLS)))

rows <- list()

for (line in LINES) {
  source(sprintf("R/05_impute_%s.R", line))
  fit_i   <- get(paste0("fit_imputer_",   line))
  apply_i <- get(paste0("apply_imputer_", line))

  for (k in sort(unique(f_pool))) {
    set.seed(SEED + k)
    tr <- which(f_pool != k); va <- which(f_pool == k)

    # 1. 记下验证折里哪些格子原本是缺的 —— 插补之后就看不出来了
    was_na <- lapply(setNames(TE_COLS, TE_COLS),
                     function(cc) is.na(X_pool[[cc]][va]))

    # 2. 走与建模完全相同的路径：插补器只在训练折拟合
    imp <- fit_i(X_pool[tr])
    A   <- apply_i(imp, data.table::copy(X_pool[tr]))
    B   <- apply_i(imp, data.table::copy(X_pool[va]))

    # 3. 编码表也只在训练折拟合
    #    对 A 显式 copy()：apply_imputer_L3() 用 attr(dt, "l3_delta_trace") <- ...
    #    记收敛诊断，attr<- 在 R 里会整表复制、留下失效的 selfref，
    #    derive_features() 里的 := 因此触发（无害的）shallow-copy 警告。
    #    这里补一次 copy() 让 selfref 重新对齐，值不受影响。
    A <- derive_features(data.table::copy(A))
    enc <- fit_target_encoder(A, y_pool[tr])

    for (cc in TE_COLS) {
      idx <- which(was_na[[cc]])
      if (!length(idx)) next
      filled <- B[[cc]][idx]
      known  <- enc$maps[[cc]]$v
      n_hit  <- sum(filled %in% known)
      rows[[length(rows) + 1L]] <- data.table(
        line = line, col = cc, fold = k,
        n_imputed = length(idx), n_hit = n_hit,
        hit_rate = n_hit / length(idx))
    }
    cat(sprintf("  %s 第 %d 折完成\n", line, k))
  }
}

res <- rbindlist(rows)
saveRDS(res, "output/lattice_hit.rds")

cat("\n============ 按插补线汇总 ============\n")
s <- res[, .(n_imputed = sum(n_imputed), n_hit = sum(n_hit)), by = line]
s[, hit_rate := n_hit / n_imputed]
for (i in seq_len(nrow(s)))
  cat(sprintf("  %-3s 被插补 %s 个格子，查到 %s 个，命中率 %.4f%%\n",
              s$line[i], format(s$n_imputed[i], big.mark = ","),
              format(s$n_hit[i], big.mark = ","), 100 * s$hit_rate[i]))

cat("\n============ 逐列命中率（%）============\n")
w <- dcast(res[, .(hit_rate = sum(n_hit) / sum(n_imputed)), by = .(line, col)],
           col ~ line, value.var = "hit_rate")
print(w[, lapply(.SD, function(z) if (is.numeric(z)) round(100 * z, 4) else z)])

cat("\n已保存 output/lattice_hit.rds\n")
