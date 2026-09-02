# =============================================================================
# 11_report.R —— 汇总全部实验结果，生成 Markdown 表格
#
# 用法：Rscript R/11_report.R
# 产出：output/results.md
#
# 文档里的每一个数字都应该来自这个脚本，而不是手抄。
# 手抄一次错一次，而且改了实验之后文档不会跟着变。
# =============================================================================

suppressMessages({library(data.table)})
source("R/09_stats.R")

out <- character(0)
add <- function(...) out <<- c(out, sprintf(...))

add("# 实验结果汇总")
add("")
add("> 本文件由 `R/11_report.R` 自动生成于 %s。",
    format(Sys.time(), "%Y-%m-%d %H:%M"))
add("> 不要手工编辑 —— 重跑脚本即可刷新。")
add("")
add("## ⚠ 口径声明")
add("")
add("全部实验结果的统一汇总与结论见 **[docs/实验报告.md](../docs/实验报告.md)**；")
add("本文件是原始数据表，由脚本自动生成。")
add("")
add("本文件包含**两套口径**的数字，不要混着读：")
add("")
add("| 口径 | 特征 | 数据 | 覆盖哪几节 |")
add("|---|---|---|---|")
add("| **交付口径** | 25（含逐取值编码） | 全量 691,369 行 | 第七节 |")
add("| **对比口径（现役）** | 25（含逐取值编码） | Tier A 200,000 子样本 | 第三~六、八节 |")
add("| **对比口径（历史）** | 17（编码之前） | Tier A 200,000 子样本 | **仅第一节** |")
add("")
add("两套并存是有意的：Tier A 便宜、是所有方法对比的载体，其内部可比性")
add("由冻结折叠保证；全量昂贵，但它是真正要交付的数字。")
add("**第九节（实验报告）已验证：在 20 万上做模型选择与集成配权，")
add("选择遗憾为 0、权重迁移代价上界 0.00003。**")
add("")
add("⚠ **第一节仍是编码之前的口径**（数据源 `output/oof/meta_grid_*.rds`），")
add("保留作历史记录，备份在 `output/archive_pre_te/tierA_grid/`。")
add("**它的绝对数值不能与其余各节相减。**")
add("加了编码之后的 20 万口径见第八节的逐格 AUC 表（`200k` 那一列）。")
add("")
add("⚠ 另有一个已知代价：**Tier A 会系统性低估参数个数随取值数增长的方法**。")
add("实测精确取值 one-hot 在 16 万训练行上是 0.95583、55 万行上是 0.95929，")
add("差 0.0035 纯粹来自样本量。见 docs/实验报告.md 第 10.2 节。")
add("")
add("")

# -----------------------------------------------------------------------------
# 一、实验网格
# -----------------------------------------------------------------------------
g <- grid_table()
add("## 一、实验网格（**历史口径**：Tier A 20 万行，17 特征，**编码之前**）")
add("")

if (!nrow(g)) {
  add("_尚无结果。_")
} else {
  add("### 完整排名")
  add("")
  add("| 排名 | 模型 | 插补 | 算法 | CV AUC | sd | 早停轮数 | 耗时(min) |")
  add("|---|---|---|---|---|---|---|---|")
  for (i in seq_len(nrow(g))) {
    r <- g[i]
    add("| %d | `%s` | %s | %s | **%.5f** | %.5f | %s | %.1f |",
        i, r$model, r$impute, r$algo, r$cv_mean, r$cv_sd,
        if (is.na(r$best_iter)) "—" else sprintf("%.0f", r$best_iter),
        r$minutes)
  }
  add("")

  # 网格矩阵
  add("### 网格矩阵")
  add("")
  algos <- c("xgboost", "lightgbm", "ranger", "glmnet")
  lines <- c("L1", "L2", "L3", "L4")
  hdr <- paste(c("| |", paste(algos, collapse = " | ")), collapse = " ")
  add("%s |", hdr)
  add("|%s", paste(rep("---|", length(algos) + 1), collapse = ""))
  for (ln in lines) {
    cells <- vapply(algos, function(al) {
      v <- g[impute == ln & algo == al, cv_mean]
      if (!length(v)) {
        if (ln == "L1" && al %in% c("ranger", "glmnet")) "—（不支持 NA）" else "⬜"
      } else sprintf("%.5f", v[1])
    }, character(1))
    add("| **%s** | %s |", ln, paste(cells, collapse = " | "))
  }
  add("")

  # 按算法看插补的影响
  add("### 同一算法内，插补策略的影响")
  add("")
  for (al in algos) {
    sub <- g[algo == al][order(-cv_mean)]
    if (nrow(sub) < 2) next
    add("**%s**：%s", al,
        paste(sprintf("%s %.5f", sub$impute, sub$cv_mean), collapse = " > "))
    add("")
  }
}

# -----------------------------------------------------------------------------
# 二、配对检验
# -----------------------------------------------------------------------------
add("## 二、配对检验")
add("")
add("同折同行比较。`n=15` 来自重复交叉验证（3 个 fold seed × 5 折），")
add("`n=5` 是单次 CV 的回退值。")
add("")
add("**关于检验方法**：n=5 时符号秩检验的双侧 p 值下限是 2/2⁵ = 0.0625，")
add("永远达不到 0.05 —— 这是检验本身的分辨率极限，不是证据不足。")
add("因此主口径用重复 CV 把 n 提到 15，并始终同时报 Cohen's d 与符号一致性。")
add("")

pairs <- list(
  c("L1_xgboost", "L2_xgboost"), c("L2_xgboost", "L3_xgboost"),
  c("L3_xgboost", "L4_xgboost"), c("L1_xgboost", "L4_xgboost"),
  c("L3_glmnet",  "L2_glmnet"),
  c("L1_lightgbm","L4_lightgbm"), c("L3_ranger", "L2_ranger")
)
rows <- rbindlist(lapply(pairs, function(p) compare(p[1], p[2])), fill = TRUE)

if (nrow(rows)) {
  add("| A | B | 来源 | n | 均值差 | Cohen's d | 效应量 | 同号 | p(t) | p(符号秩) | p(符号) |")
  add("|---|---|---|---|---|---|---|---|---|---|---|")
  for (i in seq_len(nrow(rows))) {
    r <- rows[i]
    add("| `%s` | `%s` | %s | %d | %+.5f | %.2f | %s | %s | %.4f | %.4f | %.4f |",
        r$a, r$b, if (r$source == "repeated") "重复CV" else "单次CV",
        r$n, r$mean_diff, r$cohens_d, interpret_d(r$cohens_d),
        r$same_sign, r$p_t, r$p_wilcox, r$p_sign)
  }
} else {
  add("_尚无可比较的结果。_")
}
add("")

# -----------------------------------------------------------------------------
# 三、消融
# -----------------------------------------------------------------------------
add("## 三、特征消融（**现役口径**：Tier A 20 万行，25 特征含编码）")
add("")
if (file.exists("output/ablation.rds")) {
  ab <- readRDS("output/ablation.rds")
  add("| 变体 | 特征数 | AUC | sd | 与同线 full 的差 | Cohen's d | 同号 | p(t) |")
  add("|---|---|---|---|---|---|---|---|")
  for (k in names(ab)) {
    base_line <- sub("_.*$", "", k)
    b <- ab[[paste0(base_line, "_full")]]
    r <- ab[[k]]
    if (endsWith(k, "_full") || is.null(b)) {
      add("| `%s` | %d | %.5f | %.5f | — | — | — | — |",
          k, r$n_feat, mean(r$auc), sd(r$auc))
    } else {
      d <- r$auc - b$auc
      coh <- if (sd(d) > 0) mean(d) / sd(d) else NA_real_
      add("| `%s` | %d | %.5f | %.5f | %+.5f | %.2f | %d/%d | %.4f |",
          k, r$n_feat, mean(r$auc), sd(r$auc), mean(d), coh,
          sum(sign(d) == sign(d[1])), length(d), t.test(d)$p.value)
    }
  }
} else {
  add("_尚未运行 `R/09_ablation.R`。_")
}
add("")

# -----------------------------------------------------------------------------
# 四、调参
# -----------------------------------------------------------------------------
add("## 四、超参数搜索（**现役口径**：Tier A 20 万行，25 特征含编码）")
add("")
for (algo in c("xgboost", "lightgbm")) {
  f <- sprintf("output/tune_%s.rds", algo)
  if (!file.exists(f)) next
  tn <- readRDS(f)
  sc <- vapply(tn$all, function(z) z$auc, 0)
  add("### %s", algo)
  add("")
  add("搜索了 %d 组参数（3 折，20 万行）。", length(sc))
  add("")
  add("| 参数组 | AUC |")
  add("|---|---|")
  for (i in head(order(-sc), 8)) add("| %s | %.5f |", names(sc)[i], sc[i])
  add("")
  add("**最优**：`%s`，AUC %.5f", tn$best_name, max(sc))
  add("")
  if (length(tn$robustness)) {
    add("**稳健性检查** —— 用最优参数重跑四条插补线，看排序是否翻转：")
    add("")
    rb <- sort(unlist(tn$robustness), decreasing = TRUE)
    add("| 插补线 | AUC |")
    add("|---|---|")
    for (nm in names(rb)) add("| %s | %.5f |", nm, rb[nm])
    add("")
    add("排序：**%s**", paste(names(rb), collapse = " > "))
    add("")
  }
}
if (!file.exists("output/tune_xgboost.rds") &&
    !file.exists("output/tune_lightgbm.rds")) {
  add("_尚未运行 `R/10_tune.R`。_")
  add("")
}

# -----------------------------------------------------------------------------
# 五、alpha 敏感性
# -----------------------------------------------------------------------------
add("## 五、glmnet 的 alpha 敏感性（**现役口径**：Tier A 20 万行，25 特征含编码）")
add("")
if (file.exists("output/alpha_scan.rds")) {
  as_ <- readRDS("output/alpha_scan.rds")
  add("| alpha | 含义 | AUC | sd |")
  add("|---|---|---|---|")
  mean_desc <- c("0" = "纯 ridge", "0.25" = "偏 ridge", "0.5" = "弹性网各半",
                 "0.75" = "偏 lasso", "1" = "纯 lasso")
  for (a in names(as_)) {
    add("| %s | %s | %.5f | %.5f |", a,
        if (a %in% names(mean_desc)) mean_desc[[a]] else "",
        mean(as_[[a]]), sd(as_[[a]]))
  }
  rng <- diff(range(vapply(as_, mean, 0)))
  add("")
  add("极差 %.5f —— %s", rng,
      if (rng < 0.001) "alpha 的选择对本题几乎没有影响" else "alpha 有实质影响")
} else {
  add("_尚未运行 `R/09_alpha_scan.R`。_")
}
add("")

# -----------------------------------------------------------------------------
# 六、概率校准
# -----------------------------------------------------------------------------
add("## 六、概率校准（**现役口径**：Tier A 20 万行，25 特征含编码）")
add("")
if (file.exists("output/calibration.rds")) {
  cb <- readRDS("output/calibration.rds")
  add("| 模型 | AUC | Brier | ECE |")
  add("|---|---|---|---|")
  ord <- order(-vapply(cb, function(z) z$auc, 0))
  for (i in ord) {
    z <- cb[[i]]
    add("| `%s` | %.5f | %.5f | %.5f |", names(cb)[i], z$auc, z$brier, z$ece)
  }
  add("")
  add("参照点：全预测基准率 0.7094 时 Brier = 0.2062。")
  add("")
  add("**只有单模型输出才是概率。** 集成里的 rank 平均和爬山法输出的是秩，")
  add("线性压到 (0,1) 后形似概率但不具备概率含义，谈其校准没有意义。")
} else {
  add("_尚未运行 `R/09_calibration.R`。_")
}
add("")

# -----------------------------------------------------------------------------
# 七、Tier B 与集成
# -----------------------------------------------------------------------------
add("## 七、全量结果与集成（**交付口径**：691,369 行，25 特征，含逐取值编码）")
add("")
mb <- list.files("output/oof", pattern = "^meta_[^g]", full.names = TRUE)
if (length(mb)) {
  add("### 全量重训（69 万行）")
  add("")
  add("| 模型 | Tier A | Tier B | 差值 |")
  add("|---|---|---|---|")
  for (f in mb) {
    m <- readRDS(f)
    ga <- g[model == m$model, cv_mean]
    add("| `%s` | %s | %.5f | %s |", m$model,
        if (length(ga)) sprintf("%.5f", ga[1]) else "—",
        m$cv_mean,
        if (length(ga)) sprintf("%+.5f", m$cv_mean - ga[1]) else "—")
  }
  add("")
}
if (file.exists("output/ensemble_best.rds")) {
  en <- readRDS("output/ensemble_best.rds")
  add("### 集成方式对比")
  add("")
  add("| 方式 | CV AUC |")
  add("|---|---|")
  for (nm in names(en$all_scores))
    add("| %s | %.5f%s |", nm, en$all_scores[[nm]],
        if (nm == en$method) " ← 最优" else "")
  add("")
  add("最好的单模型 %.5f，集成提升 **%+.5f**。",
      max(en$single_aucs), en$cv_auc - max(en$single_aucs))
  add("")
  add("参与集成的模型：%s", paste(sprintf("`%s`", en$models), collapse = "、"))
} else {
  add("_尚未运行 `R/07_ensemble.R`。_")
}
add("")

# -----------------------------------------------------------------------------
# 八、样本量阶梯（子样本选择的有效性）
# -----------------------------------------------------------------------------
# 这一节是第一~六节那套「对比口径」本身的验证：在 20 万上做筛选合不合法。
# 顺带产出的 20 万那一级用的是同一套 25 特征，所以它也是
# 第一节那张编码前 Tier A 表的**新口径对应版本** —— 两张表不能相减，
# 但可以对照着看同一批格子在换了特征之后的位置变化。
add("## 八、样本量阶梯（**新增**：子样本选择的有效性）")
add("")
if (file.exists("output/size_ladder.rds")) {
  lad <- readRDS("output/size_ladder.rds")
  s <- lad$summary
  add("固定 25 特征、固定冻结折叠、池互相嵌套。详见 docs/实验报告.md 第九节。")
  add("")
  add("| 规模 | 行数 | Spearman | Kendall | 该级冠军 | 命中 | 选择遗憾 |")
  add("|---|---|---|---|---|---|---|")
  for (i in seq_len(nrow(s)))
    add("| %s | %s | %+.3f | %+.3f | `%s` | %s | **%.5f** |",
        s$rung[i], format(s$n[i], big.mark = ","), s$spearman[i], s$kendall[i],
        s$pick[i], if (s$hit[i]) "是" else "否", s$regret[i])
  add("")

  w <- lad$wide
  cols <- intersect(c("50k", "100k", "200k", "400k", "full"), names(w))
  add("### 逐格 AUC")
  add("")
  add("| 模型 | %s |", paste(cols, collapse = " | "))
  add("|---|%s", paste(rep("---|", length(cols)), collapse = ""))
  for (i in seq_len(nrow(w)))
    add("| `%s` | %s |", w$model[i],
        paste(sprintf("%.5f", unlist(w[i, ..cols])), collapse = " | "))
  add("")

  if (!is.null(lad$topk)) {
    add("### top-k 集合与全量的一致性")
    add("")
    add("| 规模 | top-1 | top-3 | top-5 |")
    add("|---|---|---|---|")
    for (i in seq_len(nrow(lad$topk)))
      add("| %s | %s | %d/3 | %d/5 |", lad$topk$rung[i],
          if (lad$topk$top1[i] == 1L) "是" else "否",
          lad$topk$top3[i], lad$topk$top5[i])
    add("")
  }
} else {
  add("_尚未运行 `R/25_size_ladder.R`。_")
  add("")
}

if (file.exists("output/ladder_pairs.rds")) {
  lp <- readRDS("output/ladder_pairs.rds")
  add("### 发生名次交换的配对：逐对实测分辨率下限")
  add("")
  add("下限是**那一对**的性质，必须逐对实测，不能互相套用（见 R/28_ladder_pairs.R）。")
  add("")
  add("| 配对 | 交换于 | 全量差 | 该对下限 | 判决 |")
  add("|---|---|---|---|---|")
  for (i in seq_len(nrow(lp)))
    add("| `%s` vs `%s` | %s | %.5f | %.5f | %s |",
        lp$a[i], lp$b[i], lp$rungs[i], lp$gap_full[i], lp$floor95[i], lp$verdict[i])
  add("")
}

if (file.exists("output/weight_transfer.rds")) {
  wt <- readRDS("output/weight_transfer.rds")
  add("### 集成权重的迁移")
  add("")
  add("两组权重都作用在同一份全量成员预测上。对「全量权重」有利，故为代价上界。")
  add("")
  add("| 评估集 | 20 万权重 | 全量权重 | 代价上界 |")
  add("|---|---|---|---|")
  for (i in seq_len(nrow(wt$res)))
    add("| %s | %.5f | %.5f | %+.5f |", wt$res$评估集[i],
        wt$res$AUC_20万权重[i], wt$res$AUC_全量权重[i], wt$res$迁移代价上界[i])
  add("")
  add("权重向量相关：Pearson %.3f，Spearman %.3f；符号一致的成员 %d 个。",
      wt$pearson, wt$spearman, wt$same_sign)
  add("")
}

writeLines(out, "output/results.md")
cat("已生成 output/results.md（", length(out), " 行）\n", sep = "")
