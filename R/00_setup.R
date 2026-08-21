# =============================================================================
# 00_setup.R —— 环境准备
#
# 作用：检查 R 版本、安装所有依赖包、报告环境状态。
# 用法：source("R/00_setup.R")
# 只需在第一次使用本项目时跑一次。
#
# 注意：本项目的包库钉在 D 盘（见项目根目录的 .Renviron），
#       避免往 C:\Users\<用户名>\AppData\Local\R 里写东西。
# =============================================================================

cat("========================================\n")
cat("  智能手机成瘾预测 —— 环境检查\n")
cat("========================================\n\n")

# ---- 1. R 版本 --------------------------------------------------------------
# 本项目在 R 4.6.1 上开发。低于 4.2 的版本在 Windows 上没有原生 UTF-8 支持，
# 中文注释会乱码，因此设为硬性下限。
r_ver <- getRversion()
cat("R 版本：", as.character(r_ver), "\n", sep = "")

if (r_ver < "4.2.0") {
  stop("R 版本过低（需要 >= 4.2.0）。Windows 上 4.2 以下没有原生 UTF-8 支持，中文会乱码。")
}

# ---- 2. 包库路径 ------------------------------------------------------------
cat("包安装路径：\n")
for (p in .libPaths()) cat("  - ", p, "\n", sep = "")

# 第一个路径是 install.packages() 的默认目标。检查它是否可写。
target_lib <- .libPaths()[1]
if (!dir.exists(target_lib)) {
  dir.create(target_lib, recursive = TRUE)
  cat("已创建包库目录：", target_lib, "\n", sep = "")
}
if (file.access(target_lib, mode = 2) != 0) {
  stop("包库目录不可写：", target_lib,
       "\n请以管理员身份运行 RStudio，或修改项目根目录的 .Renviron。")
}

# ---- 3. 依赖清单 ------------------------------------------------------------
# 分成两组：核心包缺一不可；可选包缺了会降级但不阻塞。
pkgs_core <- c(
  "data.table",   # 69 万行的读写与操作，base R 的 read.csv 在这个规模下不可用
  "xgboost",      # 梯度提升，支持原生 NaN
  "lightgbm",     # 梯度提升，支持原生 NaN，通常比 xgboost 快
  "ranger",       # 随机森林（比 randomForest 快一个数量级）
  "glmnet",       # 正则化 logistic 回归，可解释基线
  "pROC",         # 计算 AUC
  "missRanger"    # 基于 ranger 的链式随机森林插补，用于 L4 多重插补线
)

pkgs_optional <- c(
  "renv",         # 依赖版本锁定，用于复现
  "ggplot2",      # 画图
  "knitr",        # 报告渲染
  "rmarkdown"     # 报告渲染
)

# ---- 4. 安装 ----------------------------------------------------------------
options(repos = c(CRAN = "https://cloud.r-project.org"))

install_missing <- function(pkgs, label) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) {
    cat("[", label, "] 全部已安装\n", sep = "")
    return(character(0))
  }
  cat("[", label, "] 需要安装：", paste(missing, collapse = ", "), "\n", sep = "")
  # type = "binary" 强制走 CRAN 预编译包，避免在没装 Rtools 的机器上尝试源码编译。
  # 如果某个包没有对应 R 版本的二进制，这里会失败——那就是需要装 Rtools 的信号。
  for (p in missing) {
    cat("  正在安装 ", p, " ... ", sep = "")
    ok <- tryCatch({
      install.packages(p, type = "binary", quiet = TRUE)
      requireNamespace(p, quietly = TRUE)
    }, error = function(e) FALSE, warning = function(w) {
      requireNamespace(p, quietly = TRUE)
    })
    cat(if (isTRUE(ok)) "成功\n" else "失败\n")
  }
  # 返回装完之后仍然缺失的
  pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
}

cat("\n---- 安装依赖 ----\n")
still_missing_core <- install_missing(pkgs_core, "核心")
still_missing_opt  <- install_missing(pkgs_optional, "可选")

# ---- 5. 报告 ----------------------------------------------------------------
cat("\n---- 已安装版本 ----\n")
for (p in c(pkgs_core, pkgs_optional)) {
  if (requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("  %-12s %s\n", p, as.character(utils::packageVersion(p))))
  } else {
    cat(sprintf("  %-12s 未安装\n", p))
  }
}

cat("\n---- 数据文件 ----\n")
data_files <- c("train.csv", "test.csv", "sample_submission.csv")
for (f in data_files) {
  path <- file.path("data", "raw", f)
  if (file.exists(path)) {
    cat(sprintf("  %-24s %.1f MB\n", f, file.size(path) / 1024^2))
  } else {
    cat(sprintf("  %-24s 缺失 —— 请从 Kaggle 竞赛页下载后放入 data/raw/\n", f))
  }
}

cat("\n========================================\n")
if (length(still_missing_core) > 0) {
  cat("  环境未就绪\n")
  cat("========================================\n")
  cat("以下核心包安装失败：", paste(still_missing_core, collapse = ", "), "\n\n", sep = "")
  cat("最可能的原因是 CRAN 还没有为当前 R 版本提供预编译包。\n")
  cat("解决办法：安装与本机 R 主版本匹配的 Rtools，然后改用源码安装：\n")
  cat('  install.packages(c("', paste(still_missing_core, collapse = '", "'), '"), type = "source")\n', sep = "")
  cat("Rtools 下载页：https://cran.r-project.org/bin/windows/Rtools/\n")
} else {
  cat("  环境就绪\n")
  cat("========================================\n")
  cat("下一步：\n")
  cat('  source("R/01_load.R")      # 读数据\n')
  cat('  source("R/03_features.R")  # 特征工程\n')
  cat('  source("R/04_folds.R")     # 生成折叠契约（全组只跑一次）\n')
}
cat("\n")
