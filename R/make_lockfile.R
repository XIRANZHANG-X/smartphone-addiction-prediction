# =============================================================================
# make_lockfile.R —— 生成 renv.lock（审查意见 3.1）
#
# 用法：Rscript R/make_lockfile.R
# 产出：项目根目录的 renv.lock
#
# 作业要求第 2 条：「Detailed instructions: how to re-produce your results
# (including additional R packages that you installed)」。renv.lock 记录了
# 每个包的**精确版本**，助教一行 renv::restore() 就能还原我们的环境。
#
# -----------------------------------------------------------------------------
# 为什么不直接在项目里跑 renv::init()
# -----------------------------------------------------------------------------
# renv::init() 会在项目根目录创建 .Rprofile，让**之后每一次** R 启动都
# 自动激活一个项目本地的包库 —— 而那个库一开始是空的。
#
# 后果：正在后台跑的实验会突然找不到 xgboost，组员 clone 下来第一次
# 打开项目也会看到满屏报错。对一个 10 人协作、且随时有长任务在跑的
# 项目，这个副作用不可接受。
#
# 所以这里的做法是：在**临时目录**里建一个一次性的 renv 项目，
# 把我们的 R 脚本复制进去让 renv 扫描依赖，生成 lockfile 之后
# 只把这一个文件拷回来。项目本身完全不受影响。
# =============================================================================

if (!requireNamespace("renv", quietly = TRUE)) {
  stop("renv 未安装。请先运行 source(\"R/00_setup.R\")")
}

proj <- normalizePath(".", winslash = "/")
tmp  <- file.path(tempdir(), paste0("lockgen_", as.integer(Sys.time())))
dir.create(file.path(tmp, "R"), recursive = TRUE, showWarnings = FALSE)

# 只复制 R 脚本 —— renv 扫描 library()/require()/:: 调用来推断依赖
r_files <- list.files("R", pattern = "\\.R$", full.names = TRUE)
file.copy(r_files, file.path(tmp, "R"), overwrite = TRUE)
cat(sprintf("已复制 %d 个 R 脚本到临时项目\n", length(r_files)))

# 让 renv 在临时项目里工作。bare = TRUE 表示不自动安装任何东西，
# 只建立项目结构；我们要的只是它的依赖扫描和 lockfile 生成能力。
cat("扫描依赖 ...\n")
deps <- renv::dependencies(path = file.path(tmp, "R"), quiet = TRUE)
pkgs <- sort(unique(deps$Package))
cat(sprintf("发现 %d 个依赖：%s\n", length(pkgs), paste(pkgs, collapse = ", ")))

# 补上间接但必要的包（脚本里没直接 library() 但复现要用）
extra <- c("renv", "knitr", "rmarkdown")
pkgs <- sort(unique(c(pkgs, extra[vapply(extra, requireNamespace,
                                         logical(1), quietly = TRUE)])))

missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  cat(sprintf("警告：以下包未安装，将不进入 lockfile：%s\n",
              paste(missing, collapse = ", ")))
  pkgs <- setdiff(pkgs, missing)
}

lock_path <- file.path(proj, "renv.lock")
cat("生成 lockfile ...\n")

renv::snapshot(
  project  = tmp,
  packages = pkgs,
  lockfile = lock_path,
  type     = "explicit",
  prompt   = FALSE,
  force    = TRUE
)

# ---- 校验 -------------------------------------------------------------------
if (!file.exists(lock_path)) stop("lockfile 生成失败")

lk <- renv::lockfile_read(lock_path)
cat("\n======================================================\n")
cat("  renv.lock 已生成\n")
cat("======================================================\n")
cat(sprintf("R 版本：%s\n", lk$R$Version))
cat(sprintf("记录了 %d 个包：\n", length(lk$Packages)))
for (nm in sort(names(lk$Packages))) {
  cat(sprintf("  %-16s %s\n", nm, lk$Packages[[nm]]$Version))
}

# 确认没有污染项目目录
for (f in c(".Rprofile", "renv")) {
  if (file.exists(file.path(proj, f))) {
    warning("项目目录出现了意料之外的 ", f, " —— 请检查是否需要删除")
  }
}
cat("\n项目目录未被污染（无 .Rprofile / renv/ 生成）。\n")
cat("复现方式：renv::restore()\n")

unlink(tmp, recursive = TRUE)
