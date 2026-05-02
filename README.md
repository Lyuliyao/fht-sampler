# MetaTensor GitHub Subset

这是从原始 `meta_tensor/` 工作目录中整理出的 GitHub 版本，目标是保留可以公开分享的代码、模板、notebook 和代表性结果，同时排除体量很大的运行中间文件与轨迹数据。

这个仓库适合：

- 展示实验组织方式和参数配置
- 保留画图脚本、notebook 和模板输入文件
- 分享论文或汇总图
- 作为后续正式 GitHub 仓库的起点

这个仓库不等同于完整原始数据归档。很多用于真正重跑分析的重数据文件已经被刻意省略。

## 保留内容

- `1UAO_3/`
  - `case0.5/`、`case2/` 的 `config.py`、`template/`、部分 notebook 与流程脚本
  - `figures/` 中的整理后图件
- `1VII2/`
  - 多个 case 的 `config.py`、`template/`、流程脚本与分析 notebook
- `ala2_gaussian/`
  - `case7` 到 `case9` 的配置、模板、figure notebook
  - `compare/` 下保留轻量输入文件，不保留 `sim_*` 轨迹目录
- `ala4_gaussian/`
  - 主要 case 的配置、模板、流程脚本
  - `compare/simulation/` 与 `simulation2/` 下保留轻量输入文件
- `chi.1/`、`chi.3/`
  - case 配置、模板、流程脚本
  - `compare/` 下仅保留轻量输入文件
- `muller/`
  - `Code/` 中的画图与 notebook
  - `Data/` 中的 `summary.txt` 和图像文件
  - 不保留大型 `.npz`
- `result/`
  - 汇总图、PDF、画图脚本和 notebook

## 刻意省略的内容

以下内容没有放进这个 GitHub 子集，或者已通过 `.gitignore` 设为默认不跟踪：

- GROMACS / PLUMED 运行输出：`*.edr`、`*.tpr`、`*.trr`、`*.xtc`、`*.cpt`
- 大量并行采样目录：`run*/`、`w*/`、`sim_*/`
- 聚合后的大文本或大数组：`ALL_COLVAR*`、`COLVAR*`、`results.npz`、`unbiased_traj.npz`
- 编译产物：`*.o`、`*.so`
- 缓存与系统垃圾文件：`__pycache__/`、`.ipynb_checkpoints/`、`.DS_Store`、`._*`

## 目录定位

如果你是给别人看代码和结果，优先看这些位置：

- `result/figure.py`
- `result/figure_1uao.py`
- `result/figure.ipynb`
- `1UAO_3/figures/`
- `muller/Code/`

如果你是想看每个体系的输入模板和参数，优先看：

- `*/case*/config.py`
- `*/case*/template/`
- `sequence` / `sequence_rerun` / `collect_data`

## 复现说明

当前仓库主要保证“结构可读、结果可展示、脚本可追溯”，不保证在没有原始大数据目录的情况下直接完整重跑全部图件。部分 notebook 和脚本仍然会引用被省略的原始 `run*`、`w*`、`sim_*`、`COLVAR` 或 `.npz` 数据。

如果后续需要做完整复现，建议把原始重数据单独存放在本地磁盘、服务器路径或对象存储中，而不是直接进 Git。

## 建议依赖

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `pillow`
- `jupyter`

部分脚本还会依赖：

- GROMACS
- PLUMED
- PyMOL
- LaTeX

## 上传到 GitHub

如果你准备把这个整理后的目录单独建仓库，可以直接在本目录执行：

```bash
git init
git add .
git commit -m "Initial curated MetaTensor subset"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

推荐只把当前这个整理后的目录作为仓库根目录上传，而不是把原始 `meta_tensor/` 整体上传。
