# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
import os
import subprocess
import tempfile
import textwrap
os.environ["PATH"] = "/apps/spack/anvil/apps/texlive/20200406-gcc-11.2.0-eeavxnm/bin/x86_64-linux:/Library/TeX/texbin:" + os.environ.get("PATH","")
from scipy.stats import gaussian_kde
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # ---------- LaTeX ----------
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    # ---------- Font sizes ----------
    "font.size": 24,            # 全局默认字号
    "axes.labelsize": 24,       # 坐标轴标题
    "axes.titlesize": 24,       # 子图标题
    "xtick.labelsize": 24,      # x 轴刻度
    "ytick.labelsize": 24,      # y 轴刻度
    "legend.fontsize": 24,      # 图例字号
    "legend.title_fontsize": 24,

    # ---------- Axes ----------
    "axes.linewidth": 1.2,      # 坐标轴边框线宽
    "axes.unicode_minus": False,

    # ---------- Lines ----------
    "lines.linewidth": 2.0,
    "lines.markersize": 8,

    # ---------- Ticks ----------
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # ---------- Legend ----------
    "legend.frameon": False,    # 论文图通常不用边框
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,

    # ---------- Figure ----------
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # ---------- PDF/PS ----------
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# %%
kB = 8.314462618e-3  # kJ/(mol*K)
T = 300.0
def compute_fes(data, N, bin_edges,weight=None):
    # print(data.shape)
    if weight is None:
        weight = np.ones(data[:N].shape[0])
    hist, _ = np.histogram(data[:N], bins=bin_edges, density=True,weights=np.exp(weight[:N]/kB/T))
    fes = -np.log(hist + 1e-10) * kB * T  # kJ/mol
    fes -= fes.min()  # optional: shift each FES so minimum is 0
    return fes

# %%
bias_alpha_05 = [
    np.loadtxt(f"../ala2_gaussian/case7/finial_all/w{i+1:02d}/COLVAR.{i}")
    for i in range(32)
]
bias_alpha_05_array = np.array(bias_alpha_05)
bias_alpha_10 = [
    np.loadtxt(f"../ala2_gaussian/case8/finial_all/w{i+1:02d}/COLVAR.{i}")
    for i in range(32)
]
bias_alpha_10_array = np.array(bias_alpha_10)
bias_alpha_20 = [
    np.loadtxt(f"../ala2_gaussian/case9/finial_all/w{i+1:02d}/COLVAR.{i}")
    for i in range(32)
]
bias_alpha_20_array = np.array(bias_alpha_20)
unbias_data = [
    np.loadtxt(f"../ala2_gaussian/compare/simulation/sim_{i:03d}/COLVAR")
    for i in range(32)
]
unbias_data_array = np.array(unbias_data)


# %%


# %%
fig, axes = plt.subplots(
    2, 2, figsize=(8, 8), sharex=True, sharey=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)
axes_flat = axes.ravel()

bin_edges = np.linspace(-np.pi, np.pi, 101)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

sampling_sizes = [10000, 20000, 40000, 80000]
panel_labels = [f'(a) {0.001*10000*50} ps' , f'(b) {0.001*20000*50} ps',\
                f'(c){0.001*40000*50} ps', f'(d) {0.001*80000*50} ps']

all_ymins = []
all_ymaxs = []

for i, (ax, sampling_size) in enumerate(zip(axes_flat, sampling_sizes)):
    fes_all = np.array([compute_fes(d[:,1], sampling_size, bin_edges) for d in unbias_data])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label="unbiased MD", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,1], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_05])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=0.5$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,1], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_10])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=1.0$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,1], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_20])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=2.0$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    # panel label
    ax.text(
        0.03, 0.95, panel_labels[i],
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=24
    )

    # panel title at a unified position
    # ax.set_title(rf"$N={sampling_size}$", pad=-30)

    # only keep outer tick labels
    ax.label_outer()

# unified y-range
ymin = min(all_ymins)
ymax = max(all_ymaxs)
ypad = 0.05 * (ymax - ymin)

for ax in axes_flat:
    ax.set_ylim(ymin - ypad, ymax + ypad)

# optional: unified ticks
for ax in axes_flat:
    ax.set_xlim(-np.pi, np.pi)
    ax.tick_params(axis='both', labelsize=24)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=4,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.0
)

plt.subplots_adjust(top=0.88)
plt.savefig("ala2_phi_converge.pdf")

# %%


# %%
fig, axes = plt.subplots(
    2, 2, figsize=(8, 8), sharex=True, sharey=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)
axes_flat = axes.ravel()

bin_edges = np.linspace(-np.pi, np.pi, 101)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

sampling_sizes = [10000, 20000, 40000, 80000]
panel_labels = [f'(a) {0.001*10000*50} ps' , f'(b) {0.001*20000*50} ps',\
                f'(c){0.001*40000*50} ps', f'(d) {0.001*80000*50} ps']

all_ymins = []
all_ymaxs = []

for i, (ax, sampling_size) in enumerate(zip(axes_flat, sampling_sizes)):
    fes_all = np.array([compute_fes(d[:,2], sampling_size, bin_edges) for d in unbias_data])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label="unbiased MD", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,2], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_05])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=0.5$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,2], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_10])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=1.0$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    fes_all = np.array([compute_fes(d[:,2], sampling_size, bin_edges, d[:,-2]) for d in bias_alpha_20])
    fes_mean = fes_all.mean(axis=0)
    fes_std = fes_all.std(axis=0)
    ax.plot(centers, fes_mean, label=r"$\alpha=2.0$", lw=2)
    ax.fill_between(centers, fes_mean-fes_std, fes_mean+fes_std, alpha=0.35)
    all_ymins.append(np.min(fes_mean - fes_std))
    all_ymaxs.append(np.max(fes_mean + fes_std))

    # panel label
    ax.text(
        0.03, 0.95, panel_labels[i],
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=24
    )

    # panel title at a unified position
    # ax.set_title(rf"$N={sampling_size}$", pad=-30)

    # only keep outer tick labels
    ax.label_outer()

# unified y-range
ymin = min(all_ymins)
ymax = max(all_ymaxs)
ypad = 0.05 * (ymax - ymin)

for ax in axes_flat:
    ax.set_ylim(ymin - ypad, ymax + ypad)

# optional: unified ticks
for ax in axes_flat:
    ax.set_xlim(-np.pi, np.pi)
    ax.tick_params(axis='both', labelsize=24)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=4,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.0
)

plt.subplots_adjust(top=0.88)
plt.savefig("ala2_psi_converge.pdf")

# %%

fig, axes = plt.subplots(
    2, 4, figsize=(16, 8), sharex=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)

bin_edges = np.linspace(-np.pi, np.pi, 101)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
sampling_sizes = [10000, 20000, 40000, 80000]

panel_labels = [
    "(a) 500 ps", "(b) 1000 ps", "(c) 2000 ps", "(d) 4000 ps",
    "(e) 500 ps", "(f) 1000 ps", "(g) 2000 ps", "(h) 4000 ps"
]

# 两行分别对应 d[:,1] 和 d[:,2]
col_indices = [1, 2]
row_names = [r"$F(\phi)\,[\mathrm{kJ/mol}]$", r"$F(\psi)\,[\mathrm{kJ/mol}]$"]

# 用来存每一行的 y 范围
row_ymins = [[], []]
row_ymaxs = [[], []]

for row, col_idx in enumerate(col_indices):
    for col, sampling_size in enumerate(sampling_sizes):
        ax = axes[row, col]

        # unbiased
        fes_all = np.array([compute_fes(d[:, col_idx], sampling_size, bin_edges) for d in unbias_data])
        fes_mean = fes_all.mean(axis=0)+30
        fes_std = fes_all.std(axis=0)
        ax.plot(centers, fes_mean, label="unbiased MD", lw=2)
        ax.fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
        row_ymins[row].append(np.min(fes_mean - fes_std))
        row_ymaxs[row].append(np.max(fes_mean + fes_std))

        # alpha = 0.5
        fes_all = np.array([compute_fes(d[:, col_idx], sampling_size, bin_edges, d[:, -2]) for d in bias_alpha_05])
        fes_mean = fes_all.mean(axis=0)+20
        fes_std = fes_all.std(axis=0)
        ax.plot(centers, fes_mean, label=r"$\alpha=0.5$", lw=2)
        ax.fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
        row_ymins[row].append(np.min(fes_mean - fes_std))
        row_ymaxs[row].append(np.max(fes_mean + fes_std))

        # alpha = 1.0
        fes_all = np.array([compute_fes(d[:, col_idx], sampling_size, bin_edges, d[:, -2]) for d in bias_alpha_10])
        fes_mean = fes_all.mean(axis=0)+10
        fes_std = fes_all.std(axis=0)
        ax.plot(centers, fes_mean, label=r"$\alpha=1.0$", lw=2)
        ax.fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
        row_ymins[row].append(np.min(fes_mean - fes_std))
        row_ymaxs[row].append(np.max(fes_mean + fes_std))

        # alpha = 2.0
        fes_all = np.array([compute_fes(d[:, col_idx], sampling_size, bin_edges, d[:, -2]) for d in bias_alpha_20])
        fes_mean = fes_all.mean(axis=0)
        fes_std = fes_all.std(axis=0)
        ax.plot(centers, fes_mean, label=r"$\alpha=2.0$", lw=2)
        ax.fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
        row_ymins[row].append(np.min(fes_mean - fes_std))
        row_ymaxs[row].append(np.max(fes_mean + fes_std))

        # panel label
        ax.text(
            0.03, 0.95, panel_labels[row * 4 + col],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=24
        )

        # 每一行左边加 y-label
        if col == 0:
            ax.set_ylabel(row_names[row], fontsize=24)

        ax.label_outer()
        ax.tick_params(axis='both', labelsize=24)
        ax.set_xlim(-np.pi, np.pi)

# 每一行统一 y 轴范围
for row in range(2):
    ymin = min(row_ymins[row])
    ymax = max(row_ymaxs[row])
    ypad = 0.05 * (ymax - ymin)
    for col in range(4):
        axes[row, col].set_ylim(ymin - ypad, ymax + ypad)

for col in range(4):
    axes[1, col].set_xlabel("Dihedral angle", fontsize=24)
                            
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=4,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.0
)

plt.subplots_adjust(top=0.88)
plt.savefig("ala2_fes_convergence_2x4.pdf")
plt.savefig("ala2_fes_convergence_2x4.png", dpi=300)

# %%
# sampling_sizes = np.array([10000, 20000, 40000, 80000,120000,160000,200000,240000,280000,320000,360000,400000,440000,480000])  # 和你2x4一致
# col_indices = [1, 2]
# row_titles = [r"$\phi$", r"$\psi$"]

# # 这里把各组数据打包：name -> (runs, has_weight)
# groups = {
#     "unbiased MD": (unbias_data, False),
#     r"$\alpha=0.5$": (bias_alpha_05, True),
#     r"$\alpha=1.0$": (bias_alpha_10, True),
#     r"$\alpha=2.0$": (bias_alpha_20, True),
# }

# def valid_mask(runs, idx, N, threshold=1e-6):
#     """用跨-run 平均(加权)直方图筛掉几乎空的 bins，避免 log 放大尾部噪声。"""
#     h_all = []
#     for d in runs:
#         if d.shape[0] < N:
#             raise ValueError(f"Trajectory shorter than N={N}")
#         if d.shape[1] < 3:
#             raise ValueError("d seems too narrow; check indices")
#         if threshold is None:
#             return slice(None)

#         if d.shape[1] >= 2 and True:
#             if d.shape[1] >= 2:
#                 if d.shape[1] >= 2:
#                     pass

#         # weights: only if has_weight; caller decides by passing runs already
#         # here we detect if last-2 column exists and use it if present (works for biased data)
#         if d.shape[1] >= 2 and d.shape[1] >= 2:
#             if d.shape[1] >= 2:
#                 pass

#         # if runs are biased, assume d[:, -2] exists
#         # if not, use uniform weights
#         if d.shape[1] >= 2 and d.shape[1] >= 2:
#             pass

#         if d.shape[1] >= 2 and d.shape[1] >= 2:
#             pass

#         if d.shape[1] >= 2 and d.shape[1] >= 2:
#             pass

#         # robust detection: if last-2 is available, treat as bias (as in your compute_fes usage)
#         if d.shape[1] >= 2 and (d.shape[1] >= 2):
#             # if we *intend* unbiased, caller will not use this branch by passing has_weight=False later
#             pass

#     # We'll implement mask outside (needs has_weight). Keep simple below.
#     raise RuntimeError("Use valid_mask_by_group instead.")

# def valid_mask_by_group(runs, idx, N, has_weight, threshold=1e-6):
#     h_all = []
#     for d in runs:
#         if has_weight:
#             w = np.exp(d[:N, -2] / (kB * T))
#             h, _ = np.histogram(d[:N, idx], bins=bin_edges, density=True, weights=w)
#         else:
#             h, _ = np.histogram(d[:N, idx], bins=bin_edges, density=True)
#         h_all.append(h)
#     h_mean = np.mean(h_all, axis=0)
#     return h_mean > threshold

# def mean_std_across_bins(runs, idx, N, has_weight, threshold=1e-6):
#     fes_all = np.array([
#         compute_fes(d[:, idx], N, bin_edges, d[:, -2]) if has_weight
#         else compute_fes(d[:, idx], N, bin_edges)
#         for d in runs
#     ])
#     std_bin = fes_all.std(axis=0)

#     mask = valid_mask_by_group(runs, idx, N, has_weight, threshold=threshold)
#     return std_bin[mask].mean()

# def tail_slope(Ns, ys, tail=4):
#     logN = np.log(Ns[-tail:].astype(float))
#     logy = np.log(np.array(ys[-tail:], dtype=float))
#     m, b = np.polyfit(logN, logy, 1)
#     return m, b

# # ---- plot convergence for phi and psi separately ----
# for idx, title in zip(col_indices, row_titles):
#     fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.grid(alpha=0.25)
#     ax.set_xlabel("N (sampling size)")
#     ax.set_ylabel(r"Mean std across bins (kJ/mol)")
#     ax.set_title(fr"Convergence of FES uncertainty for {title}")

#     for name, (runs, has_weight) in groups.items():
#         vals = []
#         for N in sampling_sizes:
#             vals.append(mean_std_across_bins(runs, idx, int(N), has_weight, threshold=1e-6))

#         m, b = tail_slope(sampling_sizes, vals, tail=min(4, len(sampling_sizes)))

#         ax.plot(
#             sampling_sizes, vals,
#             marker="o", lw=2,
#             label=fr"{name})"
#         )

#     ax.legend(frameon=False)
#     plt.tight_layout()
#     plt.show()

# %%

# 读数据
bias_alpha_05_run = [
    np.loadtxt(f"../ala2_gaussian/case7/run{i}/COLVAR")
    for i in range(1, 20)
]
bias_alpha_10_run = [
    np.loadtxt(f"../ala2_gaussian/case8/run{i}/COLVAR")
    for i in range(1, 20)
]
bias_alpha_20_run = [
    np.loadtxt(f"../ala2_gaussian/case9/run{i}/COLVAR")
    for i in range(1, 20)
]

# 拼接 bias
phi_05 = np.concatenate([d[:, 1] for d in bias_alpha_05_run])
phi_10 = np.concatenate([d[:, 1] for d in bias_alpha_10_run])
phi_20 = np.concatenate([d[:, 1] for d in bias_alpha_20_run])

# unbiased 只取到和 bias 相同长度
target_len = len(phi_05)
phi_unbiased_all = np.concatenate([d[:, 1] for d in unbias_data[:3]])
phi_unbiased = phi_unbiased_all[:target_len]


# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde

# -------- inputs --------
img = Image.open("ala2.png")

dt = 0.001 * 50  # ps
datasets = [
    (phi_unbiased, "Unbiased MD"),
    (phi_05, r"$\alpha=0.5$"),
    (phi_10, r"$\alpha=1.0$"),
    (phi_20, r"$\alpha=2.0$"),
]

# -------- figure layout --------
fig = plt.figure(figsize=(16, 8))

# 外层：左列（示意图 + KDE） vs 右边 2x2 dynamics
outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2.0], wspace=0.22)

# 左列再拆成上下两块
left_gs = GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer_gs[0, 0],
    height_ratios=[1.5, 2.0], hspace=0.18
)

# (a) 左上示意图
ax_img = fig.add_subplot(left_gs[0, 0])
ax_img.imshow(img)
ax_img.axis("off")
ax_img.text(0.03, 0.97, "(a)", transform=ax_img.transAxes,
            ha="left", va="top", fontsize=24)

# (b) 左下 KDE 密度图
ax_kde = fig.add_subplot(left_gs[1, 0])
x_grid = np.linspace(-np.pi, np.pi, 1000)

# KDE bandwidth：你原来用 0.15，这里保留
bw = 0.15
for data, label in datasets:
    kde = gaussian_kde(data[::100], bw_method=bw)
    y = kde(x_grid)
    ax_kde.plot(x_grid, y, linewidth=2, label=label)
    ax_kde.fill_between(x_grid, y, alpha=0.25)

ax_kde.set_xlim(-np.pi, np.pi)
ax_kde.set_xlabel(r"$\phi$")
ax_kde.set_ylabel("Density")
ax_kde.legend(frameon=False, fontsize=10, loc="best")
ax_kde.text(0.03, 0.95, "(b)", transform=ax_kde.transAxes,
            ha="left", va="top")

# 右边内部 2x2，内部保持无缝
inner_gs = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer_gs[0, 1], wspace=0, hspace=0
)

ax1 = fig.add_subplot(inner_gs[0, 0])
ax2 = fig.add_subplot(inner_gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(inner_gs[1, 0], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(inner_gs[1, 1], sharex=ax1, sharey=ax1)
axes = [ax1, ax2, ax3, ax4]

# 统一 y 轴范围
all_phi = np.concatenate([d for d, _ in datasets])
ymin, ymax = all_phi.min(), all_phi.max()
ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0

# 右侧面板标签：顺延为 (c)-(f)
panel_labels = ["(c)", "(d)", "(e)", "(f)"]

for ax, (phi, title), label in zip(axes, datasets, panel_labels):
    time = dt * np.arange(phi.shape[0])
    ax.plot(time[::10], phi[::10], ".", ms=0.2)

    ax.text(0.03, 0.95, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=24)
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=20)

    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.label_outer()

ax3.set_xlabel("Time [ps]")
ax4.set_xlabel("Time [ps]")
ax1.set_ylabel(r"$\phi$")
ax3.set_ylabel(r"$\phi$")

# 你的刻度设置保留
ax3.set_xticks([0, 25000, 50000])

plt.savefig("ala2_exploration.png", bbox_inches="tight", dpi=300)
plt.savefig("ala2_exploration.pdf", bbox_inches="tight")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.figure(figsize=(8,6))

datasets = [
    (phi_unbiased[::100], "Unbiased MD"),
    (phi_05[::100], r'$\alpha=0.5$'),
    (phi_10[::100], r'$\alpha=1.0$'),
    (phi_20[::100], r'$\alpha=2.0$'),
]

x_grid = np.linspace(-np.pi, np.pi, 1000)

for data, label in datasets:
    kde = gaussian_kde(data, bw_method=0.15)  # bw越小越尖锐，可调
    y = kde(x_grid)

    plt.plot(x_grid, y, linewidth=2, label=label)
    plt.fill_between(x_grid, y, alpha=0.25)

plt.xlabel(r'$\phi$')
plt.ylabel('Density')
plt.xlim(-np.pi, np.pi)
plt.legend()
plt.tight_layout()
plt.show()

# %%
bias_alpha_10_run = [
    np.loadtxt(f"../ala4_gaussian/case_alpha_1_tau_0.5_eps_0.05_deg_20/run{i}/COLVAR")
    for i in range(1, 20)
]
bias_alpha_20_run = [
    np.loadtxt(f"../ala4_gaussian/case_alpha_2_tau_0.5_eps_0.05_deg_20/run{i}/COLVAR")
    for i in range(1, 20)
]
unbias_data = [
    np.loadtxt(f"../ala4_gaussian/simulation2/sim_{i:03d}/COLVAR")
    for i in range(10)
]

# %%


# 拼接 bias
phi1_10 = np.concatenate([d[:, 1] for d in bias_alpha_10_run])
phi1_20 = np.concatenate([d[:, 1] for d in bias_alpha_20_run])
target_len = len(phi1_10)
phi1_unbiased_all = np.concatenate([d[:, 1] for d in unbias_data[:3]])
phi1_unbiased = phi1_unbiased_all[:target_len]

phi2_10 = np.concatenate([d[:, 3] for d in bias_alpha_10_run])
phi2_20 = np.concatenate([d[:, 3] for d in bias_alpha_20_run])
phi2_unbiased_all = np.concatenate([d[:, 3] for d in unbias_data[:3]])
phi2_unbiased = phi2_unbiased_all[:target_len]


psi1_10 = np.concatenate([d[:, 2] for d in bias_alpha_10_run])
psi1_20 = np.concatenate([d[:, 2] for d in bias_alpha_20_run])

psi1_unbiased_all = np.concatenate([d[:, 2] for d in unbias_data[:3]])
psi1_unbiased = psi1_unbiased_all[:target_len]




# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

img = Image.open("ala4_rm.png")

fig = plt.figure(figsize=(16, 8))

# 外层：左图 vs 右边整个 dynamics 区域
outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2.0], wspace=0.2)

# 左边示意图
ax_img = fig.add_subplot(outer_gs[0, 0])
ax_img.imshow(img)
ax_img.axis("off")
ax_img.text(0.03, 0.97, "(a)", transform=ax_img.transAxes, ha="left", va="top", fontsize=24)

# 右边内部 2x2，内部保持无缝
inner_gs = GridSpecFromSubplotSpec(
    3, 3, subplot_spec=outer_gs[0, 1], wspace=0, hspace=0
)

ax1 = fig.add_subplot(inner_gs[0, 0])
ax2 = fig.add_subplot(inner_gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(inner_gs[0, 2], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(inner_gs[1, 0], sharex=ax1, sharey=ax1)
ax5 = fig.add_subplot(inner_gs[1, 1], sharex=ax1, sharey=ax1)
ax6 = fig.add_subplot(inner_gs[1, 2], sharex=ax1, sharey=ax1)
ax7 = fig.add_subplot(inner_gs[2, 0], sharex=ax1, sharey=ax1)
ax8 = fig.add_subplot(inner_gs[2, 1], sharex=ax1, sharey=ax1)
ax9 = fig.add_subplot(inner_gs[2, 2], sharex=ax1, sharey=ax1)
axes = [ax1, ax2, ax3, ax4,ax5, ax6, ax7, ax8,ax9]

dt = 0.0005 * 200

datasets = [
    (phi1_unbiased, "Unbiased MD" ),
    (phi1_10, r"$\alpha=1.0$"),
    (phi1_20, r"$\alpha=2.0$"),
    (phi2_unbiased, "Unbiased MD" ),
    (phi2_10, r"$\alpha=1.0$"),
    (phi2_20, r"$\alpha=2.0$"),
    (psi1_unbiased, "Unbiased MD" ),
    (psi1_10, r"$\alpha=1.0$"),
    (psi1_20, r"$\alpha=2.0$"),
]

panel_labels = ["(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)"]

# 统一 y 轴范围
ymin, ymax = -np.pi,np.pi
ypad = 0.05 * (ymax - ymin)
index = 0 
for ax, (phi,  title), label in zip(axes, datasets, panel_labels):
    time = dt * np.arange(phi.shape[0])
    ax.plot(time, phi, ".", ms=0.2)
    if index <3:
        ax.text(0.5, 0.85, title, transform=ax.transAxes, ha="center", va="top")
    ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top")
    index += 1
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.label_outer()

ax3.set_xlabel("Time [ps]")
ax4.set_xlabel("Time [ps]")
ax1.set_ylabel(r"$\phi_1$")
ax4.set_ylabel(r"$\phi_2$")
ax7.set_ylabel(r"$\psi_1$")
ax3.set_xticks([0,15000,30000])
plt.savefig("ala4_exploration.png", bbox_inches="tight")
plt.savefig("ala4_exploration.pdf", bbox_inches="tight")

# %%
bias_alpha_40_run = [
    np.loadtxt(f"../ala4_gaussian/case_alpha_2_tau_0.5_eps_0.05_deg_20/finial_all/w{i+1:02d}/COLVAR")
    for i in range(32)
]

# %%

sampling_sizes = [10000, 20000, 40000, 80000]

idx_list = [1, 2, 3, 4]
titles = [r"$\phi_1$", r"$\psi_1$", r"$\phi_2$", r"$\psi_2$"]
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(
    1, 4, figsize=(16, 4),
    sharex=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)

row_ymins = []
row_ymaxs = []

for ax, idx, title, plab in zip(axes, idx_list, titles, panel_labels):
    index = 0
    for sampling_size in sampling_sizes:

        fes_all = np.array([
            compute_fes(d[:, idx], sampling_size, bin_edges, d[:, -2])
            for d in bias_alpha_40_run
        ])
        fes_mean = fes_all.mean(axis=0)+index
        fes_std  = fes_all.std(axis=0)
        index += 10
        ax.plot(centers, fes_mean,  lw=2, label=f"{sampling_size*0.0005*200} ps")
        ax.fill_between(
            centers,
            fes_mean - fes_std,
            fes_mean + fes_std,
            alpha=0.25
        )

        row_ymins.append(np.min(fes_mean - fes_std))
        row_ymaxs.append(np.max(fes_mean + fes_std))

    # panel label + title
    ax.text(0.03, 0.95, plab, transform=ax.transAxes,
            ha="left", va="top", fontsize=18)
    ax.text(0.50, 0.95, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=18)

    ax.label_outer()
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(-np.pi, np.pi)

# 统一 y 轴（全局）
ymin = min(row_ymins)
ymax = max(row_ymaxs)
ypad = 0.05 * (ymax - ymin + 1e-12)
for ax in axes:
    ax.set_ylim(ymin - ypad, ymax + ypad)

# 只在最左边显示 y label / y tick label
axes[0].set_ylabel(r"$F$ (kJ/mol)", fontsize=18)
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# x label（底部统一）
for ax in axes:
    ax.set_xlabel("Dihedral angle", fontsize=18)

# 全局 legend（顶部居中），只显示 sampling size
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=4,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.0,
    title_fontsize=16,
    fontsize=14
)

plt.subplots_adjust(top=0.82)  # 给顶部 legend 留空间
plt.savefig("ala4_sampling_compare_1x4.pdf")
plt.savefig("ala4_sampling_compare_1x4.png", dpi=300)
plt.show()

# %%

kB = 8.314462618e-3
T = 300.0

def compute_fes_2d(data1, data2, N, bin_edges, weight=None):
    if weight is None:
        weight = np.ones(len(data1[:N]))

    hist, xedges, yedges = np.histogram2d(
        data1[:N], data2[:N],
        bins=bin_edges,
        density=True,
        weights=np.exp(weight[:N] / (kB * T))
    )

    fes = -np.log(hist + 1e-10) * kB * T
    fes -= np.nanmin(fes)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    return fes.T, xcenters, ycenters


def mean_fes_2d_all_runs(runs, idx1, idx2, N, bin_edges):
    fes_all = []
    for d in runs:
        fes, x, y = compute_fes_2d(
            d[:, idx1],
            d[:, idx2],
            N,
            bin_edges,
            d[:, -2]
        )
        fes_all.append(fes)

    fes_all = np.array(fes_all)
    return fes_all.mean(axis=0), x, y


# ---- compute all four combinations ----
N = 500001

pairs = [
    (1, 2, r"$\phi_1$ vs $\psi_1$"),
    (3, 4, r"$\phi_2$ vs $\psi_2$"),
    (1, 3, r"$\phi_1$ vs $\phi_2$"),
    (2, 4, r"$\psi_1$ vs $\psi_2$")
]

fes_list = []
for i1, i2, _ in pairs:
    fes_mean, x, y = mean_fes_2d_all_runs(
        bias_alpha_40_run,
        i1, i2,
        N,
        bin_edges
    )
    fes_list.append(fes_mean)

# %%
fig, axes = plt.subplots(
    2, 2,
    figsize=(8, 8),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, fes, label in zip(axes.flatten(), fes_list, panel_labels):
    vmin = 0 
    vmax = 60
    cf = ax.contourf(
        x, y, fes,
        levels=np.linspace(vmin, vmax, 60),
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu"
    )

    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold"
    )

    ax.label_outer()
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

# 手动为 colorbar 留空间
fig.subplots_adjust(right=0.88)

cax = fig.add_axes([0.90, 0.15, 0.025, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label("FES (kJ/mol)")
cbar.set_ticks([10, 30, 50])

plt.savefig("ala4_fes.pdf")
plt.savefig("ala4_fes.png", dpi=300)

# %%
unbias_data = [
    np.loadtxt(f"../chi.1/compare/simulation/sim_{i:03d}/COLVAR")
    for i in range(3)
]

# %%
bias_alpha_10_run = [
    np.loadtxt(f"../chi.1/case_alpha_1_tau_0.2_eps_0.05_deg_10/run{i}/COLVAR")
    for i in range(1, 20)
]

bias_alpha_20_run = [
    np.loadtxt(f"../chi.1/case_alpha_2_tau_0.2_eps_0.05_deg_10/run{i}/COLVAR")
    for i in range(1, 20)
]


bias_alpha_40_run = [
    np.loadtxt(f"../chi.1/case_alpha_4_tau_0.2_eps_0.05_deg_10/run{i}/COLVAR")
    for i in range(1, 30)
]

bias_alpha_80_run = [
    np.loadtxt(f"../chi.1/case_alpha_8_tau_0.2_eps_0.05_deg_10/run{i}/COLVAR")
    for i in range(1, 30)
]


# %%
# 拼接 bias
phi_10 = np.concatenate([d[:, 1] for d in bias_alpha_10_run])
phi_20 = np.concatenate([d[:, 1] for d in bias_alpha_20_run])
phi_40 = np.concatenate([d[:, 1] for d in bias_alpha_40_run])
phi_80 = np.concatenate([d[:, 1] for d in bias_alpha_80_run])

# unbiased 只取到和 bias 相同长度
target_len = len(phi_40)
phi_unbiased_all = np.concatenate([d[:, 1] for d in unbias_data[:3]])
phi_unbiased = phi_unbiased_all[:target_len]

# %%
phi_40.shape

# %%

img = Image.open("chi.1_rm.png")

fig = plt.figure(figsize=(16, 8))

# 外层：左列（示意图 + density）vs 右侧 dynamics
outer_gs = GridSpec(1, 2, figure=fig,
                    width_ratios=[1, 2.0],
                    wspace=0.22)

# 左侧再拆成上下两块
left_gs = GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=outer_gs[0, 0],
    height_ratios=[1.0, 1.5],
    hspace=0.18
)

# ---------------- (a) 示意图 ----------------
ax_img = fig.add_subplot(left_gs[0, 0])
ax_img.imshow(img)
ax_img.axis("off")
ax_img.text(0.03, 0.97, "(a)",
            transform=ax_img.transAxes,
            ha="left", va="top", fontsize=24)

# ---------------- (b) Density 图 ----------------
ax_kde = fig.add_subplot(left_gs[1, 0])

datasets = [
    (phi_unbiased, "Unbiased MD"),
    (phi_20, r"$\alpha=2.0$"),
    (phi_40, r"$\alpha=4.0$"),
    (phi_80, r"$\alpha=8.0$"),
]

x_grid = np.linspace(-np.pi, np.pi, 1000)
bw = 0.15  # 可根据平滑程度调整

for data, label in datasets:
    kde = gaussian_kde(data[::100], bw_method=bw)
    y = kde(x_grid)
    ax_kde.plot(x_grid, y, linewidth=2, label=label)
    ax_kde.fill_between(x_grid, y, alpha=0.25)

ax_kde.set_xlim(-np.pi, np.pi)
ax_kde.set_xlabel(r"$\omega$")
ax_kde.set_ylabel("Density")
ax_kde.legend(frameon=False, fontsize=10)
ax_kde.text(0.1, 0.95, "(b)",
            transform=ax_kde.transAxes,
            ha="left", va="top", fontsize=24)

# ---------------- 右侧 2x2 dynamics ----------------
inner_gs = GridSpecFromSubplotSpec(
    2, 2,
    subplot_spec=outer_gs[0, 1],
    wspace=0,
    hspace=0
)

ax1 = fig.add_subplot(inner_gs[0, 0])
ax2 = fig.add_subplot(inner_gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(inner_gs[1, 0], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(inner_gs[1, 1], sharex=ax1, sharey=ax1)
axes = [ax1, ax2, ax3, ax4]

dt = 0.001 * 50

# y 轴统一
ymin, ymax = -np.pi,np.pi
ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0

panel_labels = ["(c)", "(d)", "(e)", "(f)"]

for ax, (phi, title), label in zip(axes, datasets, panel_labels):
    time = dt * np.arange(phi.shape[0])
    ax.plot(time[::10], phi[::10], ".", ms=0.2)

    ax.text(0.03, 0.85, label,
            transform=ax.transAxes,
            ha="left", va="top")
    ax.text(0.5, 0.85, title,
            transform=ax.transAxes,
            ha="center", va="top")

    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.label_outer()

ax3.set_xlabel("Time [ps]")
ax4.set_xlabel("Time [ps]")
ax1.set_ylabel(r"$\omega$")
ax3.set_ylabel(r"$\omega$")
ax3.set_xticks([0, 20000, 40000])
ax2.set_xlim([0,50000])
plt.savefig("chi1_exploration.png", bbox_inches="tight", dpi=300)
plt.savefig("chi1_exploration.pdf", bbox_inches="tight")
plt.show()

# %%


# %%
import numpy as np

bias_alpha_40 = [
    np.loadtxt(
        f"../chi.1/case_alpha_4_tau_0.2_eps_0.05_deg_10/finial_all/w{i+1:02d}/COLVAR"
    )
    for i in range(32)
]

# %%
# for b in bias_alpha_40:
#     print(b.shape)
bias_alpha_40[0].shape[0]*0.0005*200

# %%
fig, axes = plt.subplots(
    1, 3, figsize=(9, 3), sharex=True,sharey=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)
fes_all = np.array([compute_fes(d[:, 1], 500001, bin_edges, d[:, -1]) for d in bias_alpha_40])
fes_mean = fes_all.mean(axis=0)+10
fes_std = fes_all.std(axis=0)
axes[0].plot(centers, fes_mean, label=r"$\alpha=4.0$", lw=2)
axes[0].fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
fes_all = np.array([compute_fes(d[:, 2], 500001, bin_edges, d[:, -1]) for d in bias_alpha_40])
fes_mean = fes_all.mean(axis=0)+10
fes_std = fes_all.std(axis=0)
axes[1].plot(centers, fes_mean, label=r"$\alpha=4.0$", lw=2)
axes[1].fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
fes_all = np.array([compute_fes(d[:, 3], 500001, bin_edges, d[:, -1]) for d in bias_alpha_40])
fes_mean = fes_all.mean(axis=0)+10
fes_std = fes_all.std(axis=0)
axes[2].plot(centers, fes_mean, label=r"$\alpha=4.0$", lw=2)
axes[2].fill_between(centers, fes_mean - fes_std, fes_mean + fes_std, alpha=0.35)
axes[0].set_ylabel(r"$F(\cdot)[\mathrm{kJ/mol}]$", fontsize=24)
axes[1].set_xlabel(
    r"Dihedral angle",
    fontsize=24
)
axes[0].text(
            0.03, 0.95, r"(a) $\omega$",
            transform=axes[0].transAxes,
            ha="left", va="top",
            fontsize=24
        )
axes[1].text(
            0.03, 0.95, r"(b) $\phi$",
            transform=axes[1].transAxes,
            ha="left", va="top",
            fontsize=24
        )
axes[2].text(
            0.03, 0.95, r"(c) $\psi$",
            transform=axes[2].transAxes,
            ha="left", va="top",
            fontsize=24
        )

axes[0].label_outer()
axes[1].label_outer()
axes[2].label_outer()
axes[0].tick_params(axis='both', labelsize=24)
axes[1].tick_params(axis='both', labelsize=24)
axes[2].tick_params(axis='both', labelsize=24)
axes[0].set_xlim(-np.pi, np.pi)
plt.savefig("chi1_fes.png", bbox_inches="tight", dpi=300)
plt.savefig("chi1_fes.pdf", bbox_inches="tight")
plt.show()

# %%
img = Image.open("chi.3_rm.png")

fig = plt.figure(figsize=(16,8))

outer_gs = GridSpec(
    1,2,
    figure=fig,
    width_ratios=[1.2,2.2],
    wspace=0.18
)

# ---------- (a) image ----------
ax_img = fig.add_subplot(outer_gs[0,0])
ax_img.imshow(img)
ax_img.axis("off")

ax_img.text(
    0.03,0.97,
    "(a)",
    transform=ax_img.transAxes,
    ha="left",va="top",
    fontsize=24
)

# ---------- 3x3 FES ----------
inner_gs = GridSpecFromSubplotSpec(
    3,3,
    subplot_spec=outer_gs[0,1],
    wspace=0,
    hspace=0
)

axes = np.array([
    [fig.add_subplot(inner_gs[i,j]) for j in range(3)]
    for i in range(3)
])

labels = [
    r"(b) $\omega_1$", r"(c) $\phi_1$", r"(d) $\psi_1$",
    r"(e) $\omega_2$", r"(f) $\phi_2$", r"(g) $\psi_2$",
    r"(h) $\omega_3$", r"(i) $\phi_3$", r"(j) $\psi_3$",
]
bias_alpha_4_run = []
for i in range(1, 32):
    try:
        data =  np.loadtxt(f"../chi.3/case_alpha_4_tau_5_eps_1_deg_10/run{i}/COLVAR")
        bias_alpha_4_run.append(data[:,:11])
    except:
        print(f"../chi.3/case_alpha_4_tau_5_eps_1_deg_10/run{i}/COLVAR")
        print(data.shape)

data = np.concatenate(bias_alpha_4_run,axis=0)
print(data.shape)
k = 0
for i in range(3):
    for j in range(3):

        ax = axes[i,j]
        time = np.arange(data.shape[0])*0.0005*200
        ax.plot(time[::100],data[::100,1+i*3+j], ".", ms=0.2)


        # panel label inside
        ax.text(
            0.03,0.92,
            labels[k],
            transform=ax.transAxes,
            fontsize=16,
            ha="left",
            va="top"
        )

        k += 1

# only outer ticks
for i in range(3):
    for j in range(3):

        if j != 0:
            axes[i,j].set_yticklabels([])

        if i != 2:
            axes[i,j].set_xticklabels([])

# shared axis labels
axes[1,0].set_ylabel(
    r"$F(\cdot)\,[\mathrm{kJ/mol}]$",
    fontsize=24
)

axes[2,1].set_xlabel(
    r"Dihedral angle",
    fontsize=22
)

plt.savefig("chi3_explore.png",dpi=300,bbox_inches="tight")
plt.savefig("chi3_explore.pdf",bbox_inches="tight")

plt.show()

# %%
fig,axes = plt.subplots(3,3,figsize=(8,8), sharex=True,sharey=True,
    gridspec_kw={"wspace": 0, "hspace": 0}
)


labels = [
    r"(b) $\omega_1$", r"(c) $\phi_1$", r"(d) $\psi_1$",
    r"(e) $\omega_2$", r"(f) $\phi_2$", r"(g) $\psi_2$",
    r"(h) $\omega_3$", r"(i) $\phi_3$", r"(j) $\psi_3$",
]

data = np.concatenate(bias_alpha_4_run,axis=0)
k = 0
for i in range(3):
    for j in range(3):

        file = f"../chi.3/case_alpha_4_tau_5_eps_1_deg_10/mbar_1d_marginal_T300K/dih-{i+1:03d}-{j:02d}_1d_fes.csv"

        data = np.loadtxt(file,delimiter=",",skiprows=1)

        ax = axes[i,j]
        ax.plot(data[:,0],data[:,2],lw=2)


        # panel label inside
        ax.text(
            0.03,0.92,
            labels[k],
            transform=ax.transAxes,
            fontsize=16,
            ha="left",
            va="top"
        )

        k += 1

# # only outer ticks
# for i in range(3):
#     for j in range(3):

#         if j != 0:
#             axes[i,j].set_yticklabels([])

#         if i != 2:
#             axes[i,j].set_xticklabels([])

# shared axis labels
axes[1,0].set_ylabel(
    r"$F(\cdot)\,[\mathrm{kJ/mol}]$",
    fontsize=24
)

axes[2,1].set_xlabel(
    r"Dihedral angle",
    fontsize=22
)

plt.savefig("chi3_fes.png",dpi=300,bbox_inches="tight")
plt.savefig("chi3_fes.pdf",bbox_inches="tight")


# %%


# %%
bias_alpha_4_run[1].shape

# %%
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde

root = Path("..")
kBT = 2.5
dt = 0.0005

MULLER_A = np.array([-200, -100, -170, 15])
MULLER_a = np.array([-1, -1, -6.5, 0.7])
MULLER_b = np.array([0, 0, 11, 0.6])
MULLER_c = np.array([-10, -10, -6.5, 0.7])
MULLER_x0 = np.array([1, 0, -0.5, -1])
MULLER_y0 = np.array([0, 0.5, 1.5, 1])

def muller_potential(x, y):
    V = 0.0
    for i in range(4):
        dx = x - MULLER_x0[i]
        dy = y - MULLER_y0[i]
        V += MULLER_A[i] * np.exp(MULLER_a[i] * dx**2 + MULLER_b[i] * dx * dy + MULLER_c[i] * dy**2)
    return V

def count_transitions(trajectory):
    n_A_to_B = 0
    n_B_to_A = 0
    last_state = None
    for x, y in trajectory:
        if x < -0.3 and y > 1.0:
            current = "A"
        elif x > 0.4 and y < 0.3:
            current = "B"
        else:
            continue
        if last_state is not None and current != last_state:
            if last_state == "A":
                n_A_to_B += 1
            else:
                n_B_to_A += 1
        last_state = current
    return n_A_to_B, n_B_to_A

def compute_2d_kde_fes(
    trajectory,
    weights,
    x_range=(-1.5, 1.2),
    y_range=(-0.5, 2.0),
    n_grid=180,
    bw_method=0.12,
    max_kde_points=400000,
):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n_points = len(trajectory)
    if n_points > max_kde_points:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_points, size=max_kde_points, replace=False))
        traj_use = trajectory[idx]
        weights_use = weights[idx]
        weights_use = weights_use / weights_use.sum()
    else:
        traj_use = trajectory
        weights_use = weights

    kde = gaussian_kde(traj_use.T, weights=weights_use, bw_method=bw_method)
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)
    y_grid = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    density = np.maximum(density, 1e-300)
    fes = -kBT * np.log(density)
    fes -= np.nanmin(fes)
    mask = np.isfinite(fes)
    return X, Y, fes, mask

unbiased_traj = np.load(root / "muller" / "Data" / "paper_figures_4885160" / "unbiased_traj.npz")["trajectory"]

biased_info = [
    (
        r"$\alpha=13$",
        root / "muller" / "Data" / "softplus_deg15_rank15_alpha13_eps0.05_tau0.1_traj113" / "results.npz",
    ),
    (
        r"$\alpha=16$",
        root / "muller" / "Data" / "softplus_deg15_rank15_alpha16_eps0.1_tau0.1_traj64" / "results.npz",
    ),
    (
        r"$\alpha=20$",
        root / "muller" / "Data" / "softplus_deg15_rank15_alpha20_eps0.05_tau0.05_traj198" / "results.npz",
    ),
]

biased_trajs = []
biased_weights = []
biased_labels = []
for label, npz_path in biased_info:
    data = np.load(npz_path, allow_pickle=True)
    biased_trajs.append(data["trajectory"])
    biased_weights.append(data["weights"])
    biased_labels.append(label)

fig = plt.figure(figsize=(17.2, 9.4))
outer_gs = GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[1.02, 2.28],
    height_ratios=[1.0, 1.08],
    wspace=0.18,
    hspace=0.30,
)

energy_levels_filled = np.linspace(0, 120, 241)
energy_levels_lines = np.linspace(0, 120, 41)
energy_ticks = np.arange(0, 121, 30)
energy_cmap = plt.get_cmap("turbo").copy()
energy_cmap.set_over("#6b0000")
energy_norm = mcolors.Normalize(vmin=0, vmax=120)

# (a) Muller potential
ax_potential = fig.add_subplot(outer_gs[:, 0])
x_grid = np.linspace(-1.42, 1.10, 500)
y_grid = np.linspace(-0.25, 1.99, 500)
X0, Y0 = np.meshgrid(x_grid, y_grid)
V0 = muller_potential(X0, Y0)
V0 = V0 - V0.min()
cf0 = ax_potential.contourf(
    X0, Y0, V0,
    levels=energy_levels_filled,
    cmap=energy_cmap, norm=energy_norm, extend="max",
)
ax_potential.contour(
    X0, Y0, V0,
    levels=energy_levels_lines,
    colors="white", linewidths=0.7, alpha=0.55,
)
ax_potential.set_xlabel(r"$x$")
ax_potential.set_ylabel(r"$y$")
ax_potential.set_aspect("equal")
ax_potential.set_title("(a) Muller potential", loc="left", pad=8)
ax_potential.tick_params(direction="in", top=True, right=True)
cbar0 = fig.colorbar(cf0, ax=ax_potential, fraction=0.05, pad=0.03, ticks=energy_ticks)
cbar0.set_label(r"$F(x, y)$")

# (b-e) time series
ts_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0, 1], wspace=0.0, hspace=0.0)
ax_ts_ref = fig.add_subplot(ts_gs[0, 0])
ts_axes = [
    ax_ts_ref,
    fig.add_subplot(ts_gs[0, 1], sharex=ax_ts_ref, sharey=ax_ts_ref),
    fig.add_subplot(ts_gs[1, 0], sharex=ax_ts_ref, sharey=ax_ts_ref),
    fig.add_subplot(ts_gs[1, 1], sharex=ax_ts_ref, sharey=ax_ts_ref),
]
time_series = [(unbiased_traj, "Unbiased")] + list(zip(biased_trajs, biased_labels))
panel_labels = ["(b)", "(c)", "(d)", "(e)"]
thin = 100
time_scale = 1e3
for ax, (traj, label), plabel in zip(ts_axes, time_series, panel_labels):
    idx = np.arange(0, len(traj), thin)
    t = idx * dt / time_scale
    x = traj[idx, 0]
    ax.scatter(t, x, s=0.1, alpha=0.35, color="#1f77b4", rasterized=True)
    ax.axhline(-0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0.6, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylim(-1.8, 1.8)
    n_ab, n_ba = count_transitions(traj)
    ax.text(
        0.02, 0.98, f"{plabel} {label}",
        transform=ax.transAxes, ha="left", va="top", fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        zorder=5,
    )
    ax.text(
        0.98, 0.98, f"transitions: {n_ab + n_ba}",
        transform=ax.transAxes, ha="right", va="top", fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        zorder=5,
    )
    ax.tick_params(direction="in", top=True, right=True)
    ax.label_outer()

ts_axes[0].set_ylabel(r"$x$")
ts_axes[2].set_ylabel(r"$x$")
ts_axes[2].set_xlabel(r"$t$ [$10^3$ a.u.]", labelpad=2)
ts_axes[3].set_xlabel(r"$t$ [$10^3$ a.u.]", labelpad=2)

# (f-g) 2D FES from KDE
best_traj = biased_trajs[-1]
best_weights = biased_weights[-1]
X, Y, fes_sim, _ = compute_2d_kde_fes(best_traj, best_weights)
fes_theory = muller_potential(X, Y)
fes_theory_aligned = fes_theory - np.min(fes_theory)
plot_mask = fes_theory_aligned <= 120
fes_plot = np.where(plot_mask, fes_sim, np.nan)
fes_diff = np.full_like(fes_sim, np.nan)
fes_diff[plot_mask] = fes_sim[plot_mask] - fes_theory_aligned[plot_mask]
rmse = np.sqrt(np.nanmean(fes_diff[plot_mask]**2))

fes_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1, 1], wspace=0.28)
ax_f_rew = fig.add_subplot(fes_gs[0, 0])
ax_f_diff = fig.add_subplot(fes_gs[0, 1], sharex=ax_f_rew, sharey=ax_f_rew)

cf1 = ax_f_rew.contourf(X, Y, fes_plot, levels=energy_levels_filled, cmap=energy_cmap, norm=energy_norm, extend="max")
ax_f_rew.contour(X, Y, fes_plot, levels=energy_levels_lines, colors="white", linewidths=0.55, alpha=0.40)
diff_limit = min(np.nanmax(np.abs(fes_diff)), 10.0)
levels_diff = np.linspace(-diff_limit, diff_limit, 41)
cf2 = ax_f_diff.contourf(X, Y, fes_diff, levels=levels_diff, cmap="RdBu_r", extend="both")

ax_f_rew.text(
    0.02, 0.98, "(f) KDE reweighted FES",
    transform=ax_f_rew.transAxes, ha="left", va="top", fontsize=18,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
    zorder=5,
)
ax_f_diff.text(
    0.02, 0.98, "(g) Difference",
    transform=ax_f_diff.transAxes, ha="left", va="top", fontsize=18,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
    zorder=5,
)
ax_f_diff.text(
    0.98, 0.98, f"RMSE = {rmse:.3f}",
    transform=ax_f_diff.transAxes, ha="right", va="top", fontsize=12,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
    zorder=5,
)

for ax in [ax_f_rew, ax_f_diff]:
    ax.set_xlabel(r"$x$")
    ax.set_aspect("equal")
    ax.tick_params(direction="in", top=True, right=True)
ax_f_rew.set_ylabel(r"$y$")
ax_f_diff.tick_params(labelleft=False)

cbar1 = fig.colorbar(cf1, ax=ax_f_rew, fraction=0.05, pad=0.03, ticks=energy_ticks)
cbar1.set_label(r"$F(x, y)$")
cbar2 = fig.colorbar(cf2, ax=ax_f_diff, fraction=0.05, pad=0.03)
cbar2.set_label(r"$\Delta F$")

plt.savefig("muller_result.png", dpi=300, bbox_inches="tight")
plt.savefig("muller_result.pdf", bbox_inches="tight")
plt.show()

# %%
from scipy.ndimage import uniform_filter1d

# ── Physical constants ────────────────────────────────────────────────────────
kB   = 8.31446261815324e-3   # kJ mol⁻¹ K⁻¹
T    = 300.0                  # K
kBT  = kB * T                # kJ mol⁻¹
kCal = 4.184                  # kJ / kcal

CASE_DIR = Path("/Volumes/passport/meta_tensor/1UAO_3/case2/")
OUT_DIR  = CASE_DIR / "mbar_1d_marginal_T300K"
FIG_DIR  = "./"


STRIDE   = 20
N_RUNS   = 65
DT_PS    = 0.1   # ps per COLVAR frame
N_WALKERS = 16
print(f"Output figures: {FIG_DIR}")

# %%
# ── Cell 2: Load data ─────────────────────────────────────────────────────────
# MBAR weights (1 per frame)
csv_data = np.loadtxt(OUT_DIR / "mbar_unbiased_samples_with_weights.csv",
                      delimiter=",", skiprows=1)
weights  = csv_data[:, -1]          # shape (N,)
phi_psi  = csv_data[:, :-1]         # shape (N, 16)  — phi2..psi9 in rad

# Raw COLVAR data (same ordering: runs 1..65, stride=20)
blocks  = []
run_tag = []          # which run each frame came from
for rid in range(1, N_RUNS + 1):
    d = np.load(CASE_DIR / f"run{rid}/COLVAR.npy")[::STRIDE]
    blocks.append(d)
    run_tag.extend([rid] * len(d))

raw      = np.vstack(blocks)          # (N, 23)
run_tag  = np.array(run_tag)          # (N,)
assert len(raw) == len(weights), f"{len(raw)} vs {len(weights)}"

# ── Extract collective variables ──────────────────────────────────────────────
# COLVAR columns: time | phi2 psi2 ... phi9 psi9 | rg | e2e | rmsd_ca | d1 | d2 | metad.bias
rmsd_ca = raw[:, 19] * 10    # Å
rg      = raw[:, 17] * 10    # Å
e2e     = raw[:, 18] * 10    # Å
d1      = raw[:, 20] * 10    # Å
d2      = raw[:, 21] * 10    # Å
bias    = raw[:, 22]          # kJ mol⁻¹
phi2    = raw[:, 1]           # rad  (the MetaD CV)

w    = weights / weights.sum()
ESS  = 1.0 / np.sum(w**2)
print(f"Total frames N = {len(w):,}")
print(f"ESS = {ESS:,.0f}  ({ESS/len(w)*100:.1f}% of N)")
print(f"Max bias deposited: {bias.max():.1f} kJ/mol")
print(f"RMSD_CA range: {rmsd_ca.min():.2f} – {rmsd_ca.max():.2f} Å")
print(f"Rg range:      {rg.min():.2f} – {rg.max():.2f} Å")
print(f"d1 range:      {d1.min():.2f} – {d1.max():.2f} Å")
print(f"d2 range:      {d2.min():.2f} – {d2.max():.2f} Å")

# %%
def load_window_continuous(window):
    chunks = []
    t_offset = 0.0
    for rid in range(1, N_RUNS + 1):
        path = CASE_DIR / f'run{rid}/{window}/COLVAR'
        if not path.exists(): continue
        d = np.loadtxt(path, comments='#')
        if d.ndim == 1: d = d[None, :]
        t_abs = t_offset + np.arange(len(d)) * DT_PS
        t_offset += len(d) * DT_PS
        chunks.append((rid, t_abs, d))
    return chunks

# %%

def count_real_transitions(rmsd, folded_thresh=2.0, unfolded_thresh=3.5):
    """Two-threshold (deadband) transition counting."""
    state = 0  # 0=unknown, 1=folded, 2=unfolded
    if rmsd[0] < folded_thresh: state = 1
    elif rmsd[0] > unfolded_thresh: state = 2
    n_FU = 0; n_UF = 0
    for v in rmsd:
        if state == 1 and v > unfolded_thresh:
            state = 2; n_FU += 1
        elif state == 2 and v < folded_thresh:
            state = 1; n_UF += 1
    return n_FU, n_UF

# %%


# ── Precompute: RMSD per walker, all iterations ──────────────────────────────
print("Loading all walkers...")
# For each walker, get full continuous RMSD
walker_rmsd_all = {}
for wi in range(1, N_WALKERS + 1):
    w = f'w{wi:02d}'
    chunks = load_window_continuous(w)
    rmsd = np.concatenate([d[:, 19]*10 for _, _, d in chunks])
    rid  = np.concatenate([np.full(len(t_), r) for r, t_, _ in chunks])
    walker_rmsd_all[w] = (rmsd, rid)

# Per-iteration: RMSD distribution across all walkers
rmsd_by_iter = {}
for it in range(1, N_RUNS + 1):
    all_r = []
    for wi in range(1, N_WALKERS + 1):
        rmsd, rid = walker_rmsd_all[f'w{wi:02d}']
        all_r.append(rmsd[rid == it])
    rmsd_by_iter[it] = np.concatenate(all_r)

print("Done loading.")

fig = plt.figure(figsize=(8.2, 7.6))
gs = GridSpec(
    3, 2, figure=fig,
    hspace=0.40, wspace=0.18,
    height_ratios=[1.45, 1.05, 1.18],
)
fig.subplots_adjust(left=0.10, right=0.87, top=0.95, bottom=0.09)

cmap = plt.cm.viridis
iter_norm = plt.Normalize(1, N_RUNS)
folded_thr = 2.0
unfolded_thr = 3.5
folded_color = '#2b6cb0'
unfolded_color = '#d94841'
trace_color = '#1f77b4'
median_color = '#0f4c5c'
early_color = '#2c7a7b'
late_color = '#b85c38'
label_fs = 14
tick_fs = 11
header_fs = 9.6
subheader_fs = 8.0
legend_fs = 7.2

all_rmsd = np.concatenate([vals[0] for vals in walker_rmsd_all.values()])
y_max = max(unfolded_thr + 0.8, np.percentile(all_rmsd, 99.7) + 0.2)
iter_early = min(5, N_RUNS)
iter_late = min(50, N_RUNS)
iter_ticks = np.unique(np.round(np.linspace(1, N_RUNS, 4)).astype(int))

def style_rmsd_axis(ax, xlabel=None, ylabel=True):
    ax.set_ylim(0.0, y_max)
    ax.grid(axis='y', color='0.88', linewidth=0.8)
    ax.axhspan(0.0, folded_thr, color=folded_color, alpha=0.05, zorder=0)
    ax.axhspan(unfolded_thr, y_max, color=unfolded_color, alpha=0.05, zorder=0)
    ax.axhline(folded_thr, color=folded_color, lw=1.0, ls='--', alpha=0.9)
    ax.axhline(unfolded_thr, color=unfolded_color, lw=1.0, ls='--', alpha=0.9)
    ax.tick_params(direction='in', top=True, right=True, labelsize=tick_fs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=4)
    if ylabel:
        ax.set_ylabel(r'$C_\alpha$ RMSD ($\AA$)', fontsize=label_fs, labelpad=6)

def panel_header(ax, label, title, subtitle=None):
    ax.text(
        0.01, 0.98, f'{label} {title}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=header_fs, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.90, pad=1.1),
        zorder=6,
    )
    if subtitle is not None:
        ax.text(
            0.99, 0.98, subtitle,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=subheader_fs, color='0.35',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.9),
            zorder=6,
        )

def get_iteration_stack(iter_id):
    segments = []
    min_len = None
    for wi in range(1, N_WALKERS + 1):
        rmsd, rid = walker_rmsd_all[f'w{wi:02d}']
        seg = rmsd[rid == iter_id]
        if len(seg) == 0:
            continue
        segments.append(seg)
        min_len = len(seg) if min_len is None else min(min_len, len(seg))
    segments = [seg[:min_len] for seg in segments]
    return np.vstack(segments), np.arange(min_len) * DT_PS / 1000.0

# (a) Continuous RMSD trace for w01 across all iterations
ax_top = fig.add_subplot(gs[0, :])
chunks_w01 = load_window_continuous('w01')
for rid, t_abs, d in chunks_w01:
    rmsd_seg = d[:, 19] * 10
    col = cmap((rid - 1) / max(N_RUNS - 1, 1))
    ax_top.plot(t_abs / 1000.0, rmsd_seg, lw=0.55, color=col, alpha=0.75, rasterized=True)
rmsd_w01 = np.concatenate([d[:, 19] * 10 for _, _, d in chunks_w01])
t_w01 = np.concatenate([t for _, t, _ in chunks_w01]) / 1000.0
rmsd_smooth = uniform_filter1d(rmsd_w01, size=500)
ax_top.plot(t_w01, rmsd_smooth, color='k', lw=1.6, alpha=0.82, label='50-ps running avg')
style_rmsd_axis(ax_top, xlabel='Time (ns)')
ax_top.set_xlim(0.0, t_w01[-1])
panel_header(ax_top, '(a)', 'Single-walker continuous trajectory', 'w01, colored by iteration')
ax_top.text(
    0.995, folded_thr + 0.08, 'native-like',
    transform=ax_top.get_yaxis_transform(), ha='right', va='bottom',
    fontsize=8.0, color=folded_color,
)
ax_top.text(
    0.995, unfolded_thr + 0.08, 'high-deviation',
    transform=ax_top.get_yaxis_transform(), ha='right', va='bottom',
    fontsize=8.0, color=unfolded_color,
)
ax_top.legend(
    loc='upper left', bbox_to_anchor=(0.01, 0.84),
    framealpha=0.92, fontsize=legend_fs,
    handlelength=1.5, borderpad=0.25, labelspacing=0.25, handletextpad=0.45,
)

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=iter_norm)
sm.set_array([])
cax = fig.add_axes([0.89, 0.695, 0.018, 0.255])
cb = plt.colorbar(sm, cax=cax)
cb.set_label('Iteration', fontsize=legend_fs + 1)
cb.set_ticks(iter_ticks)
cb.ax.tick_params(labelsize=tick_fs - 1)

# (b) RMSD distribution across all walkers per iteration
ax_mid = fig.add_subplot(gs[1, :])
iters = np.arange(1, N_RUNS + 1)
pct10 = [np.percentile(rmsd_by_iter[it], 10) for it in iters]
pct25 = [np.percentile(rmsd_by_iter[it], 25) for it in iters]
pct50 = [np.percentile(rmsd_by_iter[it], 50) for it in iters]
pct75 = [np.percentile(rmsd_by_iter[it], 75) for it in iters]
pct90 = [np.percentile(rmsd_by_iter[it], 90) for it in iters]
rmsd_min = [rmsd_by_iter[it].min() for it in iters]
rmsd_max = [rmsd_by_iter[it].max() for it in iters]

ax_mid.fill_between(iters, rmsd_min, rmsd_max, color='0.45', alpha=0.06, linewidth=0)
ax_mid.fill_between(iters, pct10, pct90, alpha=0.16, color=trace_color, label='10-90%')
ax_mid.fill_between(iters, pct25, pct75, alpha=0.28, color=trace_color, label='25-75%')
ax_mid.plot(iters, pct50, color=median_color, lw=2.0, label='Median')
ax_mid.plot(iters, rmsd_min, color='0.55', lw=0.8, alpha=0.85)
ax_mid.plot(iters, rmsd_max, color='0.55', lw=0.8, alpha=0.85)
ax_mid.axvline(iter_early, color=early_color, lw=1.0, ls=':', alpha=0.9)
ax_mid.axvline(iter_late, color=late_color, lw=1.0, ls=':', alpha=0.9)
ax_mid.scatter([iter_early, iter_late], [pct50[iter_early - 1], pct50[iter_late - 1]],
               color=[early_color, late_color], s=28, zorder=5, edgecolor='white', linewidth=0.6)
style_rmsd_axis(ax_mid, xlabel='Iteration')
ax_mid.set_xlim(1, N_RUNS)
ax_mid.set_xticks(iter_ticks)
panel_header(ax_mid, '(b)', 'Distribution across all walkers')
ax_mid.legend(
    loc='upper right', bbox_to_anchor=(0.995, 0.98),
    framealpha=0.92, fontsize=legend_fs, ncol=3,
    handlelength=1.4, borderpad=0.25, labelspacing=0.25, handletextpad=0.45,
)

# (c-d) Early and late iterations across all walkers
ax_early = fig.add_subplot(gs[2, 0])
ax_late = fig.add_subplot(gs[2, 1], sharex=ax_early, sharey=ax_early)

stack_early, t_early = get_iteration_stack(iter_early)
stack_late, t_late = get_iteration_stack(iter_late)
x_max = max(t_early[-1], t_late[-1])

for seg in stack_early:
    ax_early.plot(t_early, seg, color=early_color, lw=0.7, alpha=0.18, rasterized=True)
ax_early.fill_between(t_early,
                      np.percentile(stack_early, 25, axis=0),
                      np.percentile(stack_early, 75, axis=0),
                      color=early_color, alpha=0.18, linewidth=0)
ax_early.plot(t_early, np.median(stack_early, axis=0), color=early_color, lw=2.2)
style_rmsd_axis(ax_early, xlabel='Time within iteration (ns)')
ax_early.set_xlim(0.0, x_max)
panel_header(ax_early, '(c)', f'Early exploration (iter {iter_early})')
ax_early.legend(handles=[
    matplotlib.lines.Line2D([0], [0], color=early_color, lw=0.8, alpha=0.25, label='individual walkers'),
    matplotlib.patches.Patch(facecolor=early_color, alpha=0.18, label='25-75% range'),
    matplotlib.lines.Line2D([0], [0], color=early_color, lw=2.2, label='walker median'),
], loc='lower left', bbox_to_anchor=(0.01, 0.02), framealpha=0.92,
   fontsize=6.6, handlelength=1.2, borderpad=0.20, labelspacing=0.20, handletextpad=0.35)

for seg in stack_late:
    ax_late.plot(t_late, seg, color=late_color, lw=0.7, alpha=0.18, rasterized=True)
ax_late.fill_between(t_late,
                     np.percentile(stack_late, 25, axis=0),
                     np.percentile(stack_late, 75, axis=0),
                     color=late_color, alpha=0.18, linewidth=0)
ax_late.plot(t_late, np.median(stack_late, axis=0), color=late_color, lw=2.2)
style_rmsd_axis(ax_late, xlabel='Time within iteration (ns)', ylabel=False)
ax_late.set_xlim(0.0, x_max)
ax_late.tick_params(labelleft=False)
panel_header(ax_late, '(d)', f'Bias-strengthened exploration (iter {iter_late})')

fig.savefig('fig1_phase_space_exploration.png', dpi=300, bbox_inches='tight')
fig.savefig('fig1_phase_space_exploration.pdf', bbox_inches='tight')

# %%
# -- R_g version: compaction-oriented view ------------------------------------
print("Loading all walkers for R_g figure...")
walker_rg_all = {}
for wi in range(1, N_WALKERS + 1):
    w = f'w{wi:02d}'
    chunks = load_window_continuous(w)
    rg = np.concatenate([d[:, 17] * 10 for _, _, d in chunks])
    rid = np.concatenate([np.full(len(t_), r) for r, t_, _ in chunks])
    walker_rg_all[w] = (rg, rid)

rg_by_iter = {}
for it in range(1, N_RUNS + 1):
    all_rg = []
    for wi in range(1, N_WALKERS + 1):
        rg, rid = walker_rg_all[f'w{wi:02d}']
        all_rg.append(rg[rid == it])
    rg_by_iter[it] = np.concatenate(all_rg)

print("Done loading R_g.")

fig = plt.figure(figsize=(8.2, 7.6))
gs = GridSpec(
    3, 2, figure=fig,
    hspace=0.40, wspace=0.18,
    height_ratios=[1.45, 1.05, 1.18],
)
fig.subplots_adjust(left=0.10, right=0.87, top=0.95, bottom=0.12)

cmap = plt.cm.viridis
iter_norm = plt.Normalize(1, N_RUNS)
trace_color = '#3a86b8'
median_color = '#0f4c5c'
early_color = '#2c7a7b'
late_color = '#b85c38'
label_fs = 14
tick_fs = 11
header_fs = 9.6
subheader_fs = 8.0
legend_fs = 7.2

all_rg = np.concatenate([vals[0] for vals in walker_rg_all.values()])
y_min = max(0.0, np.percentile(all_rg, 0.3) - 0.25)
y_max = np.percentile(all_rg, 99.7) + 0.25
iter_early = min(5, N_RUNS)
iter_late = min(50, N_RUNS)
iter_ticks = np.unique(np.round(np.linspace(1, N_RUNS, 4)).astype(int))

def style_rg_axis(ax, xlabel=None, ylabel=True):
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', color='0.88', linewidth=0.8)
    ax.tick_params(direction='in', top=True, right=True, labelsize=tick_fs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=4)
    if ylabel:
        ax.set_ylabel(r'$R_g$ ($\AA$)', fontsize=label_fs, labelpad=6)

def panel_header(ax, label, title, subtitle=None):
    ax.text(
        0.01, 0.98, f'{label} {title}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=header_fs, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.90, pad=1.1),
        zorder=6,
    )
    if subtitle is not None:
        ax.text(
            0.99, 0.98, subtitle,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=subheader_fs, color='0.35',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, pad=0.9),
            zorder=6,
        )

def get_iteration_stack(metric_dict, iter_id):
    segments = []
    min_len = None
    for wi in range(1, N_WALKERS + 1):
        metric, rid = metric_dict[f'w{wi:02d}']
        seg = metric[rid == iter_id]
        if len(seg) == 0:
            continue
        segments.append(seg)
        min_len = len(seg) if min_len is None else min(min_len, len(seg))
    segments = [seg[:min_len] for seg in segments]
    return np.vstack(segments), np.arange(min_len) * DT_PS / 1000.0

# (a) Continuous R_g trace for w01 across all iterations
ax_top = fig.add_subplot(gs[0, :])
chunks_w01 = load_window_continuous('w01')
for rid, t_abs, d in chunks_w01:
    rg_seg = d[:, 17] * 10
    col = cmap((rid - 1) / max(N_RUNS - 1, 1))
    ax_top.plot(t_abs / 1000.0, rg_seg, lw=0.55, color=col, alpha=0.75, rasterized=True)
rg_w01 = np.concatenate([d[:, 17] * 10 for _, _, d in chunks_w01])
t_w01 = np.concatenate([t for _, t, _ in chunks_w01]) / 1000.0
rg_smooth = uniform_filter1d(rg_w01, size=500)
ax_top.plot(t_w01, rg_smooth, color='k', lw=1.6, alpha=0.82, label='50-ps running avg')
style_rg_axis(ax_top, xlabel='Time (ns)')
ax_top.set_xlim(0.0, t_w01[-1])
panel_header(ax_top, '(a)', 'Single-walker compaction trace', 'w01, colored by iteration')
ax_top.text(
    0.995, 0.03, r'smaller $R_g$ = more compact',
    transform=ax_top.transAxes, ha='right', va='bottom',
    fontsize=7.8, color='0.35',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.8),
)
ax_top.legend(
    loc='upper left', bbox_to_anchor=(0.01, 0.84),
    framealpha=0.92, fontsize=legend_fs,
    handlelength=1.5, borderpad=0.25, labelspacing=0.25, handletextpad=0.45,
)

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=iter_norm)
sm.set_array([])
cax = fig.add_axes([0.89, 0.695, 0.018, 0.255])
cb = plt.colorbar(sm, cax=cax)
cb.set_label('Iteration', fontsize=legend_fs + 1)
cb.set_ticks(iter_ticks)
cb.ax.tick_params(labelsize=tick_fs - 1)

# (b) R_g distribution across all walkers per iteration
ax_mid = fig.add_subplot(gs[1, :])
iters = np.arange(1, N_RUNS + 1)
pct10 = [np.percentile(rg_by_iter[it], 10) for it in iters]
pct25 = [np.percentile(rg_by_iter[it], 25) for it in iters]
pct50 = [np.percentile(rg_by_iter[it], 50) for it in iters]
pct75 = [np.percentile(rg_by_iter[it], 75) for it in iters]
pct90 = [np.percentile(rg_by_iter[it], 90) for it in iters]
rg_min = [rg_by_iter[it].min() for it in iters]
rg_max = [rg_by_iter[it].max() for it in iters]

ax_mid.fill_between(iters, rg_min, rg_max, color='0.45', alpha=0.06, linewidth=0)
ax_mid.fill_between(iters, pct10, pct90, alpha=0.16, color=trace_color, label='10-90%')
ax_mid.fill_between(iters, pct25, pct75, alpha=0.28, color=trace_color, label='25-75%')
ax_mid.plot(iters, pct50, color=median_color, lw=2.0, label='Median')
ax_mid.plot(iters, rg_min, color='0.55', lw=0.8, alpha=0.85)
ax_mid.plot(iters, rg_max, color='0.55', lw=0.8, alpha=0.85)
ax_mid.axvline(iter_early, color=early_color, lw=1.0, ls=':', alpha=0.9)
ax_mid.axvline(iter_late, color=late_color, lw=1.0, ls=':', alpha=0.9)
ax_mid.scatter([iter_early, iter_late], [pct50[iter_early - 1], pct50[iter_late - 1]],
               color=[early_color, late_color], s=28, zorder=5, edgecolor='white', linewidth=0.6)
style_rg_axis(ax_mid, xlabel='Iteration')
ax_mid.set_xlim(1, N_RUNS)
ax_mid.set_xticks(iter_ticks)
panel_header(ax_mid, '(b)', 'Distribution across all walkers')
ax_mid.legend(
    loc='upper right', bbox_to_anchor=(0.995, 0.98),
    framealpha=0.92, fontsize=legend_fs, ncol=3,
    handlelength=1.4, borderpad=0.25, labelspacing=0.25, handletextpad=0.45,
)

# (c-d) Early and late iterations across all walkers
ax_early = fig.add_subplot(gs[2, 0])
ax_late = fig.add_subplot(gs[2, 1], sharex=ax_early, sharey=ax_early)

stack_early, t_early = get_iteration_stack(walker_rg_all, iter_early)
stack_late, t_late = get_iteration_stack(walker_rg_all, iter_late)
x_max = max(t_early[-1], t_late[-1])

for seg in stack_early:
    ax_early.plot(t_early, seg, color=early_color, lw=0.7, alpha=0.18, rasterized=True)
ax_early.fill_between(t_early,
                      np.percentile(stack_early, 25, axis=0),
                      np.percentile(stack_early, 75, axis=0),
                      color=early_color, alpha=0.18, linewidth=0)
ax_early.plot(t_early, np.median(stack_early, axis=0), color=early_color, lw=2.2)
style_rg_axis(ax_early, xlabel='Time within iteration (ns)')
ax_early.set_xlim(0.0, x_max)
panel_header(ax_early, '(c)', f'Early compaction (iter {iter_early})')
ax_early.legend(handles=[
    matplotlib.lines.Line2D([0], [0], color=early_color, lw=0.8, alpha=0.25, label='individual walkers'),
    matplotlib.patches.Patch(facecolor=early_color, alpha=0.18, label='25-75% range'),
    matplotlib.lines.Line2D([0], [0], color=early_color, lw=2.2, label='walker median'),
], loc='lower left', bbox_to_anchor=(0.01, 0.02), framealpha=0.92,
   fontsize=6.6, handlelength=1.2, borderpad=0.20, labelspacing=0.20, handletextpad=0.35)

for seg in stack_late:
    ax_late.plot(t_late, seg, color=late_color, lw=0.7, alpha=0.18, rasterized=True)
ax_late.fill_between(t_late,
                     np.percentile(stack_late, 25, axis=0),
                     np.percentile(stack_late, 75, axis=0),
                     color=late_color, alpha=0.18, linewidth=0)
ax_late.plot(t_late, np.median(stack_late, axis=0), color=late_color, lw=2.2)
style_rg_axis(ax_late, xlabel='Time within iteration (ns)', ylabel=False)
ax_late.set_xlim(0.0, x_max)
ax_late.tick_params(labelleft=False)
panel_header(ax_late, '(d)', f'Late compaction (iter {iter_late})')

fig.savefig('fig1_phase_space_exploration_rg.png', dpi=300, bbox_inches='tight')
fig.savefig('fig1_phase_space_exploration_rg.pdf', bbox_inches='tight')
print('R_g figure saved')


# %%
# ── Figure 3: 2D FES with annotated metastable states ───────────────────────
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec

def fes_2d(x, y, sample_weights, bin_edges_x, bin_edges_y, sigma_smooth=1.0, max_fes=None):
    sample_weights = np.asarray(sample_weights, dtype=float)
    sample_weights = sample_weights / sample_weights.sum()

    hist, xedges, yedges = np.histogram2d(
        x, y,
        bins=[bin_edges_x, bin_edges_y],
        weights=sample_weights,
    )
    hist = gaussian_filter(hist, sigma=sigma_smooth)

    prob = hist / hist.sum()
    fes = np.full_like(prob, np.nan, dtype=float)
    mask = prob > 0
    fes[mask] = -kBT * np.log(prob[mask])
    fes[mask] -= np.nanmin(fes[mask])

    if max_fes is not None:
        fes[fes > max_fes] = np.nan

    cx = 0.5 * (xedges[:-1] + xedges[1:])
    cy = 0.5 * (yedges[:-1] + yedges[1:])
    return cx, cy, fes

def load_pdb_atoms(pdb_path, model_index=1):
    atoms = []
    has_models = False
    active_model = None
    with open(pdb_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                has_models = True
                active_model = int(line[10:14].strip())
                continue
            if record == "ENDMDL":
                if active_model == model_index:
                    break
                active_model = None
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            if has_models and active_model != model_index:
                continue
            atoms.append({
                "serial": int(line[6:11]),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "A",
                "resseq": int(line[22:26]),
                "coord": np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ], dtype=float),
            })
    return atoms

def project_structure_to_2d(atoms):
    coords = np.array([atom["coord"] for atom in atoms], dtype=float)
    coords = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    xy = coords @ vh[:2].T

    ca_idx = [idx for idx, atom in enumerate(atoms) if atom["name"] == "CA"]
    if ca_idx:
        if xy[ca_idx[0], 0] > xy[ca_idx[-1], 0]:
            xy[:, 0] *= -1
        if xy[ca_idx[2], 1] < xy[ca_idx[-2], 1]:
            xy[:, 1] *= -1
    return xy

def crop_nonwhite_rgba(image, pad_px=18):
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    alpha_mask = arr[..., 3] > 0
    color_mask = np.any(arr[..., :3] < 250, axis=2)
    mask = alpha_mask & color_mask
    if not np.any(mask):
        return rgba
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = max(int(rows[0]) - pad_px, 0)
    bottom = min(int(rows[-1]) + pad_px + 1, rgba.height)
    left = max(int(cols[0]) - pad_px, 0)
    right = min(int(cols[-1]) + pad_px + 1, rgba.width)
    return rgba.crop((left, top, right, bottom))

def generate_pymol_render(structure_path, render_path, reference_path=None, image_size=900):
    pymol_python = Path("/Applications/PyMOL.app/Contents/bin/python3.10")
    if not pymol_python.exists():
        return None

    structure_path = Path(structure_path).resolve()
    render_path = Path(render_path).resolve()
    render_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = None if reference_path is None else Path(reference_path).resolve()

    reference_block = ""
    if reference_path is not None:
        reference_block = textwrap.dedent(
            f"""
            cmd.load({str(reference_path)!r}, 'ref')
            cmd.remove('ref and hydro')
            cmd.remove('ref and not polymer.protein')
            cmd.align('mol and name CA', 'ref and name CA')
            cmd.disable('ref')
            """
        )

    script_body = textwrap.dedent(
        f"""
        from pathlib import Path
        import pymol2

        structure_path = Path({str(structure_path)!r})
        render_path = Path({str(render_path)!r})

        with pymol2.PyMOL() as pm:
            cmd = pm.cmd
            cmd.reinitialize()
            cmd.bg_color('white')
            cmd.set('ray_opaque_background', 0)
            cmd.set_color('state_red', [0.86, 0.44, 0.47])
            cmd.set_color('state_blue', [0.22, 0.38, 0.76])
            cmd.set_color('state_green', [0.18, 0.64, 0.47])
            cmd.set_color('state_gold', [0.86, 0.68, 0.27])
            cmd.set_color('backbone_gray', [0.83, 0.84, 0.86])
            cmd.load(str(structure_path), 'mol')
            cmd.remove('mol and hydro')
            cmd.remove('mol and solvent')
            cmd.remove('mol and not polymer.protein')
{textwrap.indent(reference_block, '            ')}
            cmd.hide('everything', 'all')
            cmd.show('cartoon', 'polymer.protein')
            cmd.color('backbone_gray', 'polymer.protein')
            cmd.set('cartoon_fancy_sheets', 1)
            cmd.set('cartoon_smooth_loops', 1)
            cmd.set('cartoon_side_chain_helper', 1)

            cmd.show('sticks', 'polymer.protein and resi 3+7+8+9')
            cmd.set('stick_radius', 0.18)
            cmd.color('state_red', 'resi 3')
            cmd.color('state_blue', 'resi 7')
            cmd.color('state_green', 'resi 8')
            cmd.color('state_gold', 'resi 9')

            cmd.show('spheres', 'id 31+92+106')
            cmd.set('sphere_scale', 0.28, 'id 31+92+106')
            cmd.color('state_red', 'id 31')
            cmd.color('state_blue', 'id 92')
            cmd.color('state_green', 'id 106')

            cmd.distance('d1_obj', 'id 31', 'id 92')
            cmd.distance('d2_obj', 'id 31', 'id 106')
            cmd.hide('labels', 'd1_obj d2_obj')
            cmd.set('dash_width', 2.6)
            cmd.set('dash_gap', 0.22)
            cmd.set('dash_length', 0.22)
            cmd.color('state_blue', 'd1_obj')
            cmd.color('state_green', 'd2_obj')

            cmd.orient('polymer.protein')
            cmd.turn('x', -18)
            cmd.turn('y', 18)
            cmd.turn('z', -28)
            cmd.zoom('polymer.protein', 1.8)
            cmd.set('ambient', 0.45)
            cmd.set('specular', 0.2)
            cmd.set('antialias', 2)
            cmd.ray({int(image_size)}, {int(image_size)})
            cmd.png(str(render_path), dpi=300)
        """
    )

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp", encoding="utf-8"
        ) as handle:
            handle.write(script_body)
            script_path = Path(handle.name)

        result = subprocess.run(
            [str(pymol_python), str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or "PyMOL render failed")
        if not render_path.exists():
            raise RuntimeError("PyMOL did not create the render image")
        return render_path
    finally:
        if script_path is not None and script_path.exists():
            script_path.unlink()

CA_ATOM_IDS = [5, 12, 33, 53, 59, 74, 88, 95, 109, 133]

def read_ref_ca_coords(path):
    points = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                points.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
    return np.asarray(points, dtype=float)

def read_gro_atoms(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    n_atoms = int(lines[1].strip())
    atoms = {}
    for line in lines[2:2 + n_atoms]:
        atom_id = int(line[15:20])
        atom_name = line[10:15].strip()
        atoms[atom_id] = {
            "name": atom_name,
            "coord": np.array([
                float(line[20:28]) * 10.0,
                float(line[28:36]) * 10.0,
                float(line[36:44]) * 10.0,
            ], dtype=float),
        }
    return atoms

def kabsch_rmsd(P, Q):
    P_centered = P - P.mean(axis=0, keepdims=True)
    Q_centered = Q - Q.mean(axis=0, keepdims=True)
    cov = P_centered.T @ Q_centered
    V, _, Wt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(V @ Wt))
    rotation = V @ np.diag([1.0, 1.0, sign]) @ Wt
    P_rot = P_centered @ rotation
    return np.sqrt(np.mean(np.sum((P_rot - Q_centered) ** 2, axis=1)))

def radius_of_gyration(coords):
    center = coords.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))

def select_state_structures(case_dir, state_specs):
    ref_ca = read_ref_ca_coords(case_dir / "template" / "ref_ca.pdb")
    structure_rows = []
    for gro_path in sorted(case_dir.glob("run*/w*/md_final.gro")):
        atoms = read_gro_atoms(gro_path)
        required_ids = set(CA_ATOM_IDS) | {31, 92, 106}
        if not required_ids.issubset(atoms):
            continue
        ca_coords = np.array([atoms[idx]["coord"] for idx in CA_ATOM_IDS], dtype=float)
        structure_rows.append({
            "path": gro_path,
            "rmsd": float(kabsch_rmsd(ca_coords, ref_ca)),
            "rg": radius_of_gyration(ca_coords),
            "d1": float(np.linalg.norm(atoms[31]["coord"] - atoms[92]["coord"])),
            "d2": float(np.linalg.norm(atoms[31]["coord"] - atoms[106]["coord"])),
        })

    selected = []
    used_paths = set()
    for spec in state_specs:
        ranked = sorted(
            structure_rows,
            key=lambda row: (row["rmsd"] - spec["xy"][0]) ** 2 + (row["rg"] - spec["xy"][1]) ** 2,
        )
        for row in ranked:
            if row["path"] in used_paths:
                continue
            selected.append({**spec, **row})
            used_paths.add(row["path"])
            break
    return selected

def draw_distance_legend(ax, label_fs=10):
    legend_items = [
        (0.10, 0.16, "#355C9A", r"d$_1$"),
        (0.10, 0.09, "#2F9E69", r"d$_2$"),
    ]
    for x0, y0, color, text in legend_items:
        ax.plot(
            [x0, x0 + 0.10], [y0, y0],
            transform=ax.transAxes,
            color=color,
            linestyle=(0, (3, 2)),
            lw=2.0,
            solid_capstyle="round",
            clip_on=False,
        )
        ax.text(
            x0 + 0.12, y0, text,
            transform=ax.transAxes,
            fontsize=label_fs,
            color=color,
            ha="left",
            va="center",
        )

def plot_chignolin_schematic(ax, pdb_path, title_fs=10.5, tick_fs=8, label_fs=10):
    atoms = load_pdb_atoms(pdb_path, model_index=1)
    xy = project_structure_to_2d(atoms)

    heavy_xy = np.array([
        xy[idx] for idx, atom in enumerate(atoms)
        if not atom["name"].startswith("H")
    ])
    ca_xy = np.array([
        xy[idx] for idx, atom in enumerate(atoms)
        if atom["name"] == "CA"
    ])
    serial_to_xy = {
        atom["serial"]: xy[idx]
        for idx, atom in enumerate(atoms)
    }

    ax.scatter(
        heavy_xy[:, 0], heavy_xy[:, 1],
        s=10, color="0.86", alpha=0.8, linewidths=0, zorder=1,
    )
    ax.plot(
        ca_xy[:, 0], ca_xy[:, 1],
        color="0.35", lw=2.6, solid_capstyle="round", zorder=2,
    )
    ax.scatter(
        ca_xy[:, 0], ca_xy[:, 1],
        c=np.linspace(0.15, 0.9, len(ca_xy)),
        cmap="cividis",
        s=32, edgecolors="white", linewidths=0.6, zorder=3,
    )

    highlight_specs = {
        31: dict(color="#C44E52", text="31", offset=(-0.35, 0.28)),
        92: dict(color="#4C72B0", text="92", offset=(0.28, -0.05)),
        106: dict(color="#55A868", text="106", offset=(0.28, 0.22)),
    }
    for serial, spec in highlight_specs.items():
        point = serial_to_xy[serial]
        ax.scatter(
            point[0], point[1],
            s=52, color=spec["color"], edgecolors="white", linewidths=0.8, zorder=5,
        )
        ax.text(
            point[0] + spec["offset"][0],
            point[1] + spec["offset"][1],
            spec["text"],
            color=spec["color"], fontsize=tick_fs, fontweight="bold",
            ha="center", va="center", zorder=6,
        )

    contact_specs = [
        (31, 92, "#4C72B0", r"d$_1$", (0.12, 0.10)),
        (31, 106, "#55A868", r"d$_2$", (-0.18, -0.16)),
    ]
    for start_serial, end_serial, color, label, offset in contact_specs:
        start = serial_to_xy[start_serial]
        end = serial_to_xy[end_serial]
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            linestyle=(0, (3, 2)), lw=1.6, color=color, alpha=0.95, zorder=4,
        )
        midpoint = 0.5 * (start + end) + np.array(offset)
        ax.text(
            midpoint[0], midpoint[1], label,
            color=color, fontsize=label_fs, ha="center", va="center", zorder=6,
        )

    ax.text(
        ca_xy[0, 0] - 0.28, ca_xy[0, 1] + 0.18, "N",
        fontsize=tick_fs, fontweight="bold", ha="center", va="center",
    )
    ax.text(
        ca_xy[-1, 0] + 0.30, ca_xy[-1, 1] + 0.16, "C",
        fontsize=tick_fs, fontweight="bold", ha="center", va="center",
    )

    ax.set_title("(a) Chignolin", loc="left", fontsize=title_fs)
    ax.set_aspect("equal")
    ax.axis("off")
    x_pad = 0.8
    y_pad = 0.8
    ax.set_xlim(ca_xy[:, 0].min() - x_pad, ca_xy[:, 0].max() + x_pad)
    ax.set_ylim(ca_xy[:, 1].min() - y_pad, ca_xy[:, 1].max() + y_pad)

def plot_chignolin_panel(ax, pdb_path, render_path, title_fs=10.5, tick_fs=8, label_fs=10):
    try:
        render_file = generate_pymol_render(
            pdb_path,
            render_path,
            reference_path=pdb_path,
            image_size=900,
        )
        if render_file is None:
            raise RuntimeError("PyMOL.app not found")
        img = crop_nonwhite_rgba(Image.open(render_file))
        ax.imshow(img)
        ax.set_title("(a) Chignolin", loc="left", fontsize=title_fs)
        ax.axis("off")
        draw_distance_legend(ax, label_fs=label_fs)
        return
    except Exception as exc:
        print(f"PyMOL render unavailable; falling back to schematic: {exc}")

    plot_chignolin_schematic(
        ax,
        pdb_path,
        title_fs=title_fs,
        tick_fs=tick_fs,
        label_fs=label_fs,
    )

def annotate_state_structures(ax, case_dir, fig_dir, state_specs, title_fs=10.5, tick_fs=8):
    try:
        selected_states = select_state_structures(case_dir, state_specs)
        for state in selected_states:
            render_path = Path(fig_dir) / f"fig3_state_{state['name']}_render.png"
            render_file = generate_pymol_render(
                state["path"],
                render_path,
                reference_path=case_dir / "template" / "processed.pdb",
                image_size=600,
            )
            if render_file is None:
                raise RuntimeError("PyMOL.app not found")
            img = crop_nonwhite_rgba(Image.open(render_file), pad_px=12)
            artist = AnnotationBbox(
                OffsetImage(img, zoom=state["zoom"]),
                state["xy"],
                xybox=state["xybox"],
                xycoords="data",
                boxcoords="data",
                frameon=True,
                pad=0.10,
                bboxprops=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec=state["edgecolor"],
                    lw=1.1,
                    alpha=0.96,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=state["edgecolor"],
                    lw=1.0,
                    alpha=0.9,
                ),
            )
            ax.add_artist(artist)
            ax.plot(
                state["xy"][0], state["xy"][1],
                marker="o", ms=3.5,
                color=state["edgecolor"],
                mec="white", mew=0.5,
                zorder=8,
            )
        return selected_states
    except Exception as exc:
        print(f"Representative state renders unavailable: {exc}")
        return []

FES_MAX = 12.0
N_BINS = 80
label_fs = 10
title_fs = 10.5
tick_fs = 8
annot_fs = 11



be_rmsd = np.linspace(0.5, 8.0, N_BINS + 1)
be_rg = np.linspace(3.8, 9.5, N_BINS + 1)
be_d1 = np.linspace(2.0, 16.0, N_BINS + 1)
be_d2 = np.linspace(2.0, 18.0, N_BINS + 1)

# Rebuild CV arrays locally so later notebook cells cannot overwrite them.
rmsd_fes = np.asarray(raw[:, 19], dtype=float) * 10
rg_fes = np.asarray(raw[:, 17], dtype=float) * 10
d1_fes = np.asarray(raw[:, 20], dtype=float) * 10
d2_fes = np.asarray(raw[:, 21], dtype=float) * 10
frame_weights = np.asarray(weights, dtype=float)

n_common = min(len(rmsd_fes), len(rg_fes), len(d1_fes), len(d2_fes), len(frame_weights))
rmsd_fes = rmsd_fes[:n_common]
rg_fes = rg_fes[:n_common]
d1_fes = d1_fes[:n_common]
d2_fes = d2_fes[:n_common]
frame_weights = frame_weights[:n_common]
frame_weights = frame_weights / frame_weights.sum()

cx_rmsd, cx_rg, fes_rmsd_rg = fes_2d(
    rmsd_fes, rg_fes, frame_weights, be_rmsd, be_rg,
    sigma_smooth=1.0, max_fes=FES_MAX,
)
cx_d1, cx_d2, fes_d1_d2 = fes_2d(
    d1_fes, d2_fes, frame_weights, be_d1, be_d2,
    sigma_smooth=1.0, max_fes=FES_MAX,
)

cmap = 'RdYlBu_r'
levels_fill = np.linspace(0, FES_MAX, 25)
levels_line = [1, 2, 4, 6, 8, 10]

fig = plt.figure(figsize=(10.8, 3.6), constrained_layout=True)
gs = GridSpec(1, 3, figure=fig, width_ratios=[0.72, 1.0, 1.0])
ax_mol = fig.add_subplot(gs[0, 0])
ax_fes_rg = fig.add_subplot(gs[0, 1])
ax_fes_d = fig.add_subplot(gs[0, 2])

plot_chignolin_panel(
    ax_mol,
    CASE_DIR / "template" / "processed.pdb",
    Path(FIG_DIR) / "fig3_chignolin_render.png",
    title_fs=title_fs,
    tick_fs=tick_fs,
    label_fs=label_fs,
)

# (a) RMSD_Ca vs Rg
ax = ax_fes_rg
X_rg, Y_rg = np.meshgrid(cx_rmsd, cx_rg, indexing='ij')
# np.histogram2d returns fes[x_bin, y_bin], which already matches this ij-grid.
im = ax.contourf(
    X_rg, Y_rg, fes_rmsd_rg,
    levels=levels_fill, cmap=cmap, extend='max',
)
ax.contour(
    X_rg, Y_rg, fes_rmsd_rg,
    levels=levels_line, colors='k', linewidths=0.4, alpha=0.6,
)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label('F (kJ mol$^{-1}$)', fontsize=label_fs)
cb.ax.tick_params(labelsize=tick_fs)

ax.set_xlabel(r'RMSD$_{\mathrm{C}\alpha}$ ($\AA$)', fontsize=label_fs)
ax.set_ylabel(r'R$_g$ ($\AA$)', fontsize=label_fs)
ax.set_title(r'(b) FES: RMSD$_{\mathrm{C}\alpha}$ vs R$_g$', loc='left', fontsize=title_fs)
ax.tick_params(labelsize=tick_fs)

state_annotations = [
    dict(name="F", xy=(1.55, 5.10), text='F', color='white', edgecolor="#355C9A", xybox=(1.90, 4.55), zoom=0.098),
    dict(name="I", xy=(3.15, 5.45), text='I', color='white', edgecolor="#6A56A5", xybox=(2.28, 6.15), zoom=0.102),
    dict(name="U1", xy=(4.35, 6.40), text=r'U$_1$', color='white', edgecolor="#3F8F6A", xybox=(3.78, 7.42), zoom=0.100),
    dict(name="U2", xy=(5.40, 7.30), text=r'U$_2$', color='white', edgecolor="#A55A4A", xybox=(6.38, 8.03), zoom=0.100),
]
annotate_state_structures(
    ax,
    CASE_DIR,
    FIG_DIR,
    state_annotations,
    title_fs=title_fs,
    tick_fs=tick_fs,
)
for ann in state_annotations:
    ax.text(
        *ann['xy'], ann['text'],
        fontsize=annot_fs, fontweight='bold', color=ann['color'],
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.22', fc=ann['edgecolor'], ec='none', alpha=0.78),
    )

# (b) d1 vs d2
ax = ax_fes_d
X_d, Y_d = np.meshgrid(cx_d1, cx_d2, indexing='ij')
im2 = ax.contourf(
    X_d, Y_d, fes_d1_d2,
    levels=levels_fill, cmap=cmap, extend='max',
)
ax.contour(
    X_d, Y_d, fes_d1_d2,
    levels=levels_line, colors='k', linewidths=0.4, alpha=0.6,
)
cb2 = fig.colorbar(im2, ax=ax, pad=0.02)
cb2.set_label('F (kJ mol$^{-1}$)', fontsize=label_fs)
cb2.ax.tick_params(labelsize=tick_fs)

ax.set_xlabel(r'd$_1$ ($\AA$) [atoms 31--92]', fontsize=label_fs)
ax.set_ylabel(r'd$_2$ ($\AA$) [atoms 31--106]', fontsize=label_fs)
ax.set_title('(c) FES: cross-strand contact distances', loc='left', fontsize=title_fs)
ax.tick_params(labelsize=tick_fs)

fig.savefig(Path(FIG_DIR) / 'fig3_2D_FES_metastable_states.pdf', bbox_inches='tight')
fig.savefig(Path(FIG_DIR) / 'fig3_2D_FES_metastable_states.png', dpi=300, bbox_inches='tight')
plt.show()
print('Figure 3 saved.')

# %%


# %%
