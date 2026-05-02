#!/usr/bin/env python3
"""
论文画图脚本：从已完成的 reweight 结果中读取数据，生成论文级别的图。

Figure A: Time series of x-coordinate (unbiased + multiple alpha)
Figure C: Marginal FES F(x) and F(y) convergence at different sampling lengths

用法:
    # 先生成 unbiased 轨迹（只需跑一次）
    python plot_paper_figures.py --generate_unbiased --n_steps 6000000

    # 画图（从已有 reweight 结果中读取）
    python plot_paper_figures.py \
        --reweight_dirs dir_alpha8 dir_alpha13 dir_alpha16 \
        --labels "alpha=8" "alpha=13" "alpha=16" \
        --unbiased_traj unbiased_traj.npz

    # 也可以同时生成 unbiased 并画图
    python plot_paper_figures.py \
        --generate_unbiased --n_steps 6000000 \
        --reweight_dirs dir_alpha8 dir_alpha13 dir_alpha16 \
        --labels "alpha=8" "alpha=13" "alpha=16"
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

# ============================================================
# Muller 势能
# ============================================================
MULLER_A = np.array([-200, -100, -170, 15])
MULLER_a = np.array([-1, -1, -6.5, 0.7])
MULLER_b = np.array([0, 0, 11, 0.6])
MULLER_c = np.array([-10, -10, -6.5, 0.7])
MULLER_x0 = np.array([1, 0, -0.5, -1])
MULLER_y0 = np.array([0, 0.5, 1.5, 1])

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-1.5, 2.5)


def muller_potential(x, y):
    V = 0.0
    for i in range(4):
        dx = x - MULLER_x0[i]
        dy = y - MULLER_y0[i]
        V += MULLER_A[i] * np.exp(MULLER_a[i]*dx**2 + MULLER_b[i]*dx*dy + MULLER_c[i]*dy**2)
    return V


def muller_gradient(x, y):
    scalar_input = np.isscalar(x) and np.isscalar(y)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    dV_dx = np.zeros_like(x_arr)
    dV_dy = np.zeros_like(y_arr)
    for i in range(4):
        dx = x_arr - MULLER_x0[i]
        dy = y_arr - MULLER_y0[i]
        exp_term = MULLER_A[i] * np.exp(MULLER_a[i]*dx**2 + MULLER_b[i]*dx*dy + MULLER_c[i]*dy**2)
        dV_dx += exp_term * (2*MULLER_a[i]*dx + MULLER_b[i]*dy)
        dV_dy += exp_term * (MULLER_b[i]*dx + 2*MULLER_c[i]*dy)
    grad = np.stack([dV_dx, dV_dy], axis=-1)
    return grad[0] if scalar_input else grad


# ============================================================
# Unbiased Langevin 模拟
# ============================================================

def run_unbiased_langevin(n_steps, dt, kBT, gamma, initial_pos=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = np.array(initial_pos if initial_pos is not None else [-0.55, 1.41])
    trajectory = np.empty((n_steps, 2))
    noise_coeff = np.sqrt(2 * kBT * dt / gamma)
    print_interval = max(n_steps // 10, 1)

    for step in range(n_steps):
        trajectory[step] = pos
        grad_V = muller_gradient(pos[0], pos[1])
        force_V = -grad_V
        noise = np.random.randn(2) * noise_coeff
        pos = pos + (force_V / gamma) * dt + noise
        pos[0] = np.clip(pos[0], X_RANGE[0], X_RANGE[1])
        pos[1] = np.clip(pos[1], Y_RANGE[0], Y_RANGE[1])

        if (step + 1) % print_interval == 0:
            print(f"  Unbiased progress: {(step+1)/n_steps*100:.0f}%")

    return trajectory


def count_transitions(trajectory):
    n_A_to_B = 0
    n_B_to_A = 0
    last_state = None
    for pos in trajectory:
        x, y = pos[0], pos[1]
        if x < -0.3 and y > 1.0:
            current = 'A'
        elif x > 0.4 and y < 0.3:
            current = 'B'
        else:
            continue
        if last_state is not None and current != last_state:
            if last_state == 'A':
                n_A_to_B += 1
            else:
                n_B_to_A += 1
        last_state = current
    return n_A_to_B, n_B_to_A


# ============================================================
# 理论 marginal FES
# ============================================================

def compute_theory_marginal_fes(kBT, x_range=(-1.5, 1.2), y_range=(-0.5, 2.0), n_grid=500):
    """计算理论 marginal FES: F(x) = -kBT * log( ∫ exp(-V/kBT) dy )"""
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)
    y_grid = np.linspace(y_range[0], y_range[1], n_grid)
    dy = y_grid[1] - y_grid[0]
    dx = x_grid[1] - x_grid[0]
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    V = muller_potential(X, Y)
    boltzmann = np.exp(-V / kBT)

    # F(x) = -kBT * log( ∫ exp(-V/kBT) dy )
    marginal_x = np.sum(boltzmann, axis=1) * dy
    fes_x = -kBT * np.log(marginal_x)
    fes_x -= fes_x.min()

    # F(y) = -kBT * log( ∫ exp(-V/kBT) dx )
    marginal_y = np.sum(boltzmann, axis=0) * dx
    fes_y = -kBT * np.log(marginal_y)
    fes_y -= fes_y.min()

    return x_grid, fes_x, y_grid, fes_y


def compute_reweighted_marginal_fes(trajectory, weights, kBT,
                                     x_range=(-1.5, 1.2), y_range=(-0.5, 2.0),
                                     bins=150):
    """从 reweighted 轨迹计算 marginal FES"""
    # F(x)
    hist_x, x_edges = np.histogram(trajectory[:, 0], bins=bins,
                                    range=x_range, weights=weights, density=True)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    fes_x = np.full_like(hist_x, np.nan)
    mask_x = hist_x > 0
    fes_x[mask_x] = -kBT * np.log(hist_x[mask_x])
    fes_x[mask_x] -= np.nanmin(fes_x)

    # F(y)
    hist_y, y_edges = np.histogram(trajectory[:, 1], bins=bins,
                                    range=y_range, weights=weights, density=True)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    fes_y = np.full_like(hist_y, np.nan)
    mask_y = hist_y > 0
    fes_y[mask_y] = -kBT * np.log(hist_y[mask_y])
    fes_y[mask_y] -= np.nanmin(fes_y)

    return x_centers, fes_x, y_centers, fes_y


def compute_unbiased_marginal_fes(trajectory, kBT,
                                   x_range=(-1.5, 1.2), y_range=(-0.5, 2.0),
                                   bins=150):
    """从 unbiased 轨迹计算 marginal FES（uniform weights）"""
    n = len(trajectory)
    weights = np.ones(n) / n
    return compute_reweighted_marginal_fes(trajectory, weights, kBT,
                                            x_range, y_range, bins)


# ============================================================
# Figure A: Time series of x-coordinate
# ============================================================

def plot_figure_a(unbiased_traj, biased_trajs, labels, dt, save_path,
                  thin_factor=10, max_display=None):
    """
    画 time series of x-coordinate.

    Parameters:
        unbiased_traj: (N, 2) unbiased trajectory
        biased_trajs: list of (N, 2) biased trajectories
        labels: list of str, labels for biased trajectories
        dt: simulation timestep
        save_path: output path
        thin_factor: 每隔多少步画一个点（避免图太密）
        max_display: 最多显示多少步（None = 全部）
    """
    n_panels = 1 + len(biased_trajs)
    n_cols = 2 if n_panels == 4 else n_panels
    n_rows = 2 if n_panels == 4 else 1
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3.2 * n_rows),
                              sharey=True)
    axes_flat = axes.flatten() if n_panels == 4 else (axes if n_panels > 1 else [axes])

    all_trajs = [unbiased_traj] + biased_trajs
    all_labels = ['Unbiased'] + labels
    panel_labels = [chr(ord('a') + i) for i in range(n_panels)]

    for i, (traj, label, plabel) in enumerate(zip(all_trajs, all_labels, panel_labels)):
        ax = axes_flat[i]

        x = traj[:, 0]
        if max_display is not None:
            x = x[:max_display]

        n_pts = len(x)
        t = np.arange(n_pts) * dt
        # thin for plotting
        idx = np.arange(0, n_pts, thin_factor)
        t_thin = t[idx]
        x_thin = x[idx]

        ax.scatter(t_thin, x_thin, s=0.05, alpha=0.4, color='#1f77b4', rasterized=True)
        ax.set_xlabel(r'$t$ [a.u.]', fontsize=11)
        if i % n_cols == 0:
            ax.set_ylabel(r'$x$', fontsize=13)

        n_ab, n_ba = count_transitions(traj[:len(x)])
        total = n_ab + n_ba

        ax.text(0.03, 0.97, f'({plabel}) {label}',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.text(0.97, 0.97, f'transitions: {total}',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_ylim(-1.8, 1.8)
        ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.6, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # 简化 x 轴刻度标签
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Figure A: {save_path}")


# ============================================================
# Figure C: Marginal FES convergence
# ============================================================

def plot_figure_c(unbiased_traj, biased_trajs, biased_weights_list, labels,
                  kBT, save_path, sampling_fractions=None,
                  x_range=(-1.5, 1.2), y_range=(-0.5, 2.0)):
    """
    画 marginal FES F(x) 和 F(y) 在不同采样长度下的收敛。
    类似合作者 Figure 2 的布局：上行 F(x)，下行 F(y)，列为不同采样长度。

    Parameters:
        unbiased_traj: (N, 2) unbiased trajectory
        biased_trajs: list of (N, 2) biased trajectories
        biased_weights_list: list of (N,) reweighting weights
        labels: list of str
        kBT: thermal energy
        sampling_fractions: list of float, fractions of total trajectory to use
    """
    if sampling_fractions is None:
        sampling_fractions = [0.1, 0.25, 0.5, 1.0]

    n_cols = len(sampling_fractions)

    # 理论 FES
    x_theory, fes_x_theory, y_theory, fes_y_theory = \
        compute_theory_marginal_fes(kBT, x_range, y_range)

    # 颜色方案（与合作者保持一致）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), sharey='row')

    for col, frac in enumerate(sampling_fractions):
        ax_x = axes[0, col]
        ax_y = axes[1, col]

        n_total_unbiased = len(unbiased_traj)
        n_use_unbiased = int(n_total_unbiased * frac)

        # 理论（黑色虚线）
        ax_x.plot(x_theory, fes_x_theory, 'k--', linewidth=1.5, alpha=0.7,
                  label='Theory' if col == 0 else None)
        ax_y.plot(y_theory, fes_y_theory, 'k--', linewidth=1.5, alpha=0.7,
                  label='Theory' if col == 0 else None)

        # Unbiased
        traj_sub = unbiased_traj[:n_use_unbiased]
        xc, fx, yc, fy = compute_unbiased_marginal_fes(traj_sub, kBT, x_range, y_range)
        ax_x.plot(xc, fx, color=colors[0], linewidth=1.2,
                  label='Unbiased' if col == 0 else None)
        ax_y.plot(yc, fy, color=colors[0], linewidth=1.2,
                  label='Unbiased' if col == 0 else None)

        # Biased
        for j, (traj, w, lab) in enumerate(zip(biased_trajs, biased_weights_list, labels)):
            n_use = int(len(traj) * frac)
            traj_sub = traj[:n_use]
            w_sub = w[:n_use]
            w_sub = w_sub / w_sub.sum()

            xc, fx, yc, fy = compute_reweighted_marginal_fes(
                traj_sub, w_sub, kBT, x_range, y_range)
            ax_x.plot(xc, fx, color=colors[j + 1], linewidth=1.2,
                      label=lab if col == 0 else None)
            ax_y.plot(yc, fy, color=colors[j + 1], linewidth=1.2,
                      label=lab if col == 0 else None)

        # 标注
        n_steps_show = int(len(biased_trajs[0]) * frac) if biased_trajs else n_use_unbiased
        step_str = f'{n_steps_show/1e6:.1f}M steps' if n_steps_show >= 1e6 else f'{n_steps_show/1e3:.0f}K steps'

        plabel_x = chr(ord('a') + col)
        plabel_y = chr(ord('e') + col)
        ax_x.text(0.03, 0.97, f'({plabel_x}) {step_str}',
                  transform=ax_x.transAxes, fontsize=11, va='top', fontweight='bold')
        ax_y.text(0.03, 0.97, f'({plabel_y}) {step_str}',
                  transform=ax_y.transAxes, fontsize=11, va='top', fontweight='bold')

        ax_y.set_xlabel(r'Coordinate', fontsize=11)

        if col == 0:
            ax_x.set_ylabel(r'$F(x)$', fontsize=13)
            ax_y.set_ylabel(r'$F(y)$', fontsize=13)

    # 统一 y 轴范围
    for row in range(2):
        ymax = 0
        for col in range(n_cols):
            ylims = axes[row, col].get_ylim()
            ymax = max(ymax, ylims[1])
        ymax = min(ymax, 80)  # cap
        for col in range(n_cols):
            axes[row, col].set_ylim(-2, ymax)

    # Legend
    axes[0, 0].legend(fontsize=9, loc='upper right',
                       framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Figure C: {save_path}")


# ============================================================
# Figure D (bonus): RMSE convergence curve
# ============================================================

def plot_convergence_rmse(unbiased_traj, biased_trajs, biased_weights_list, labels,
                          kBT, save_path,
                          x_range=(-1.5, 1.2), y_range=(-0.5, 2.0),
                          n_points=20):
    """
    画 marginal FES 的 RMSE 随采样量的收敛曲线。
    """
    x_theory, fes_x_theory, y_theory, fes_y_theory = \
        compute_theory_marginal_fes(kBT, x_range, y_range)

    fracs = np.linspace(0.05, 1.0, n_points)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Compute RMSE for unbiased
    rmse_x_unbiased = []
    rmse_y_unbiased = []
    for frac in fracs:
        n_use = int(len(unbiased_traj) * frac)
        traj_sub = unbiased_traj[:n_use]
        xc, fx, yc, fy = compute_unbiased_marginal_fes(traj_sub, kBT, x_range, y_range)
        # interpolate theory onto histogram grid
        fx_theory_interp = np.interp(xc, x_theory, fes_x_theory)
        fy_theory_interp = np.interp(yc, y_theory, fes_y_theory)
        mask_x = np.isfinite(fx)
        mask_y = np.isfinite(fy)
        rmse_x_unbiased.append(np.sqrt(np.mean((fx[mask_x] - fx_theory_interp[mask_x])**2)) if mask_x.any() else np.nan)
        rmse_y_unbiased.append(np.sqrt(np.mean((fy[mask_y] - fy_theory_interp[mask_y])**2)) if mask_y.any() else np.nan)

    n_ref = len(biased_trajs[0]) if biased_trajs else len(unbiased_traj)
    steps_axis = fracs * n_ref

    axes[0].plot(steps_axis / 1e6, rmse_x_unbiased, color=colors[0],
                 linewidth=1.5, marker='o', markersize=3, label='Unbiased')
    axes[1].plot(steps_axis / 1e6, rmse_y_unbiased, color=colors[0],
                 linewidth=1.5, marker='o', markersize=3, label='Unbiased')

    # Compute RMSE for each biased
    for j, (traj, w, lab) in enumerate(zip(biased_trajs, biased_weights_list, labels)):
        rmse_x_list = []
        rmse_y_list = []
        for frac in fracs:
            n_use = int(len(traj) * frac)
            traj_sub = traj[:n_use]
            w_sub = w[:n_use]
            w_sub = w_sub / w_sub.sum()
            xc, fx, yc, fy = compute_reweighted_marginal_fes(
                traj_sub, w_sub, kBT, x_range, y_range)
            fx_theory_interp = np.interp(xc, x_theory, fes_x_theory)
            fy_theory_interp = np.interp(yc, y_theory, fes_y_theory)
            mask_x = np.isfinite(fx)
            mask_y = np.isfinite(fy)
            rmse_x_list.append(np.sqrt(np.mean((fx[mask_x] - fx_theory_interp[mask_x])**2)) if mask_x.any() else np.nan)
            rmse_y_list.append(np.sqrt(np.mean((fy[mask_y] - fy_theory_interp[mask_y])**2)) if mask_y.any() else np.nan)

        steps_axis_b = fracs * len(traj)
        axes[0].plot(steps_axis_b / 1e6, rmse_x_list, color=colors[j + 1],
                     linewidth=1.5, marker='o', markersize=3, label=lab)
        axes[1].plot(steps_axis_b / 1e6, rmse_y_list, color=colors[j + 1],
                     linewidth=1.5, marker='o', markersize=3, label=lab)

    axes[0].set_xlabel('Steps [M]', fontsize=11)
    axes[0].set_ylabel(r'RMSE of $F(x)$', fontsize=12)
    axes[0].set_title(r'(a) Marginal FES $F(x)$ convergence', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9, framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Steps [M]', fontsize=11)
    axes[1].set_ylabel(r'RMSE of $F(y)$', fontsize=12)
    axes[1].set_title(r'(b) Marginal FES $F(y)$ convergence', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9, framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved convergence: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='从 reweight 结果画论文图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--reweight_dirs', type=str, nargs='+', default=[],
                        help='已完成的 reweight 输出目录列表（包含 results.npz）')
    parser.add_argument('--labels', type=str, nargs='+', default=[],
                        help='每个 reweight 结果的标签（如 alpha=8 alpha=13 alpha=16）')

    parser.add_argument('--unbiased_traj', type=str, default='unbiased_traj.npz',
                        help='unbiased 轨迹文件路径')
    parser.add_argument('--generate_unbiased', action='store_true',
                        help='是否生成 unbiased 轨迹')
    parser.add_argument('--n_steps', type=int, default=6000000,
                        help='unbiased 轨迹步数')
    parser.add_argument('--dt', type=float, default=0.0005, help='timestep')
    parser.add_argument('--kBT', type=float, default=2.5, help='kBT')
    parser.add_argument('--gamma', type=float, default=5.0, help='friction')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--output_dir', type=str, default='paper_figures',
                        help='输出目录')

    parser.add_argument('--sampling_fractions', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 1.0],
                        help='Figure C 的采样比例列表')

    parser.add_argument('--thin', type=int, default=10,
                        help='Figure A 的 thinning factor')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("Paper Figure Generator")
    print("=" * 60)

    # ========== Unbiased 轨迹 ==========
    if args.generate_unbiased:
        print(f"\n[1] Generating unbiased trajectory ({args.n_steps} steps)...")
        unbiased_traj = run_unbiased_langevin(
            n_steps=args.n_steps, dt=args.dt, kBT=args.kBT,
            gamma=args.gamma, seed=args.seed)
        np.savez(args.unbiased_traj, trajectory=unbiased_traj)
        print(f"  Saved: {args.unbiased_traj}")
        n_ab, n_ba = count_transitions(unbiased_traj)
        print(f"  Unbiased transitions: A->B={n_ab}, B->A={n_ba}")
    else:
        print(f"\n[1] Loading unbiased trajectory: {args.unbiased_traj}")
        if not os.path.exists(args.unbiased_traj):
            print(f"  Error: {args.unbiased_traj} not found!")
            print(f"  Run with --generate_unbiased first.")
            sys.exit(1)
        data = np.load(args.unbiased_traj)
        unbiased_traj = data['trajectory']
        n_ab, n_ba = count_transitions(unbiased_traj)
        print(f"  Loaded {len(unbiased_traj)} points, transitions: A->B={n_ab}, B->A={n_ba}")

    # ========== 加载 reweight 结果 ==========
    biased_trajs = []
    biased_weights = []
    actual_labels = []

    if args.reweight_dirs:
        print(f"\n[2] Loading reweight results...")
        for i, rdir in enumerate(args.reweight_dirs):
            results_path = os.path.join(rdir, 'results.npz')
            if not os.path.exists(results_path):
                print(f"  Warning: {results_path} not found, skipping...")
                continue

            data = np.load(results_path, allow_pickle=True)
            traj = data['trajectory']
            w = data['weights']
            summary = data['summary'].item() if 'summary' in data else {}

            label = args.labels[i] if i < len(args.labels) else f'run_{i}'
            actual_labels.append(label)
            biased_trajs.append(traj)
            biased_weights.append(w)

            n_ab, n_ba = count_transitions(traj)
            alpha_val = summary.get('alpha', '?')
            eps_val = summary.get('eps', '?')
            tau_val = summary.get('tau', '?')
            ess = summary.get('ess', '?')
            print(f"  [{label}] {len(traj)} pts, alpha={alpha_val}, eps={eps_val}, "
                  f"tau={tau_val}, transitions={n_ab+n_ba}, ESS={ess}")
    else:
        print("\n[2] No reweight directories provided, skipping biased data.")

    # ========== Figure A: Time series ==========
    if biased_trajs:
        print(f"\n[3] Plotting Figure A (time series)...")
        plot_figure_a(
            unbiased_traj, biased_trajs, actual_labels,
            dt=args.dt,
            save_path=os.path.join(args.output_dir, "figure_a_timeseries.png"),
            thin_factor=args.thin,
        )

    # ========== Figure C: Marginal FES convergence ==========
    if biased_trajs:
        print(f"\n[4] Plotting Figure C (marginal FES convergence)...")
        plot_figure_c(
            unbiased_traj, biased_trajs, biased_weights, actual_labels,
            kBT=args.kBT,
            save_path=os.path.join(args.output_dir, "figure_c_marginal_fes.png"),
            sampling_fractions=args.sampling_fractions,
        )

    # ========== Bonus: RMSE convergence ==========
    if biased_trajs:
        print(f"\n[5] Plotting RMSE convergence...")
        plot_convergence_rmse(
            unbiased_traj, biased_trajs, biased_weights, actual_labels,
            kBT=args.kBT,
            save_path=os.path.join(args.output_dir, "figure_rmse_convergence.png"),
        )

    print(f"\n{'='*60}")
    print(f"Done! All figures saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
