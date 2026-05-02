#!/usr/bin/env python3
"""
从已完成的 reweight 结果重新画 FES 对比图。
直接读取 results.npz，不需要重新运行模拟。

用法:
    python replot_fes.py --dirs dir1 dir2 dir3 ...
    python replot_fes.py --dirs reweight_4434928_kbt2.5_alpha13_...
    python replot_fes.py --dirs reweight_*/   # 通配符批量

可选参数:
    --vmax        FES colorbar 上限 (默认 80)
    --bins        直方图 bin 数 (默认 150)
    --x_range     x 范围 (默认 -1.5 1.2)
    --y_range     y 范围 (默认 -0.5 2.0)
    --suffix      输出文件名后缀 (默认 _replot)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Muller 势能
# ============================================================
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
        V += MULLER_A[i] * np.exp(MULLER_a[i]*dx**2 + MULLER_b[i]*dx*dy + MULLER_c[i]*dy**2)
    return V


# ============================================================
# 画图
# ============================================================

def plot_fes(trajectory, weights, kBT, save_path,
             x_range=(-1.5, 1.2), y_range=(-0.5, 2.0),
             bins=150, vmax=80):

    hist, x_edges, y_edges = np.histogram2d(
        trajectory[:, 0], trajectory[:, 1],
        bins=bins, range=[x_range, y_range],
        weights=weights, density=True)

    fes_sim = np.full_like(hist, np.nan)
    mask = hist > 0
    fes_sim[mask] = -kBT * np.log(hist[mask])
    fes_sim = fes_sim.T
    mask = mask.T

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)
    fes_theory = muller_potential(X, Y)

    min_sim = np.nanmin(fes_sim)
    min_theory = np.min(fes_theory[mask])
    fes_sim_aligned = fes_sim - min_sim
    fes_theory_aligned = fes_theory - min_theory

    fes_diff = np.full_like(fes_sim, np.nan)
    fes_diff[mask] = fes_sim_aligned[mask] - fes_theory_aligned[mask]

    rmse = np.sqrt(np.nanmean(fes_diff**2))
    mae = np.nanmean(np.abs(fes_diff))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    levels = np.linspace(0, vmax, 40)

    fes_theory_masked = np.where(mask, fes_theory_aligned, np.nan)
    c1 = axes[0].contourf(X, Y, fes_theory_masked, levels=levels, cmap='jet', extend='max')
    axes[0].set_title('Reference FES', fontsize=13, weight='bold')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(c1, ax=axes[0], label='Free Energy')

    c2 = axes[1].contourf(X, Y, fes_sim_aligned, levels=levels, cmap='jet', extend='max')
    axes[1].set_title('Reweighted FES', fontsize=13, weight='bold')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(c2, ax=axes[1], label='Free Energy')

    diff_limit = min(np.nanmax(np.abs(fes_diff)), 10.0)
    levels_diff = np.linspace(-diff_limit, diff_limit, 41)
    c3 = axes[2].contourf(X, Y, fes_diff, levels=levels_diff, cmap='RdBu_r', extend='both')
    axes[2].set_title('FES Difference', fontsize=13, weight='bold')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(c3, ax=axes[2], label='Delta F')
    axes[2].text(0.02, 0.98, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
                 transform=axes[2].transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}  (RMSE={rmse:.3f}, MAE={mae:.3f})")
    return rmse, mae


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='从 results.npz 重新画 FES 对比图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help='reweight 输出目录列表（包含 results.npz）')
    parser.add_argument('--vmax', type=float, default=80,
                        help='FES colorbar 上限')
    parser.add_argument('--bins', type=int, default=150,
                        help='直方图 bin 数')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1.5, 1.2],
                        help='x 范围')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-0.5, 2.0],
                        help='y 范围')
    parser.add_argument('--suffix', type=str, default='_replot',
                        help='输出文件名后缀（保存在原目录内）')
    args = parser.parse_args()

    print(f"vmax={args.vmax}, bins={args.bins}, suffix='{args.suffix}'")
    print("=" * 60)

    ok, failed = 0, 0
    for d in args.dirs:
        d = d.rstrip('/')
        npz_path = os.path.join(d, 'results.npz')
        if not os.path.exists(npz_path):
            print(f"[skip] {npz_path} not found")
            failed += 1
            continue

        print(f"\nProcessing: {d}")
        data = np.load(npz_path, allow_pickle=True)
        traj = data['trajectory']
        weights = data['weights']

        # 从 summary 读取 kBT
        summary = data['summary'].item() if 'summary' in data else {}
        kBT = float(summary.get('kBT', 2.5))
        print(f"  kBT={kBT}, n_points={len(traj)}, "
              f"alpha={summary.get('alpha','?')}, "
              f"eps={summary.get('eps','?')}, "
              f"tau={summary.get('tau','?')}")

        save_path = os.path.join(d, f'fes_comparison{args.suffix}.png')
        plot_fes(traj, weights, kBT, save_path,
                 x_range=tuple(args.x_range),
                 y_range=tuple(args.y_range),
                 bins=args.bins, vmax=args.vmax)
        ok += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok} plotted, {failed} skipped")


if __name__ == '__main__':
    main()
