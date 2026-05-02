#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.ndimage import gaussian_filter

mpl.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT_DIR / "1UAO_3" / "case2"
OUT_DIR = CASE_DIR / "mbar_1d_marginal_T300K"
FIG_DIR = Path(__file__).resolve().parent
PYMOL_PYTHON = Path("/Applications/PyMOL.app/Contents/bin/python3.10")

kB = 8.31446261815324e-3
T = 300.0
kBT = kB * T

FES_MAX = 12.0
N_BINS = 80
STRIDE = 20
N_RUNS = 65

CA_ATOM_IDS = [5, 12, 33, 53, 59, 74, 88, 95, 109, 133]

STATE_SPECS = [
    dict(name="F", xy=(1.55, 5.10), text="F", edgecolor="#355C9A", xybox=(2.05, 4.62), zoom=0.088),
    dict(name="I", xy=(3.15, 5.45), text="I", edgecolor="#7057A3", xybox=(2.48, 6.10), zoom=0.088),
    dict(name="U1", xy=(4.35, 6.40), text=r"U$_1$", edgecolor="#3D9169", xybox=(3.98, 7.40), zoom=0.086),
    dict(name="U2", xy=(5.40, 7.30), text=r"U$_2$", edgecolor="#B2644D", xybox=(6.10, 8.10), zoom=0.086),
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 2.8,
        "ytick.minor.size": 2.8,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def fes_2d(x, y, sample_weights, bin_edges_x, bin_edges_y, sigma_smooth=1.0, max_fes=None):
    sample_weights = np.asarray(sample_weights, dtype=float)
    sample_weights = sample_weights / sample_weights.sum()

    hist, xedges, yedges = np.histogram2d(
        x,
        y,
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


def read_ref_ca_coords(path):
    points = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                points.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
    return np.asarray(points, dtype=float)


def read_gro_atoms(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    n_atoms = int(lines[1].strip())
    atoms = {}
    for line in lines[2 : 2 + n_atoms]:
        atom_id = int(line[15:20])
        atoms[atom_id] = {
            "name": line[10:15].strip(),
            "coord": np.array(
                [
                    float(line[20:28]) * 10.0,
                    float(line[28:36]) * 10.0,
                    float(line[36:44]) * 10.0,
                ],
                dtype=float,
            ),
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
    return float(np.sqrt(np.mean(np.sum((P_rot - Q_centered) ** 2, axis=1))))


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
        structure_rows.append(
            {
                "path": gro_path,
                "rmsd": kabsch_rmsd(ca_coords, ref_ca),
                "rg": radius_of_gyration(ca_coords),
                "d1": float(np.linalg.norm(atoms[31]["coord"] - atoms[92]["coord"])),
                "d2": float(np.linalg.norm(atoms[31]["coord"] - atoms[106]["coord"])),
            }
        )

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


def _pymol_script_body(structure_path, render_path, reference_path, mode, image_size):
    show_contacts = mode == "overview"
    structure_path = Path(structure_path).resolve()
    render_path = Path(render_path).resolve()
    reference_path = Path(reference_path).resolve()

    contact_block = ""
    if show_contacts:
        contact_block = textwrap.dedent(
            """
            cmd.show('spheres', 'mol and id 31+92+106')
            cmd.set('sphere_scale', 0.30, 'mol and id 31+92+106')
            cmd.color('state_red', 'mol and id 31')
            cmd.color('state_blue', 'mol and id 92')
            cmd.color('state_green', 'mol and id 106')

            cmd.distance('d1_obj', 'mol and id 31', 'mol and id 92')
            cmd.distance('d2_obj', 'mol and id 31', 'mol and id 106')
            cmd.hide('labels', 'd1_obj d2_obj')
            cmd.set('dash_width', 2.5)
            cmd.set('dash_gap', 0.20)
            cmd.set('dash_length', 0.22)
            cmd.set('dash_round_ends', 1)
            cmd.color('state_blue', 'd1_obj')
            cmd.color('state_green', 'd2_obj')

            a31 = np.array(cmd.get_atom_coords('mol and id 31'))
            a92 = np.array(cmd.get_atom_coords('mol and id 92'))
            a106 = np.array(cmd.get_atom_coords('mol and id 106'))

            def add_label(name, pos, text, color):
                cmd.pseudoatom(name, pos=pos.tolist(), label=text)
                cmd.hide('everything', name)
                cmd.set('label_color', color, name)

            add_label('lab_31', a31 + np.array([-0.55, 0.35, 0.10]), '31', 'state_red')
            add_label('lab_92', a92 + np.array([0.55, -0.05, 0.10]), '92', 'state_blue')
            add_label('lab_106', a106 + np.array([0.55, 0.30, -0.05]), '106', 'state_green')
            add_label('lab_d1', 0.5 * (a31 + a92) + np.array([0.25, 0.45, 0.10]), 'd1', 'state_blue')
            add_label('lab_d2', 0.5 * (a31 + a106) + np.array([0.25, -0.45, 0.05]), 'd2', 'state_green')
            """
        ).strip()

    return textwrap.dedent(
        f"""
        from pathlib import Path
        import numpy as np
        import pymol2

        structure_path = Path({str(structure_path)!r})
        render_path = Path({str(render_path)!r})
        reference_path = Path({str(reference_path)!r})

        with pymol2.PyMOL() as pm:
            cmd = pm.cmd
            cmd.reinitialize()
            cmd.bg_color('white')
            cmd.set('ray_opaque_background', 0)
            cmd.set('orthoscopic', 1)
            cmd.set('depth_cue', 0)
            cmd.set('antialias', 2)
            cmd.set('ambient', 0.42)
            cmd.set('specular', 0.15)
            cmd.set('label_size', 18)
            cmd.set('label_font_id', 7)
            cmd.set('label_outline_color', 'white')
            cmd.set_color('backbone_gray', [0.84, 0.85, 0.88])
            cmd.set_color('state_red', [0.86, 0.43, 0.47])
            cmd.set_color('state_blue', [0.22, 0.38, 0.76])
            cmd.set_color('state_green', [0.18, 0.64, 0.47])
            cmd.set_color('state_gold', [0.84, 0.66, 0.24])
            cmd.set_color('state_ink', [0.38, 0.40, 0.44])

            cmd.load(str(reference_path), 'ref')
            cmd.remove('ref and hydro')
            cmd.remove('ref and solvent')
            cmd.remove('ref and not polymer.protein')

            cmd.load(str(structure_path), 'mol')
            cmd.remove('mol and hydro')
            cmd.remove('mol and solvent')
            cmd.remove('mol and not polymer.protein')
            cmd.align('mol and name CA', 'ref and name CA')

            cmd.hide('everything', 'all')
            cmd.show('cartoon', 'mol and polymer.protein')
            cmd.color('backbone_gray', 'mol and polymer.protein')
            cmd.set('cartoon_fancy_sheets', 1)
            cmd.set('cartoon_smooth_loops', 1)
            cmd.set('cartoon_side_chain_helper', 1)
            cmd.set('cartoon_transparency', 0.08)

            cmd.show('sticks', 'mol and resi 2+3+7+8+9')
            cmd.set('stick_radius', 0.16)
            cmd.color('state_gold', 'mol and resi 2')
            cmd.color('state_red', 'mol and resi 3')
            cmd.color('state_blue', 'mol and resi 7')
            cmd.color('state_green', 'mol and resi 8')
            cmd.color('state_ink', 'mol and resi 9')

{textwrap.indent(contact_block, '            ')}
            cmd.orient('ref and polymer.protein')
            cmd.turn('x', -18)
            cmd.turn('y', 22)
            cmd.turn('z', -28)
            cmd.zoom('mol and polymer.protein', {1.72 if show_contacts else 1.84})
            cmd.disable('ref')
            cmd.ray({int(image_size)}, {int(image_size)})
            cmd.png(str(render_path), dpi=300)
        """
    )


def render_structure(structure_path, render_path, reference_path, mode="thumbnail", image_size=900):
    if not PYMOL_PYTHON.exists():
        raise RuntimeError("PyMOL.app is not available")

    render_path = Path(render_path)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    script_body = _pymol_script_body(structure_path, render_path, reference_path, mode, image_size)

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir="/tmp",
            encoding="utf-8",
        ) as handle:
            handle.write(script_body)
            script_path = Path(handle.name)

        result = subprocess.run(
            [str(PYMOL_PYTHON), str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or "PyMOL rendering failed")
        if not render_path.exists():
            raise RuntimeError("PyMOL did not create the output image")
    finally:
        if script_path is not None and script_path.exists():
            script_path.unlink()

    return render_path


def load_unbiased_data():
    csv_data = np.loadtxt(OUT_DIR / "mbar_unbiased_samples_with_weights.csv", delimiter=",", skiprows=1)
    weights = csv_data[:, -1]
    blocks = [np.load(CASE_DIR / f"run{rid}/COLVAR.npy")[::STRIDE] for rid in range(1, N_RUNS + 1)]
    raw = np.vstack(blocks)
    return raw, weights


def annotate_state_insets(ax, state_rows, reference_path):
    for state in state_rows:
        render_path = FIG_DIR / f"figure_1uao_state_{state['name']}.png"
        render_structure(
            state["path"],
            render_path,
            reference_path=reference_path,
            mode="thumbnail",
            image_size=560,
        )
        img = crop_nonwhite_rgba(Image.open(render_path), pad_px=12)
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
                lw=1.2,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=state["edgecolor"],
                lw=1.0,
                alpha=0.85,
            ),
        )
        ax.add_artist(artist)
        ax.plot(
            state["xy"][0],
            state["xy"][1],
            marker="o",
            ms=3.5,
            color=state["edgecolor"],
            mec="white",
            mew=0.5,
            zorder=7,
        )
        ax.text(
            state["xy"][0],
            state["xy"][1],
            state["text"],
            fontsize=12,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc=state["edgecolor"], ec="none", alpha=0.80),
            zorder=8,
        )


def draw_panel_a(ax, reference_path):
    render_path = FIG_DIR / "figure_1uao_native_contacts.png"
    render_structure(
        reference_path,
        render_path,
        reference_path=reference_path,
        mode="overview",
        image_size=960,
    )
    img = crop_nonwhite_rgba(Image.open(render_path), pad_px=20)
    ax.imshow(img)
    ax.set_title("(a) Native contact definitions", loc="left")
    ax.axis("off")
    overlay_labels = [
        (0.56, 0.55, "31", "#D66A73"),
        (0.58, 0.33, "92", "#355C9A"),
        (0.66, 0.54, "106", "#3D9169"),
    ]
    for x0, y0, text, color in overlay_labels:
        ax.text(
            x0,
            y0,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.75),
        )


def main():
    raw, weights = load_unbiased_data()

    rmsd = np.asarray(raw[:, 19], dtype=float) * 10.0
    rg = np.asarray(raw[:, 17], dtype=float) * 10.0
    d1 = np.asarray(raw[:, 20], dtype=float) * 10.0
    d2 = np.asarray(raw[:, 21], dtype=float) * 10.0
    weights = np.asarray(weights, dtype=float)

    n_common = min(len(rmsd), len(rg), len(d1), len(d2), len(weights))
    rmsd = rmsd[:n_common]
    rg = rg[:n_common]
    d1 = d1[:n_common]
    d2 = d2[:n_common]
    weights = weights[:n_common]
    weights /= weights.sum()

    cx_rmsd, cx_rg, fes_rmsd_rg = fes_2d(
        rmsd,
        rg,
        weights,
        np.linspace(0.5, 8.0, N_BINS + 1),
        np.linspace(3.8, 9.5, N_BINS + 1),
        sigma_smooth=1.0,
        max_fes=FES_MAX,
    )
    cx_d1, cx_d2, fes_d1_d2 = fes_2d(
        d1,
        d2,
        weights,
        np.linspace(2.0, 16.0, N_BINS + 1),
        np.linspace(2.0, 18.0, N_BINS + 1),
        sigma_smooth=1.0,
        max_fes=FES_MAX,
    )

    selected_states = select_state_structures(CASE_DIR, STATE_SPECS)
    reference_path = CASE_DIR / "template" / "processed.pdb"

    fig = plt.figure(figsize=(11.8, 4.0), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.92, 1.10, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_panel_a(ax_a, reference_path)

    cmap = "RdYlBu_r"
    levels_fill = np.linspace(0, FES_MAX, 25)
    levels_line = [1, 2, 4, 6, 8, 10]

    X_rg, Y_rg = np.meshgrid(cx_rmsd, cx_rg, indexing="ij")
    im = ax_b.contourf(X_rg, Y_rg, fes_rmsd_rg, levels=levels_fill, cmap=cmap, extend="max")
    ax_b.contour(X_rg, Y_rg, fes_rmsd_rg, levels=levels_line, colors="k", linewidths=0.4, alpha=0.5)
    annotate_state_insets(ax_b, selected_states, reference_path)
    cb = fig.colorbar(im, ax=ax_b, pad=0.02)
    cb.set_label(r"F (kJ mol$^{-1}$)")
    cb.ax.tick_params(labelsize=9)

    ax_b.set_xlabel(r"RMSD$_{\mathrm{C}\alpha}$ ($\AA$)")
    ax_b.set_ylabel(r"R$_g$ ($\AA$)")
    ax_b.set_title(r"(b) FES: RMSD$_{\mathrm{C}\alpha}$ vs R$_g$", loc="left")
    ax_b.tick_params(labelsize=9)

    X_d, Y_d = np.meshgrid(cx_d1, cx_d2, indexing="ij")
    im2 = ax_c.contourf(X_d, Y_d, fes_d1_d2, levels=levels_fill, cmap=cmap, extend="max")
    ax_c.contour(X_d, Y_d, fes_d1_d2, levels=levels_line, colors="k", linewidths=0.4, alpha=0.5)
    cb2 = fig.colorbar(im2, ax=ax_c, pad=0.02)
    cb2.set_label(r"F (kJ mol$^{-1}$)")
    cb2.ax.tick_params(labelsize=9)

    ax_c.set_xlabel(r"d$_1$ ($\AA$) [atoms 31--92]")
    ax_c.set_ylabel(r"d$_2$ ($\AA$) [atoms 31--106]")
    ax_c.set_title("(c) FES: cross-strand contact distances", loc="left")
    ax_c.tick_params(labelsize=9)

    out_png = FIG_DIR / "figure_1uao.png"
    out_pdf = FIG_DIR / "figure_1uao.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
