import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

CASE_DIR = Path(__file__).resolve().parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from config import Case1Config
from metabias.fht.gaussian import FunctionalHierarchicalTensorGaussian
from metabias.fht.sketch_gaussian import hier_tensor_sketch_gaussian


# ---------------------------
# Utilities
# ---------------------------
def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mbar_sketch")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def parse_run_index(path: Path) -> int:
    """
    Extract trailing integer from run directory name, e.g. run1 -> 1, run02 -> 2.
    """
    m = re.search(r"(\d+)$", path.name)
    if not m:
        raise ValueError(f"Cannot parse run index from directory name: {path}")
    return int(m.group(1))


def list_run_dirs(run_glob: str) -> List[Path]:
    run_dirs = [p for p in map(Path, sorted(Path().glob(run_glob))) if p.is_dir()]
    run_dirs = sorted(run_dirs, key=parse_run_index)
    return run_dirs


def load_or_build_colvar_array(
    colvar_path: Path,
    *,
    logger: logging.Logger,
) -> np.ndarray:
    cache_path = colvar_path.with_suffix(".npy")

    if cache_path.exists():
        try:
            if cache_path.stat().st_mtime >= colvar_path.stat().st_mtime:
                data = np.load(cache_path, allow_pickle=False)
                if data.ndim == 1:
                    data = data[None, :]
                if data.ndim != 2:
                    raise ValueError(f"Cached array in {cache_path} has invalid ndim={data.ndim}")
                logger.info(f"Loaded cached COLVAR npy: {cache_path} shape {data.shape}")
                return data
        except Exception as exc:
            logger.warning(f"Failed reading cache {cache_path}: {exc}; rebuilding from text.")

    rows: List[List[float]] = []
    expected_cols = 0
    dropped_bad_cols = 0
    dropped_bad_parse = 0

    with open(colvar_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if expected_cols == 0:
                expected_cols = len(parts)

            if len(parts) != expected_cols:
                dropped_bad_cols += 1
                continue

            try:
                row = [float(x) for x in parts]
            except ValueError:
                dropped_bad_parse += 1
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"No valid numeric rows in {colvar_path}")

    data = np.asarray(rows, dtype=np.float64)

    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.save(fh, data)
        tmp_path.replace(cache_path)
    except OSError as exc:
        logger.warning(f"Failed to write cache {cache_path}: {exc}")
    else:
        logger.info(f"Saved cleaned COLVAR cache: {cache_path} shape {data.shape}")

    dropped_total = dropped_bad_cols + dropped_bad_parse
    if dropped_total > 0:
        logger.warning(
            f"{colvar_path}: dropped {dropped_total} malformed rows "
            f"(bad columns: {dropped_bad_cols}, parse errors: {dropped_bad_parse}, "
            f"expected columns: {expected_cols})."
        )

    return data


def load_cv_from_colvar(
    colvar_path: Path,
    *,
    d_true: int,
    cv_col_start: int,
    stride: int,
    max_frames: Optional[int],
    divide_by_pi: bool,
    logger: logging.Logger,
) -> np.ndarray:
    data = load_or_build_colvar_array(colvar_path, logger=logger)

    if cv_col_start + d_true > data.shape[1]:
        raise ValueError(
            f"{colvar_path} has {data.shape[1]} columns, but need at least {cv_col_start + d_true}."
        )

    cv = data[:, cv_col_start:cv_col_start + d_true].astype(np.float64, copy=False)

    if divide_by_pi:
        cv = cv / np.pi

    if stride > 1:
        cv = cv[::stride]

    if max_frames is not None and cv.shape[0] > max_frames:
        cv = cv[:max_frames]

    return cv


def colvar_has_numeric_rows(colvar_path: Path) -> bool:
    """
    Return True only if COLVAR has at least one non-comment data row.
    """
    with open(colvar_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return True
    return False



def infer_temperature_from_mdp(mdp_path: Path, default_temp: float, logger: logging.Logger) -> float:
    if mdp_path.exists():
        with open(mdp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split(";", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                key, rhs = line.split("=", 1)
                if key.strip().lower() == "ref-t":
                    vals: List[float] = []
                    for tok in rhs.split():
                        try:
                            vals.append(float(tok))
                        except ValueError:
                            continue
                    if vals:
                        temp = vals[0]
                        logger.info(f"Using temperature from {mdp_path.name}: T={temp:g} K")
                        return temp
    logger.info(f"Temperature not found in {mdp_path}; fallback T={default_temp:g} K")
    return float(default_temp)




def build_rank_structures(L: int, deg: int) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], List[int]]]:
    """
    Build r and s dictionaries exactly like your original logic, but packaged.
    """
    r: Dict[Tuple[int, int], List[int]] = {}
    s: Dict[Tuple[int, int], List[int]] = {}

    # same as: 10 + np.arange(L,0,-1)
    r_level = 10 + np.arange(L, 0, -1)
    s_level = 10 + np.arange(L, 0, -1)

    for l in reversed(range(0, L + 1)):
        for k in range(1, 2**l + 1):
            if l == L:
                r[(k, l)] = [2 * deg + 1, int(r_level[L - 1])]
                s[(k, l)] = [2 * deg + 1, int(r_level[L - 1] + s_level[L - 1])]
            elif l == 0:
                r[(k, l)] = [int(r_level[0]), int(r_level[0])]
                s[(k, l)] = [int(r_level[0] + s_level[0]), int(r_level[0] + s_level[0])]
            else:
                r[(k, l)] = [int(r_level[l - 1]), int(r_level[l]), int(r_level[l])]
                s[(k, l)] = [
                    int(r_level[l - 1] + s_level[l - 1]),
                    int(r_level[l] + s_level[l]),
                    int(r_level[l] + s_level[l]),
                ]
    return r, s


# ---------------------------
# Saving
# ---------------------------
def save_c_npz(
    c: dict,
    out_path: Path,
    *,
    d_true: int,
    d_pad: int,
    L: int,
    deg: int,
    gaussian_sigma: float,
    gaussian_center_spacing: float,
    logger: logging.Logger,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        c=c,
        d_true=d_true,
        d_pad=d_pad,
        L=L,
        deg=deg,
        gaussian_sigma=float(gaussian_sigma),
        gaussian_center_spacing=float(gaussian_center_spacing),
    )
    logger.info(f"Saved model npz -> {out_path}")



def save_binary_block(key: Tuple[int, int], data: np.ndarray, filename: Path, *, logger: logging.Logger) -> None:
    """
    Format (same spirit as your original):
      uint32: key_str_len
      bytes : key_str (e.g. "1_2")
      uint32: ndim
      uint32[ndim]: shape
      float64[...] : flattened data
    """
    filename.parent.mkdir(parents=True, exist_ok=True)

    key_str = "_".join(map(str, key))
    key_bytes = key_str.encode("utf-8")

    arr = np.asarray(data, dtype=np.float64, order="C")
    with open(filename, "wb") as f:
        f.write(np.uint32(len(key_bytes)).tobytes())
        f.write(key_bytes)
        f.write(np.uint32(arr.ndim).tobytes())
        f.write(np.array(arr.shape, dtype=np.uint32).tobytes())
        f.write(arr.reshape(-1).tobytes())

    logger.info(f"Saved bin block key={key} shape={arr.shape} -> {filename}")


def export_c_bins(c: dict, out_dir: Path, *, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in c.items():
        fn = out_dir / (("_".join(map(str, key))) + ".bin")
        save_binary_block(key, arr, fn, logger=logger)


def save_basis_params(out_dir: Path, *, sigma: float, center_spacing: float, logger: logging.Logger) -> None:
    """
    Save Gaussian basis parameters next to tensor .bin files so MetaTensor.cpp can load them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    param_path = out_dir / "basis_params.dat"
    with open(param_path, "w", encoding="ascii") as f:
        f.write("# Auto-generated by sketching.py\n")
        f.write(f"sigma={float(sigma):.17g}\n")
        f.write(f"center_spacing={float(center_spacing):.17g}\n")
    logger.info(f"Saved basis params -> {param_path}")


# ---------------------------
# Main pipeline
# ---------------------------
def main(cfg: Case1Config) -> FunctionalHierarchicalTensorGaussian:
    logger = setup_logger()

    run_dirs = list_run_dirs(cfg.run_glob)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched: {cfg.run_glob}")
    logger.info(f"Found {len(run_dirs)} run dirs:")
    for rd in run_dirs:
        logger.info(f"  {rd}")
    logger.info("Using cumulative sampling from all matched run directories.")
    d_pad = 2 ** cfg.L
    if cfg.d != d_pad:
        raise ValueError(f"Config mismatch: cfg.d={cfg.d} but 2**L={d_pad}. Keep cfg.d=2**cfg.L.")

    rng = np.random.default_rng(cfg.pad_seed)

    cv_list: List[np.ndarray] = []
    for rd in run_dirs:
        colvar_path = rd / cfg.colvar_name
        run_idx = parse_run_index(rd)
        if not colvar_path.exists():
            logger.warning(f"Run {run_idx:02d}: missing {colvar_path.name}, skipped.")
            continue
        if not colvar_has_numeric_rows(colvar_path):
            logger.warning(f"Run {run_idx:02d}: empty {colvar_path.name}, skipped.")
            continue

        try:
            cv_true = load_cv_from_colvar(
                colvar_path,
                d_true=cfg.d_true,
                cv_col_start=cfg.cv_col_start,
                stride=cfg.stride,
                max_frames=cfg.max_frames_per_run,
                divide_by_pi=cfg.divide_by_pi,
                logger=logger,
            )
        except (ValueError, OSError) as exc:
            logger.warning(f"Run {run_idx:02d}: failed to load {colvar_path.name} ({exc}), skipped.")
            continue

        if cv_true.ndim != 2 or cv_true.shape[1] != cfg.d_true:
            logger.warning(
                f"Run {run_idx:02d}: invalid CV shape {cv_true.shape} in {colvar_path.name}, "
                f"expected (*, {cfg.d_true}), skipped."
            )
            continue

        n = cv_true.shape[0]
        if n == 0:
            logger.warning(f"Run {run_idx:02d}: no frames in {colvar_path.name}, skipped.")
            continue

        cv = np.empty((n, d_pad), dtype=np.float64)
        cv[:, :cfg.d_true] = cv_true
        cv[:, cfg.d_true:] = rng.uniform(-1.0, 1.0, size=(n, d_pad - cfg.d_true))
        cv_list.append(cv)
        logger.info(f"Run {run_idx:02d}: cv shape {cv.shape}")

    if not cv_list:
        raise RuntimeError(
            f"No usable {cfg.colvar_name} files found under run dirs matched by {cfg.run_glob}."
        )
    logger.info(f"Using {len(cv_list)} non-empty {cfg.colvar_name} files for sketching.")

    s_all = np.vstack(cv_list)
    logger.info(f"Merged cumulative cv shape: {s_all.shape}")

    if cfg.max_sample_per_sketching is not None and s_all.shape[0] > cfg.max_sample_per_sketching:
        pick = rng.choice(s_all.shape[0], size=cfg.max_sample_per_sketching, replace=False)
        s_all = s_all[pick]
        logger.info(f"Subsampled to max_sample_per_sketching={cfg.max_sample_per_sketching}, shape={s_all.shape}")

    w0 = np.ones(s_all.shape[0])/s_all.shape[0]

    gaussian_sigma = float(getattr(cfg, "gaussian_sigma", 0.2))
    gaussian_center_spacing = getattr(cfg, "gaussian_center_spacing", None)
    if gaussian_center_spacing is None:
        gaussian_center_spacing = 2.0 / (2 * cfg.deg + 1)
    gaussian_center_spacing = float(gaussian_center_spacing)

    # Build r,s ranks
    r, s = build_rank_structures(cfg.L, cfg.deg)

    # Hierarchical tensor sketch
    c = hier_tensor_sketch_gaussian(
        s_all,
        cfg.L,
        cfg.d,
        cfg.deg,
        r=r,
        s=s,
        w=w0,
        sigma=gaussian_sigma,
        center_spacing=gaussian_center_spacing,
        pbc=getattr(cfg, "gaussian_pbc", True),
        period=getattr(cfg, "gaussian_period", 2.0),
        domain=getattr(cfg, "gaussian_domain", (-1.0, 1.0)),
        whiten=getattr(cfg, "gaussian_whiten", True),
        whitening_regularization=getattr(cfg, "gaussian_whitening_regularization", 1e-10),
        debug=False,
    )

    # Save outputs
    save_c_npz(
        c,
        Path(cfg.output_npz),
        d_true=cfg.d_true,
        d_pad=cfg.d,
        L=cfg.L,
        deg=cfg.deg,
        gaussian_sigma=gaussian_sigma,
        gaussian_center_spacing=gaussian_center_spacing,
        logger=logger
    )

    if cfg.export_bins:
        bins_dir = Path(cfg.bins_out_dir)
        export_c_bins(c, bins_dir, logger=logger)
        save_basis_params(
            bins_dir,
            sigma=gaussian_sigma,
            center_spacing=gaussian_center_spacing,
            logger=logger,
        )

    # Construct obtained model object
    htn_obtained = FunctionalHierarchicalTensorGaussian(
        d=cfg.d,
        L=cfg.L,
        deg=cfg.deg,
        c=c,
        ghost_pt=[],
        sigma=gaussian_sigma,
        center_spacing=gaussian_center_spacing,
        domain=getattr(cfg, "gaussian_domain", (-1.0, 1.0)),
        pbc=getattr(cfg, "gaussian_pbc", True),
        whiten=getattr(cfg, "gaussian_whiten", True),
        whitening_regularization=getattr(cfg, "gaussian_whitening_regularization", 1e-10),
        normalize=getattr(cfg, "gaussian_normalize", True),
    )
    logger.info("Constructed htn_obtained successfully.")

    return htn_obtained


if __name__ == "__main__":
    # Minimal: run with defaults
    cfg = replace(
        Case1Config(),
        run_glob="../run*",
        block_size=500_000,
        max_sample_per_sketching=20000000,
    )
    _ = main(cfg)
