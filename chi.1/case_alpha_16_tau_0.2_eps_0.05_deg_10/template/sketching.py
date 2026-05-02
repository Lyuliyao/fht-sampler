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


def parse_colvar_fields(colvar_path: Path) -> List[str]:
    with open(colvar_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                parts = line.strip().split()
                if len(parts) <= 2:
                    break
                return parts[2:]
            if not line.startswith("#"):
                break
    raise ValueError(f"Cannot find '#! FIELDS' header in {colvar_path}")


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


def load_sampling_weights_from_colvar(
    colvar_path: Path,
    *,
    run_index: int,
    stride: int,
    max_frames: Optional[int],
    mdp_path: Path,
    default_temp_k: float,
    logger: logging.Logger,
) -> np.ndarray:
    data = load_or_build_colvar_array(colvar_path, logger=logger)

    if stride > 1:
        data = data[::stride]
    if max_frames is not None and data.shape[0] > max_frames:
        data = data[:max_frames]

    n = data.shape[0]
    if n == 0:
        raise ValueError(f"No frames available for weights in {colvar_path}")

    if run_index == 1:
        logger.info("Run index is 1; using equal weights.")
        return np.full(n, 1.0 / n, dtype=np.float64)

    fields = parse_colvar_fields(colvar_path)
    field_to_idx = {name: i for i, name in enumerate(fields)}

    for col in ("weight", "weights", "w"):
        if col in field_to_idx:
            raw = data[:, field_to_idx[col]].astype(np.float64, copy=False)
            raw = np.where(np.isfinite(raw), raw, 0.0)
            raw = np.clip(raw, 0.0, None)
            s = raw.sum()
            if s <= 0.0:
                raise ValueError(f"Weight column '{col}' in {colvar_path} is non-positive after filtering.")
            logger.info(f"Loaded weights from COLVAR column '{col}'.")
            return raw / s

    bias_col = None
    for col in ("metad.rbias", "metad.bias", "rbias", "bias"):
        if col in field_to_idx:
            bias_col = col
            break
    if bias_col is None:
        raise ValueError(
            f"No weight-like columns found in {colvar_path}. "
            f"Need one of weight/weights/w/metad.rbias/metad.bias/rbias/bias."
        )

    temp_k = infer_temperature_from_mdp(mdp_path, default_temp_k, logger)
    kbt = 8.31446261815324e-3 * temp_k  # kJ/mol
    bias = data[:, field_to_idx[bias_col]].astype(np.float64, copy=False)
    logw = bias / kbt
    finite = np.isfinite(logw)
    if not np.any(finite):
        raise ValueError(f"Computed log-weights from '{bias_col}' are all invalid in {colvar_path}.")
    logw = np.where(finite, logw, -np.inf)
    logw -= np.max(logw[finite])
    w = np.exp(logw)
    s = np.sum(w)
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError(f"Computed weights from '{bias_col}' are invalid in {colvar_path}.")
    logger.info(f"Computed weights from COLVAR column '{bias_col}' via exp(bias/kBT).")
    return w / s



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
# Bias evaluators u_k(s)
# ---------------------------
def u_zero(s_batch: np.ndarray) -> np.ndarray:
    return np.zeros(s_batch.shape[0], dtype=np.float64)


def make_u_evaluator_from_npz(
    npz_path: Path,
    *,
    d_pad: int,
    d_true: int,
    L: int,
    deg: int,
    eps: float,
    sigma: float = 0.2,
    center_spacing: float = 0.25,
    pbc: bool = True,
    period: float = 2.0,
    domain: Tuple[float, float] = (-1.0, 1.0),
    whiten: bool = True,
    whitening_regularization: float = 1e-10,
    tau_factor: float = 0.5,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns u(s) = log(rho(s)) with safe cutoff:
      if rho <= eps or invalid -> log(eps)
    """
    obj = np.load(npz_path, allow_pickle=True)
    c = obj["c"].item()
    model_sigma = float(np.array(obj["gaussian_sigma"]).reshape(())) if "gaussian_sigma" in obj.files else float(sigma)
    model_center_spacing = (
        float(np.array(obj["gaussian_center_spacing"]).reshape(()))
        if "gaussian_center_spacing" in obj.files
        else float(center_spacing)
    )
    htn = FunctionalHierarchicalTensorGaussian(
        d=d_pad,
        L=L,
        deg=deg,
        c=c,
        ghost_pt=[],
        sigma=model_sigma,
        center_spacing=model_center_spacing,
        domain=domain,
        pbc=pbc,
        whiten=whiten,
        whitening_regularization=whitening_regularization,
        normalize=True,
    )
    mask = list(range(d_true + 1, d_pad + 1))  # 117..128 (1-indexed leaf ids)
    tau = float(tau_factor) * float(eps)
    
    def soft_plus(x):
        return np.log(1+np.exp(x))
    
    def u_of_s(s_batch: np.ndarray) -> np.ndarray:
        s_batch = np.asarray(s_batch, dtype=np.float64)
        if s_batch.shape[1] == d_true:
            s_pad = np.zeros((s_batch.shape[0], d_pad), dtype=np.float64)
            s_pad[:, :d_true] = s_batch
        elif s_batch.shape[1] == d_pad:
            s_pad = s_batch
        else:
            raise ValueError(f"s_batch has wrong dim {s_batch.shape[1]}, expect {d_true} or {d_pad}")

        # marginalize out the padded dims
        rho_raw = htn.evaluate_marginal(s_pad, mask=mask).astype(np.float64, copy=False)

        rho_raw = np.where(np.isfinite(rho_raw), rho_raw, 0.0)
        tau = float(tau_factor) * float(eps)
        t = rho_raw / tau
        rho_eff = eps + tau * soft_plus(t)
        return np.log(rho_eff)

    return u_of_s


def compute_unbiased_weights(
    u_kn: np.ndarray,
    N_k: np.ndarray,
    f_k: np.ndarray,
) -> np.ndarray:
    """
    weights for target state with u_target=0:
      w_n ∝ 1 / Σ_k N_k exp(f_k - u_k(x_n))
    """
    logNk = np.log(N_k.astype(np.float64))

    # log denom_n = logsumexp_k [ logNk + f_k - u_kn(k,n) ]
    log_denom = logsumexp((logNk + f_k)[:, None] - u_kn, axis=0)

    logw = -log_denom
    logw -= logw.max()   # stabilize
    w = np.exp(logw)
    w /= w.sum()
    return w


def effective_sample_size(w: np.ndarray) -> float:
    # ESS = (sum w)^2 / sum(w^2), but sum w = 1 here
    return 1.0 / np.sum(w**2)


def summarize_weights(
    w: np.ndarray,
    run_id: np.ndarray,
    N_k: np.ndarray,
    logger: logging.Logger,
) -> None:
    Ntot = w.size
    ess = effective_sample_size(w)
    logger.info(f"Global ESS: {ess:.3f} out of Ntot={Ntot}")

    w_by_run = np.array([w[run_id == k].sum() for k in range(len(N_k))], dtype=np.float64)
    ess_by_run = np.array([
        (w[run_id == k].sum()**2) / np.sum(w[run_id == k]**2) if np.any(run_id == k) else 0.0
        for k in range(len(N_k))
    ], dtype=np.float64)

    logger.info(f"N_k: {N_k}  sum={N_k.sum()}")
    logger.info(f"Weight mass by run: {w_by_run}  sum={w_by_run.sum():.6f}")
    logger.info(f"ESS by run: {ess_by_run}  sum ESS_by_run={ess_by_run.sum():.3f}")

    w_sorted = np.sort(w)
    top10 = w_sorted[-10:]
    logger.info(f"Top-10 weights: {top10}")
    logger.info(f"Top weight fraction: {w_sorted[-1]:.6g}")

    top100_mass = w_sorted[-100:].sum() if w_sorted.size >= 100 else w_sorted.sum()
    logger.info(f"Top-100 weight mass: {top100_mass:.6g}")


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
