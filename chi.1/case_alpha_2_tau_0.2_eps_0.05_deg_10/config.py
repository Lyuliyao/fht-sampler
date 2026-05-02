from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Case1Config:
    # Problem/model params
    d_true: int = 3
    d: int = 4
    L: int = 2
    deg: int = 10
    eps: float = 0.05
    alpha: float = 2
    tau_factor: float = 0.2
    pad_seed: int = 12345

    # Data loading
    run_glob: str = "./run*"
    colvar_name: str = "COLVAR"
    cv_col_start: int = 1
    divide_by_pi: bool = True
    stride: int = 1
    max_frames_per_run: Optional[int] = None
    max_sample_per_sketching: Optional[int] = None

    # Model I/O
    model_name: str = "c_save.npz"
    output_npz: str = "./c_save.npz"

    # Compute controls
    block_size: int = 50_000
    mbar_verbose: bool = True

    # Optional: export each tensor block to .bin
    export_bins: bool = True
    bins_out_dir: str = "./bins"

    # Optional: export PLUMED-compatible COLVAR with weights
    export_plumed_colvar: bool = True
    plumed_colvar_out: str = "./COLVAR.reweight"
    plumed_time_start: float = 0.0
    plumed_time_step: float = 0.025
    plumed_use_radians: bool = True
    plumed_cv_names: Optional[Tuple[str, ...]] = ("phi1", "psi1", "phi2", "psi2")

    # Gaussian basis options
    gaussian_sigma: float = 0.1
    gaussian_center_spacing: Optional[float] = None
    gaussian_pbc: bool = True
    gaussian_period: float = 2.0
    gaussian_domain: Tuple[float, float] = (-1.0, 1.0)
    gaussian_whiten: bool = True
    gaussian_whitening_regularization: float = 1e-10
    gaussian_normalize: bool = True
