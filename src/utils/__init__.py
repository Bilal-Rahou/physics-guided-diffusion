"""
utils package for Physics-Informed DDPM
---------------------------------------

This package contains helper functions for:
- Image preprocessing and normalization
- Line detection and symmetry analysis
- Analytical thermal field computation (Salazar / Gruss models)
- Residual comparison between analytical and experimental fields

Author: Bilal Rahou
Date: 11/2025
"""

# --- Image processing utilities ---
from .utils import (
    normalize_image,
    preprocess_image,
    normalize_image_with_threshold,
    gauss_legendre_integral,
    detect_lines_custom,
    calculate_angle,
    filter_longest_horizontal_vertical,
    create_vertical_line_mask,
    reflect_point_across_line,
)

# --- Analytical model and residual computation ---
from .residual import (
    analytical_solution_uncracked_sample,
    analytical_solution_cracked_sample,
    compute_residual,
)

from .physics_metrics import residual_mae

from .attend import Attend


# --- Control what gets imported with `from utils import *` ---
__all__ = [
    # utils.py
    "normalize_image",
    "preprocess_image",
    "normalize_image_with_threshold",
    "gauss_legendre_integral",
    "detect_lines_custom",
    "calculate_angle",
    "filter_longest_horizontal_vertical",
    "create_vertical_line_mask",
    "reflect_point_across_line",

    # residual.py
    "analytical_solution_uncracked_sample",
    "analytical_solution_cracked_sample",
    "compute_residual",

    # physics_metrics.py
    "residual_mae",

    # attend.py
    "Attend",
]
