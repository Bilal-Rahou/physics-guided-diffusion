import torch
import numpy as np
from skimage.filters import threshold_otsu
from .utils import (
    normalize_image,
    preprocess_image,
    normalize_image_with_threshold,
    gauss_legendre_integral,
    detect_lines_custom,
    calculate_angle,
    filter_longest_horizontal_vertical,
    create_vertical_line_mask,
    reflect_point_across_line
)


def green_depth(Fo_e, nmax=200):
    device, dtype = Fo_e.device, Fo_e.dtype
    Fo_e_safe = torch.where(Fo_e <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo_e)
    n = torch.arange(1, nmax + 1, dtype=dtype, device=device).view(nmax, 1, 1, 1)
    exp_term = torch.exp(- (n ** 2) / Fo_e_safe.unsqueeze(0))  # [nmax, B, H, W]
    s = exp_term.sum(dim=0)
    g1 = 1.0 + 2.0 * s
    return g1


def gruss_integrand(Fo, x_star, y_star, V, D, depth, r_s):
    device, dtype = Fo.device, Fo.dtype
    r_s_intern = r_s / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    Pe = (V * r_s_intern) / D  # Péclet number
    depth_star = depth / r_s_intern
    Fo_safe = torch.where(Fo <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo)
    A = 1.0 + 8.0 * Fo_safe
    exp_term = -2.0 * ((x_star**2) + (y_star + Pe * Fo_safe)**2) / A
    Fo_e = Fo_safe / (depth_star**2)
    g1 = green_depth(Fo_e)
    integrand = (g1 * torch.exp(exp_term)) / (A * torch.sqrt(Fo_safe))
    integrand = torch.where(Fo > 0, integrand, torch.tensor(0.0, dtype=dtype, device=device))
    return integrand


def analytical_solution_uncracked_sample(batch_images, alpha, P0, K, D, V, pixel_size, r_s, depth):
    """
    Compute analytical temperature fields (Krapez finite depth) for a batch of experimental images
    in a sample WITHOUT vertical cracks (only horizontal symmetry considered).
    
    Returns:
        theoretical_norm_mask_batch [B,H,W]
        mask_h_batch [B,H,W] or None
    """
    device = batch_images.device
    B, _, H, W = batch_images.shape

    # --- Step 1: Compute coordinate grids centered on each laser spot ---
    centers, X_list, Y_list = [], [], []
    for i in range(B):
        img = batch_images[i, 0]
        y_spot, x_spot = torch.unravel_index(torch.argmax(img), (H, W))
        centers.append((y_spot.item(), x_spot.item()))

        x_meters = (torch.arange(W, device=device) - x_spot) * pixel_size
        y_meters = (torch.arange(H, device=device) - y_spot) * pixel_size
        X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
        X_list.append(X)
        Y_list.append(Y)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    X_star, Y_star = X / r_s, Y / r_s

    # --- Step 2: Compute base theoretical field for the whole batch ---
    T_star = gauss_legendre_integral(
        lambda Fo, x, y: gruss_integrand(Fo, x, y, V, D, depth, r_s),
        X_star, Y_star,
        tau_min=0, tau_max=10000, n_points=1000
    )

    const = (2 * alpha * P0) / (
        K * torch.sqrt(torch.tensor(torch.pi, dtype=X.dtype, device=X.device)) *
        torch.tensor(torch.pi, dtype=X.dtype, device=X.device) * r_s
    )

    theoretical_field_batch = T_star * const  # [B,H,W]

    # --- Step 3: Process each image (horizontal symmetry only) ---
    theoretical_fields, theoretical_norms, theoretical_norm_masks = [], [], []
    mask_h_list = []

    for i in range(B):
        img = batch_images[i, 0]
        y_spot, x_spot = centers[i]
        theoretical_field = theoretical_field_batch[i].clone()

        # --- Detect horizontal line (only) ---
        image_for_lines = img.detach().cpu().numpy()
        image_uint8 = normalize_image(image_for_lines)
        clahe_blurred_img = preprocess_image(image_uint8)

        line_sets = [
            detect_lines_custom(image_uint8, 1.0, 1.2, 0),
            detect_lines_custom(image_uint8, 1.0, 1.2, 1),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
        ]
        filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
        all_filtered = [ln for sub in filtered_lines if sub is not None for ln in sub]
        final_filtered = filter_longest_horizontal_vertical(all_filtered, min_length=13)

        mask_h = None
        sym_h = None

        if final_filtered is not None:
            for line_data in final_filtered:
                line = line_data[0]
                angle = calculate_angle(*line)
                # Only horizontal lines considered
                if not (70 <= angle <= 110):
                    horiz_line = line
                    sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                    mask_h = torch.tensor(
                        create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                        device=device
                    )

        # --- Add mirrored contribution for the horizontal reflection ---
        if sym_h is not None:
            x_m = (torch.arange(W, device=device) - sym_h[0]) * pixel_size
            y_m = (torch.arange(H, device=device) - sym_h[1]) * pixel_size
            Xh, Yh = torch.meshgrid(y_m, x_m, indexing="ij")
            Xh_star, Yh_star = Xh / r_s, Yh / r_s
            T_sym_h = gauss_legendre_integral(
                lambda Fo, x, y: gruss_integrand(Fo, x, y, V, D, depth, r_s),
                Xh_star[None], Yh_star[None],
                tau_min=0, tau_max=10000, n_points=1000
            )[0]
            theoretical_field += T_sym_h * const

        # --- Normalize field ---
        theoretical_norm = (theoretical_field - theoretical_field.min()) / (
            theoretical_field.max() - theoretical_field.min() + 1e-8
        )

        # --- Apply Otsu threshold mask ---
        otsu_threshold = threshold_otsu(img.detach().cpu().numpy())
        otsu_mask = (img > otsu_threshold).float()
        theoretical_norm_mask = theoretical_norm * otsu_mask
        if mask_h is not None:
            theoretical_norm_mask *= mask_h

        # --- Collect results ---
        theoretical_fields.append(theoretical_field.unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0))
        theoretical_norm_masks.append(theoretical_norm_mask.unsqueeze(0))
        mask_h_list.append(mask_h.unsqueeze(0) if mask_h is not None else None)

    # --- Step 4: Combine batch tensors ---
    theoretical_norm_mask_batch = torch.cat(theoretical_norm_masks, dim=0)
    mask_h_batch = [m for m in mask_h_list if m is not None]

    return theoretical_norm_mask_batch, mask_h_batch


def analytical_solution_cracked_sample(batch_images, alpha, P0, K, D, V, pixel_size, r_s, depth):
    """
    Compute analytical temperature fields (Krapez finite depth) for a batch of experimental images.
    Returns:
        theoretical_norm_mask_batch [B,H,W]
        mask_h_batch [B,H,W] or None
    """
    device = batch_images.device
    B, _, H, W = batch_images.shape

    centers, X_list, Y_list = [], [], []
    for i in range(B):
        img = batch_images[i, 0]
        y_spot, x_spot = torch.unravel_index(torch.argmax(img), (H, W))
        centers.append((y_spot.item(), x_spot.item()))

        x_meters = (torch.arange(W, device=device) - x_spot) * pixel_size
        y_meters = (torch.arange(H, device=device) - y_spot) * pixel_size
        X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
        X_list.append(X)
        Y_list.append(Y)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    X_star, Y_star = X / r_s, Y / r_s

    T_star = gauss_legendre_integral(
        lambda Fo, x, y: gruss_integrand(Fo, x, y, V, D, depth, r_s),
        X_star, Y_star,
        tau_min=0, tau_max=10000, n_points=1000
    )

    const = (2 * alpha * P0) / (
        K * torch.sqrt(torch.tensor(torch.pi, dtype=X.dtype, device=X.device)) *
        torch.tensor(torch.pi, dtype=X.dtype, device=X.device) * r_s
    )

    theoretical_field_batch = T_star * const

    theoretical_fields, theoretical_norms, theoretical_norm_masks = [], [], []
    mask_h_list = []

    for i in range(B):
        img = batch_images[i, 0]
        y_spot, x_spot = centers[i]
        theoretical_field = theoretical_field_batch[i].clone()

        image_for_lines = img.detach().cpu().numpy()
        image_uint8 = normalize_image(image_for_lines)
        clahe_blurred_img = preprocess_image(image_uint8)

        line_sets = [
            detect_lines_custom(image_uint8, 1.0, 1.2, 0),
            detect_lines_custom(image_uint8, 1.0, 1.2, 1),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
        ]
        filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
        all_filtered = [ln for sub in filtered_lines if sub is not None for ln in sub]
        final_filtered = filter_longest_horizontal_vertical(all_filtered, min_length=13)

        mask_v = mask_h = None
        sym_v = sym_h = None

        if final_filtered is not None:
            for line_data in final_filtered:
                line = line_data[0]
                angle = calculate_angle(*line)
                if 70 <= angle <= 110:
                    vert_line = line
                    sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                    mask_v = torch.tensor(
                        create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                        device=device
                    )
                else:
                    horiz_line = line
                    sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                    mask_h = torch.tensor(
                        create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                        device=device
                    )

        if sym_v is not None:
            x_m = (torch.arange(W, device=device) - sym_v[0]) * pixel_size
            y_m = (torch.arange(H, device=device) - sym_v[1]) * pixel_size
            Xv, Yv = torch.meshgrid(y_m, x_m, indexing="ij")
            Xv_star, Yv_star = Xv / r_s, Yv / r_s
            T_sym_v = gauss_legendre_integral(
                lambda Fo, x, y: gruss_integrand(Fo, x, y, V, D, depth, r_s),
                Xv_star[None], Yv_star[None],
                tau_min=0, tau_max=10000, n_points=1000
            )[0]
            theoretical_field += T_sym_v * const

        if sym_h is not None:
            x_m = (torch.arange(W, device=device) - sym_h[0]) * pixel_size
            y_m = (torch.arange(H, device=device) - sym_h[1]) * pixel_size
            Xh, Yh = torch.meshgrid(y_m, x_m, indexing="ij")
            Xh_star, Yh_star = Xh / r_s, Yh / r_s
            T_sym_h = gauss_legendre_integral(
                lambda Fo, x, y: gruss_integrand(Fo, x, y, V, D, depth, r_s),
                Xh_star[None], Yh_star[None],
                tau_min=0, tau_max=10000, n_points=1000
            )[0]
            theoretical_field += T_sym_h * const

        theoretical_norm = (theoretical_field - theoretical_field.min()) / (
            theoretical_field.max() - theoretical_field.min() + 1e-8
        )

        otsu_threshold = threshold_otsu(img.detach().cpu().numpy())
        otsu_mask = (img > otsu_threshold).float()
        theoretical_norm_mask = theoretical_norm * otsu_mask
        if mask_v is not None:
            theoretical_norm_mask *= mask_v
        if mask_h is not None:
            theoretical_norm_mask *= mask_h

        theoretical_fields.append(theoretical_field.unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0))
        theoretical_norm_masks.append(theoretical_norm_mask.unsqueeze(0))
        mask_h_list.append(mask_h.unsqueeze(0) if mask_h is not None else None)

    theoretical_norm_mask_batch = torch.cat(theoretical_norm_masks, dim=0)
    mask_h_batch = [m for m in mask_h_list if m is not None]
    return theoretical_norm_mask_batch, mask_h_batch

    

def compute_residual(batch_images, alpha, P0, K, D, V, pixel_size, r_s, depth, mode="uncracked"):
    """
    Compute residuals between experimental and analytical finite-depth temperature fields.
    Returns only the residuals (batch of shape [B, 1, H, W]).
    """
    if mode == "cracked":
        theoretical_norm_mask_batch, mask_h_batch = analytical_solution_cracked_sample(
            batch_images,
            alpha,
            P0,
            K,
            D,
            V,
            pixel_size,
            r_s,
            depth
        )

    elif mode == "uncracked":
        theoretical_norm_mask_batch, mask_h_batch = analytical_solution_uncracked_sample(
            batch_images,
            alpha,
            P0,
            K,
            D,
            V,
            pixel_size,
            r_s,
            depth
        )

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Please choose either 'cracked' or 'crack-free'."
        )


    residuals = []
    B = batch_images.shape[0]

    for i in range(B):
        image = batch_images[i, 0]
        otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
        otsu_threshold = torch.tensor(otsu_threshold, dtype=image.dtype, device=image.device)
        experimental_norm = normalize_image_with_threshold(image, otsu_threshold)
        residual = experimental_norm - theoretical_norm_mask_batch[i]
        if i < len(mask_h_batch) and mask_h_batch[i] is not None:
            residual *= mask_h_batch[i].squeeze()
        residuals.append(residual.unsqueeze(0).unsqueeze(0))

    residuals = torch.cat(residuals, dim=0)
    return residuals
