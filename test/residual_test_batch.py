import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import roots_legendre
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.feature import canny
import torch
from skimage import exposure
import scienceplots

# Constants
P0 = 2  # Laser power (in watts)
K = 237  # Thermal conductivity (W/mK)
D = 1e-4  # Thermal diffusivity (m^2/s)
V = -0.0005  # Laser speed (m/s)
alpha = 0.38  # Efficiency
pixel_size = 0.0003  # Pixel size (in meters)
r_s = 0.00075  # Gaussian radius (in meters)
depth = 0.002

# Calculated constants
Pe = (V * r_s) / D  # Péclet number
epsilon = K / np.sqrt(D)

plt.style.use("science")

# Loading and preprocessing functions (same as before)
def load_images_from_folder(folder_path):
    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                path = os.path.join(root, file)
                image = io.imread(path)
                if image.ndim == 3:
                    image = rgb2gray(image)
                images.append((path, image))
    print(f"Total images found: {len(images)}")
    return images

import math
import numpy as np
import cv2
import torch
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from scipy.special import roots_legendre

# === Utility Functions ===

def pixel_to_meter(x, y, pixel_size_m=0.0003):
    return x * pixel_size_m, y * pixel_size_m

def normalize_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8

def preprocess_image(img_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_uint8)
    return cv2.GaussianBlur(clahe_img, (3, 3), 0)

def normalize_image_with_threshold(image, threshold):
    image_clipped = torch.clamp(image, min=threshold, max=image.max())
    norm = (image_clipped - threshold) / (image_clipped.max() - threshold + 1e-8)
    return norm

def detect_lines_custom(img, scale, sigma_scale, refine):
    lsd = cv2.createLineSegmentDetector(
        refine=refine, scale=scale, sigma_scale=sigma_scale,
        quant=2.0, ang_th=22.5, log_eps=0, density_th=0.7, n_bins=1024
    )
    lines, _, _, _ = lsd.detect(img)
    return lines

def calculate_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    return np.degrees(angle_rad) % 180

def filter_longest_horizontal_vertical(lines, min_length=15):
    longest_horizontal = None
    longest_vertical = None
    max_h_length = max_v_length = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(float, line[0])
            angle = calculate_angle(x1, y1, x2, y2)
            length = np.hypot(x2 - x1, y2 - y1)

            if length < min_length:
                continue

            if -10 <= angle <= 10 or 170 <= angle <= 180:
                if length > max_h_length:
                    max_h_length = length
                    longest_horizontal = [(x1, min(y1, y2), x2, min(y1, y2))]
            elif 70 <= angle <= 110:
                if length > max_v_length:
                    max_v_length = length
                    longest_vertical = [(x1, y1, x2, y2)]

    result = []
    if longest_horizontal:
        result.append(np.array([longest_horizontal[0]]))
    if longest_vertical:
        result.append(np.array([longest_vertical[0]]))
    return np.array(result) if result else None

def create_vertical_line_mask(image_shape, x1, y1, x2, y2, x_spot, y_spot):
    dx, dy = x2 - x1, y2 - y1
    A, B, C = dy, -dx, dx * y1 - dy * x1
    mask = np.zeros(image_shape, dtype=np.uint8)

    side_of_line = A * x_spot + B * y_spot + C

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            side = A * x + B * y + C
            if (side_of_line > 0 and side > 0) or (side_of_line < 0 and side < 0):
                mask[y, x] = 1
    return mask

def reflect_point_across_line(x1, y1, x2, y2, x0, y0):
    dx, dy = x2 - x1, y2 - y1
    norm_sq = dx * dx + dy * dy
    if norm_sq == 0:
        return None

    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    apx, apy = x0 - x1, y0 - y1
    dot = apx * dx + apy * dy
    proj_x = dot * dx
    proj_y = dot * dy

    proj_point_x = x1 + proj_x
    proj_point_y = y1 + proj_y

    abx, aby = x2 - x1, y2 - y1
    apx_proj, apy_proj = proj_point_x - x1, proj_point_y - y1
    dot_proj = abx * apx_proj + aby * apy_proj

    if not (0 <= dot_proj <= abx * abx + aby * aby):
        return None

    x_reflected = x1 + 2 * proj_x - apx
    y_reflected = y1 + 2 * proj_y - apy
    return int(round(x_reflected)), int(round(y_reflected))

def residual_mirrored(batch_images):
        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))

        device = batch_images.device
        batch_size, _, height, width = batch_images.shape

        residuals = []
        theoretical_norms = []
        theoretical_norms_mask = []
        experimental_norms = []
        X_list, Y_list, centers = [], [], []

        for i in range(batch_size):
            img = batch_images[i, 0]
            y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
            centers.append((y_spot.item(), x_spot.item()))

            x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
            y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
            X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
            X_list.append(X)
            Y_list.append(Y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)

        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        const = (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))
        theoretical_field_batch = T_star * const

        for i in range(batch_size):
            image_for_lines = batch_images[i, 0].detach().cpu().numpy()
            image_uint8 = normalize_image(image_for_lines)
            clahe_blurred_img = preprocess_image(image_uint8)

            line_sets = [
                detect_lines_custom(image_uint8, 1.0, 1.2, 0),
                detect_lines_custom(image_uint8, 1.0, 1.2, 1),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
            ]
            filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
            all_filtered_lines = [line for sublist in filtered_lines if sublist is not None for line in sublist]
            final_filtered_lines = filter_longest_horizontal_vertical(all_filtered_lines, min_length=13)

            y_spot, x_spot = centers[i]
            sym_h = sym_v = None
            vert_line = horiz_line = None
            mask_v = mask_h = None

            if final_filtered_lines is not None and len(final_filtered_lines) == 2:
                line1, line2 = final_filtered_lines
                horiz_line = line1[0]
                vert_line = line2[0]
                sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                mask_v = torch.tensor(mask_v, device=device)  
                mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                mask_h = torch.tensor(mask_h, device=device)  
            elif final_filtered_lines is not None and len(final_filtered_lines) == 1:
                line = final_filtered_lines[0]
                angle = calculate_angle(*line[0])
                if 70 <= angle <= 110:
                    vert_line = line[0]
                    sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                    mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                    mask_v = torch.tensor(mask_v, device=device)  
                else:
                    horiz_line = line[0]
                    sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                    mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                    mask_h = torch.tensor(mask_h, device=device)  

            theoretical_field = theoretical_field_batch[i].clone()

            if sym_v is not None:
                x_meters = (torch.arange(width, device=device) - sym_v[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_v[1]) * pixel_size
                Xv, Yv = torch.meshgrid(y_meters, x_meters, indexing="ij")
                T_sym_v = gauss_legendre_integral(salazar_integrand, Xv[None], Yv[None], tau_min=-500, tau_max=0)
                theoretical_field += T_sym_v[0] * const
            if sym_h is not None:
                x_meters = (torch.arange(width, device=device) - sym_h[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_h[1]) * pixel_size
                Xh, Yh = torch.meshgrid(y_meters, x_meters, indexing="ij")
                T_sym_h = gauss_legendre_integral(salazar_integrand, Xh[None], Yh[None], tau_min=-500, tau_max=0)
                theoretical_field += T_sym_h[0] * const

            image = batch_images[i, 0]
            otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
            otsu_mask = (image > otsu_threshold).float()
            norm_threshold = torch.tensor(np.percentile(image[image > otsu_threshold].detach().cpu().numpy(), 25), 
                     dtype=image.dtype, device=image.device)
            experimental_norm = normalize_image_with_threshold(image, norm_threshold)

            theoretical_norm = (theoretical_field - theoretical_field.min()) / (theoretical_field.max() - theoretical_field.min() + 1e-8)
            theoretical_norm_mask = theoretical_norm * otsu_mask
            if mask_h is not None:
                theoretical_norm_mask *= mask_h

            residual =  experimental_norm - theoretical_norm_mask
            if mask_h is not None:
                residual *= mask_h
            residuals.append(residual.unsqueeze(0).unsqueeze(0))
            theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]        residuals = torch.cat(residuals, dim=0)              # [B, 1, H, W]
        theoretical_norms = torch.cat(theoretical_norms, dim=0)
        theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
        experimental_norms = torch.cat(experimental_norms, dim=0)
        residuals = torch.cat(residuals, dim=0)
        return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms
def plot_infinite_line(x1, y1, x2, y2, img_shape, color="cyan", linewidth=1.5):
    """
    Draw an infinite line across the image (clipped to image borders).
    The line color matches the associated symmetric point.
    """
    h, w = img_shape

    if x2 == x1:  # vertical line
        plt.plot([x1, x1], [0, h], color=color, linestyle="--", linewidth=linewidth)
    else:
        m = (y2 - y1) / (x2 - x1)  # slope
        b = y1 - m * x1           # intercept

        # Compute intersections with the image borders
        points = []

        # Left border (x=0)
        y = b
        if 0 <= y <= h:
            points.append((0, y))

        # Right border (x=w)
        y = m * w + b
        if 0 <= y <= h:
            points.append((w, y))

        # Top border (y=0)
        x = -b / m
        if 0 <= x <= w:
            points.append((x, 0))

        # Bottom border (y=h)
        x = (h - b) / m
        if 0 <= x <= w:
            points.append((x, h))

        # If we found 2 intersections, draw the clipped line
        if len(points) >= 2:
            (xA, yA), (xB, yB) = points[:2]
            plt.plot([xA, xB], [yA, yB], color=color, linestyle="--", linewidth=linewidth)

def analytical_solution(batch_images):
    """
    Compute analytical temperature fields (Krapez finite depth) for a batch of experimental images.
    Returns:
        theoretical_field_batch [B,H,W]
        theoretical_norm_batch [B,H,W]
        theoretical_norm_mask_batch [B,H,W]
        mask_v_batch [B,H,W] or None
        mask_h_batch [B,H,W] or None
    """
    device = batch_images.device
    B, _, H, W = batch_images.shape

    # --- STEP 1: Compute coordinates for all images ---
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

    # --- STEP 2: Compute base theoretical field for the whole batch ---
    T_star = gauss_legendre_integral(
        krapez_integrand_finite_depth_torch,
        X_star, Y_star,
        tau_min=0, tau_max=10000, n_points=1000
    )

    const = (2 * alpha * P0) / (
        K * torch.sqrt(torch.tensor(torch.pi, dtype=X.dtype, device=X.device)) *
        torch.tensor(torch.pi, dtype=X.dtype, device=X.device) * r_s
    )

    theoretical_field_batch = T_star * const  # [B,H,W]

    # --- STEP 3: Per-image symmetry, masking, normalization ---
    theoretical_fields, theoretical_norms, theoretical_norm_masks = [], [], []
    mask_v_list, mask_h_list = [], []

    for i in range(B):
        img = batch_images[i, 0]
        y_spot, x_spot = centers[i]
        theoretical_field = theoretical_field_batch[i].clone()

        # --- Detect lines ---
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

        # --- Add mirrored contributions ---
        if sym_v is not None:
            x_m = (torch.arange(W, device=device) - sym_v[0]) * pixel_size
            y_m = (torch.arange(H, device=device) - sym_v[1]) * pixel_size
            Xv, Yv = torch.meshgrid(y_m, x_m, indexing="ij")
            Xv_star, Yv_star = Xv / r_s, Yv / r_s
            T_sym_v = gauss_legendre_integral(
                krapez_integrand_finite_depth_torch,
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
                krapez_integrand_finite_depth_torch,
                Xh_star[None], Yh_star[None],
                tau_min=0, tau_max=10000, n_points=1000
            )[0]
            theoretical_field += T_sym_h * const

        # --- Normalize ---
        theoretical_norm = (theoretical_field - theoretical_field.min()) / (
            theoretical_field.max() - theoretical_field.min() + 1e-8
        )

        # --- Apply Otsu mask ---
        otsu_threshold = threshold_otsu(img.detach().cpu().numpy())
        otsu_mask = (img > otsu_threshold).float()
        theoretical_norm_mask = theoretical_norm * otsu_mask
        if mask_v is not None:
            theoretical_norm_mask *= mask_v
        if mask_h is not None:
            theoretical_norm_mask *= mask_h

        # --- Store per-image results ---
        theoretical_fields.append(theoretical_field.unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0))
        theoretical_norm_masks.append(theoretical_norm_mask.unsqueeze(0))
        mask_v_list.append(mask_v.unsqueeze(0) if mask_v is not None else None)
        mask_h_list.append(mask_h.unsqueeze(0) if mask_h is not None else None)

    # --- STEP 4: Combine into batch tensors ---
    theoretical_field_batch = torch.cat(theoretical_fields, dim=0)
    theoretical_norm_batch = torch.cat(theoretical_norms, dim=0)
    theoretical_norm_mask_batch = torch.cat(theoretical_norm_masks, dim=0)
    mask_v_batch = [m for m in mask_v_list if m is not None]
    mask_h_batch = [m for m in mask_h_list if m is not None]

    return theoretical_norm_mask_batch, mask_h_batch


def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=500):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral
def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))
def krapez_integrand_finite_depth_torch(Fo, x_star, y_star):
            device, dtype = Fo.device, Fo.dtype
            r_s_intern = r_s / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            Pe = (V * r_s_intern) / D  # Péclet number
            depth_star = depth / r_s_intern
            Fo_safe = torch.where(Fo <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo)
            A = 1.0 + 8.0 * Fo_safe
            exp_term = -2.0 * ((x_star**2) + (y_star + Pe * Fo_safe)**2) / A
            Fo_e = Fo_safe / (depth_star**2)
            g1 = green_depth_torch(Fo_e)
            integrand = (g1 * torch.exp(exp_term)) / (A * torch.sqrt(Fo_safe))
            integrand = torch.where(Fo > 0, integrand, torch.tensor(0.0, dtype=dtype, device=device))
            return integrand
    
def residual_cracks_mirrored_depth_clean(batch_images):
    """
    Compute residuals between experimental and analytical finite-depth temperature fields.
    Returns only the residuals (batch of shape [B, 1, H, W]).
    """
    # --- Compute analytical solution for the batch ---
    theoretical_field_batch, theoretical_norm_batch, _, _, mask_h_batch = analytical_solution(batch_images)

    residuals = []
    B = batch_images.shape[0]

    # --- Loop over batch to compute experimental normalization & residuals ---
    for i in range(B):
        image = batch_images[i, 0]

        # Compute Otsu threshold and normalize
        otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
        norm_threshold = torch.tensor(
            np.percentile(image[image > otsu_threshold].detach().cpu().numpy(), 55),
            dtype=image.dtype,
            device=image.device
        )
        experimental_norm = normalize_image_with_threshold(image, norm_threshold)

        # Compute residual
        residual = experimental_norm - theoretical_norm_batch[i]

        # Apply horizontal mask if available
        if i < len(mask_h_batch) and mask_h_batch[i] is not None:
            residual *= mask_h_batch[i].squeeze()

        residuals.append(residual.unsqueeze(0).unsqueeze(0))

    # --- Stack results into a batch ---
    residuals = torch.cat(residuals, dim=0)
    return residuals



def residual_cracks_mirrored_padded_depth(batch_images):
    def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
        nodes, weights = roots_legendre(n_points)
        nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
        weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
        tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
        weights = weights * (tau_max - tau_min) / 2
        integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
        integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
        return integral

    def salazar_integrand(tau, x, y):
        denom = r_s**2 + 8 * D * (-tau)
        exp_term = -2 * (x**2 + (y - V * tau)**2) / denom
        return torch.exp(exp_term) / (torch.sqrt(-tau) * denom)

    device = batch_images.device
    batch_size, _, height, width = batch_images.shape

    pad = 120  # extend domain by 120 px in each direction
    H_big, W_big = height + 2 * pad, width + 2 * pad

    residuals, theoretical_norms, theoretical_norms_mask, experimental_norms = [], [], [], []
    X_list, Y_list, centers = [], [], []
    vert_line_positions, horiz_line_positions = [], []
    mask_v_list, mask_h_list = [], []

    # --- compute coordinate grids with padding ---
    for i in range(batch_size):
        img = batch_images[i, 0]
        y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
        centers.append((y_spot.item(), x_spot.item()))

        x_meters = (torch.arange(W_big, device=device) - (x_spot + pad)) * pixel_size
        y_meters = (torch.arange(H_big, device=device) - (y_spot + pad)) * pixel_size
        X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
        X_list.append(X)
        Y_list.append(Y)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    # --- compute base theoretical field on padded domain ---
    T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=500)
    const = (2 * alpha * P0) / (
        (K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device)))
        * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device))
    )
    theoretical_field_batch = T_star * const  # [B, H_big, W_big]

    # --- detect lines & masks (still done on 120×120 original images) ---
    for i in range(batch_size):
        image_for_lines = batch_images[i, 0].detach().cpu().numpy()
        image_uint8 = normalize_image(image_for_lines)
        clahe_blurred_img = preprocess_image(image_uint8)

        line_sets = [
            detect_lines_custom(image_uint8, 1.0, 1.2, 0),
            detect_lines_custom(image_uint8, 1.0, 1.2, 1),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1),
        ]
        filtered = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
        all_filtered = [ln for sub in filtered if sub is not None for ln in sub]
        final_filtered = filter_longest_horizontal_vertical(all_filtered, min_length=13)

        vert_line_x = None
        horiz_line_y = None
        mask_v = mask_h = None
        y_spot, x_spot = centers[i]

        if final_filtered is not None and len(final_filtered) == 2:
            line1, line2 = final_filtered
            horiz_line = line1[0]
            vert_line = line2[0]

            vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
            horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))

            mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)
            mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)

        elif final_filtered is not None and len(final_filtered) == 1:
            line = final_filtered[0]
            angle = calculate_angle(*line[0])
            if 70 <= angle <= 110:
                vert_line = line[0]
                vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
                mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)
            else:
                horiz_line = line[0]
                horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))
                mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)

        vert_line_positions.append(vert_line_x)
        horiz_line_positions.append(horiz_line_y)
        mask_v_list.append(mask_v)
        mask_h_list.append(mask_h)

    # --- vectorized mirroring with depth reflection (no interpolation) ---
    B, H, W = theoretical_field_batch.shape
    x = torch.arange(W, device=device).unsqueeze(0).expand(B, W)
    y = torch.arange(H, device=device).unsqueeze(0).expand(B, H)

    # replace None by center if no line detected
    vert_line_x = torch.tensor(
        [v + pad if v is not None else W // 2 for v in vert_line_positions],
        dtype=torch.long, device=device)
    horiz_line_y = torch.tensor(
        [h + pad if h is not None else H // 2 for h in horiz_line_positions],
        dtype=torch.long, device=device)

    # --- 1) standard horizontal & vertical mirrors ---
    mirrored_x_all = 2 * vert_line_x[:, None] - x
    mirrored_y_all = 2 * horiz_line_y[:, None] - y
    mirrored_x_all = mirrored_x_all.clamp(0, W - 1)
    mirrored_y_all = mirrored_y_all.clamp(0, H - 1)

    T_base = theoretical_field_batch
    T_mirrored_v = T_base.gather(2, mirrored_x_all[:, None, :].expand(B, H, W))
    T_mirrored_h = T_base.gather(1, mirrored_y_all[:, :, None].expand(B, H, W))

    # --- 2) depth mirror (distance-based reflection) ---
    # Convert pixel coordinates to meters (relative to beam center)
    x_m = (x - W // 2) * pixel_size
    depth_eff = 2.0 * depth  # mirror depth (meters)

    # Compute equivalent horizontal position for same r^2 = x^2 + (2L)^2
    x_m_mirror = torch.sign(x_m) * torch.sqrt(x_m**2 + depth_eff**2)

    # Convert back to pixel indices
    x_mirror_idx = (x_m_mirror / pixel_size + W // 2).long()
    x_mirror_idx = x_mirror_idx.clamp(0, W - 1)

    # Gather the mirrored depth contribution
    T_mirrored_depth = T_base.gather(2, x_mirror_idx[:, None, :].expand(B, H, W))

    # --- 3) combine all contributions ---
    T_combined_big = T_base + T_mirrored_v + T_mirrored_h + T_mirrored_depth

    # --- 4) crop back to original 120×120 region ---
    theoretical_field_batch = T_combined_big[:, pad:pad + height, pad:pad + width]

    y_spot, _ = centers[i]

    # --- show the 2D fields ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(T_combined_big[0].cpu(), cmap='hot')
    axes[0].set_title("Final (base + mirrors)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(T_mirrored_h[0].cpu(), cmap='hot')
    axes[1].set_title("Vertical mirror $T_v$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(T_mirrored_h[0].cpu(), cmap='hot')
    axes[2].set_title("Horizontal mirror $T_h$")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

    
    # --- normalization and residual computation ---
    for i in range(B):
        image = batch_images[i, 0]
        otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
        otsu_mask = (image > otsu_threshold).float()

        nz = image[image > otsu_threshold]
        p55 = np.percentile(nz.detach().cpu().numpy(), 55) if nz.numel() > 0 else float(image.max().item())
        norm_threshold = torch.tensor(p55, dtype=image.dtype, device=image.device)
        experimental_norm = normalize_image_with_threshold(image, norm_threshold)

        tfield = theoretical_field_batch[i]
        tmin, tmax = tfield.min(), tfield.max()
        theoretical_norm = (tfield - tmin) / (tmax - tmin + 1e-8)
        theoretical_norm_mask = theoretical_norm * otsu_mask

        if mask_v_list[i] is not None:
            theoretical_norm_mask *= mask_v_list[i]
        if mask_h_list[i] is not None:
            theoretical_norm_mask *= mask_h_list[i]

        residual = experimental_norm - theoretical_norm
        if mask_h_list[i] is not None:
            residual *= mask_h_list[i]

        residuals.append(residual.unsqueeze(0).unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0))
        theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0))
        experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0))

    residuals = torch.cat(residuals, dim=0)
    theoretical_norms = torch.cat(theoretical_norms, dim=0)
    theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
    experimental_norms = torch.cat(experimental_norms, dim=0)

    return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms

def residual_cracks_mirrored_padded(batch_images):
    def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
        nodes, weights = roots_legendre(n_points)
        nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
        weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
        tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
        weights = weights * (tau_max - tau_min) / 2
        integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
        integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
        return integral

    def salazar_integrand(tau, x, y):
        denom = r_s**2 + 8 * D * (-tau)
        exp_term = -2 * (x**2 + (y - V * tau)**2) / denom
        return torch.exp(exp_term) / (torch.sqrt(-tau) * denom)

    device = batch_images.device
    batch_size, _, height, width = batch_images.shape

    pad = 120  # extend domain by 120 px in each direction
    H_big, W_big = height + 2 * pad, width + 2 * pad

    residuals, theoretical_norms, theoretical_norms_mask, experimental_norms = [], [], [], []
    X_list, Y_list, centers = [], [], []
    vert_line_positions, horiz_line_positions = [], []
    mask_v_list, mask_h_list = [], []

    # --- compute coordinate grids with padding ---
    for i in range(batch_size):
        img = batch_images[i, 0]
        y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
        centers.append((y_spot.item(), x_spot.item()))

        x_meters = (torch.arange(W_big, device=device) - (x_spot + pad)) * pixel_size
        y_meters = (torch.arange(H_big, device=device) - (y_spot + pad)) * pixel_size
        X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
        X_list.append(X)
        Y_list.append(Y)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    # --- compute base theoretical field on padded domain ---
    T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=500)
    const = (2 * alpha * P0) / (
        (K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device)))
        * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device))
    )
    theoretical_field_batch = T_star * const  # [B, H_big, W_big]

    # --- detect lines & masks (still done on 120×120 original images) ---
    for i in range(batch_size):
        image_for_lines = batch_images[i, 0].detach().cpu().numpy()
        image_uint8 = normalize_image(image_for_lines)
        clahe_blurred_img = preprocess_image(image_uint8)

        line_sets = [
            detect_lines_custom(image_uint8, 1.0, 1.2, 0),
            detect_lines_custom(image_uint8, 1.0, 1.2, 1),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1),
        ]
        filtered = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
        all_filtered = [ln for sub in filtered if sub is not None for ln in sub]
        final_filtered = filter_longest_horizontal_vertical(all_filtered, min_length=13)

        vert_line_x = None
        horiz_line_y = None
        mask_v = mask_h = None
        y_spot, x_spot = centers[i]

        if final_filtered is not None and len(final_filtered) == 2:
            line1, line2 = final_filtered
            horiz_line = line1[0]
            vert_line = line2[0]

            vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
            horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))

            mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)
            mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)

        elif final_filtered is not None and len(final_filtered) == 1:
            line = final_filtered[0]
            angle = calculate_angle(*line[0])
            if 70 <= angle <= 110:
                vert_line = line[0]
                vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
                mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)
            else:
                horiz_line = line[0]
                horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))
                mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)

        vert_line_positions.append(vert_line_x)
        horiz_line_positions.append(horiz_line_y)
        mask_v_list.append(mask_v)
        mask_h_list.append(mask_h)

    B, H, W = theoretical_field_batch.shape
    x = torch.arange(W, device=device).unsqueeze(0).expand(B, W)
    y = torch.arange(H, device=device).unsqueeze(0).expand(B, H)

    # replace None by center if no line detected
    vert_line_x = torch.tensor(
        [v + pad if v is not None else W // 2 for v in vert_line_positions],
        dtype=torch.long, device=device)
    horiz_line_y = torch.tensor(
        [h + pad if h is not None else H // 2 for h in horiz_line_positions],
        dtype=torch.long, device=device)

    # reflection indices for cracks / edges
    mirrored_x_all = 2 * vert_line_x[:, None] - x
    mirrored_y_all = 2 * horiz_line_y[:, None] - y

    # reflection for depth (horizontal-like)
    depth_pixels = int(round(depth / pixel_size))
    mirrored_x_depth = x + depth_pixels  # shift to simulate mirror below surface

    # clamp all indices
    mirrored_x_all = mirrored_x_all.clamp(0, W - 1)
    mirrored_y_all = mirrored_y_all.clamp(0, H - 1)
    mirrored_x_depth = mirrored_x_depth.clamp(0, W - 1)

    # gather all mirrored fields
    T_base = theoretical_field_batch
    T_mirrored_v = T_base.gather(2, mirrored_x_all[:, None, :].expand(B, H, W))
    T_mirrored_h = T_base.gather(1, mirrored_y_all[:, :, None].expand(B, H, W))
    T_mirrored_depth = T_base.gather(2, mirrored_x_depth[:, None, :].expand(B, H, W))

    # combine all contributions
    T_combined_big = T_base + T_mirrored_v + T_mirrored_h + T_mirrored_depth

    # --- crop back to original 120×120 region around the laser spot ---
    theoretical_field_batch = T_combined_big[:, pad:pad + height, pad:pad + width]
    y_spot, _ = centers[i]

    # # --- show the 2D fields ---
    # fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # im0 = axes[0].imshow(T_combined_big[0].cpu(), cmap='hot')
    # axes[0].set_title("Final (base + mirrors)")
    # plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # im1 = axes[1].imshow(T_mirrored_h[0].cpu(), cmap='hot')
    # axes[1].set_title("Vertical mirror $T_v$")
    # plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # im2 = axes[2].imshow(T_mirrored_h[0].cpu(), cmap='hot')
    # axes[2].set_title("Horizontal mirror $T_h$")
    # plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # plt.tight_layout()
    # plt.show()

    
    # --- normalization and residual computation ---
    for i in range(B):
        image = batch_images[i, 0]
        otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
        otsu_mask = (image > otsu_threshold).float()

        nz = image[image > otsu_threshold]
        p55 = np.percentile(nz.detach().cpu().numpy(), 55) if nz.numel() > 0 else float(image.max().item())
        norm_threshold = torch.tensor(p55, dtype=image.dtype, device=image.device)
        experimental_norm = normalize_image_with_threshold(image, norm_threshold)

        tfield = theoretical_field_batch[i]
        tmin, tmax = tfield.min(), tfield.max()
        theoretical_norm = (tfield - tmin) / (tmax - tmin + 1e-8)
        theoretical_norm_mask = theoretical_norm * otsu_mask

        if mask_v_list[i] is not None:
            theoretical_norm_mask *= mask_v_list[i]
        if mask_h_list[i] is not None:
            theoretical_norm_mask *= mask_h_list[i]

        residual = experimental_norm - theoretical_norm
        if mask_h_list[i] is not None:
            residual *= mask_h_list[i]

        residuals.append(residual.unsqueeze(0).unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0))
        theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0))
        experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0))

    residuals = torch.cat(residuals, dim=0)
    theoretical_norms = torch.cat(theoretical_norms, dim=0)
    theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
    experimental_norms = torch.cat(experimental_norms, dim=0)

    return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms

def residual_cracks_mirrored_new_version_vertical(batch_images):
    def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
        nodes, weights = roots_legendre(n_points)
        nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
        weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
        tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
        weights = weights * (tau_max - tau_min) / 2
        integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
        integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
        return integral

    def salazar_integrand(tau, x, y):
        denom = r_s**2 + 8 * D * (-tau)
        exp_term = -2 * (x**2 + (y - V * tau)**2) / denom
        return torch.exp(exp_term) / (torch.sqrt(-tau) * denom)

    device = batch_images.device
    batch_size, _, height, width = batch_images.shape

    residuals, theoretical_norms, theoretical_norms_mask, experimental_norms = [], [], [], []
    X_list, Y_list, centers = [], [], []
    vert_line_positions, horiz_line_positions = [], []
    mask_v_list, mask_h_list = [], []

    # --- coordinate grids centered on the laser spot ---
    for i in range(batch_size):
        img = batch_images[i, 0]
        y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
        centers.append((y_spot.item(), x_spot.item()))

        x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
        y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
        X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
        X_list.append(X)
        Y_list.append(Y)

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    # --- compute base theoretical field ---
    T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
    const = (2 * alpha * P0) / (
        (K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device)))
        * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device))
    )
    theoretical_field_batch = T_star * const

    # --- detect lines & masks ---
    for i in range(batch_size):
        image_for_lines = batch_images[i, 0].detach().cpu().numpy()
        image_uint8 = normalize_image(image_for_lines)
        clahe_blurred_img = preprocess_image(image_uint8)

        line_sets = [
            detect_lines_custom(image_uint8, 1.0, 1.2, 0),
            detect_lines_custom(image_uint8, 1.0, 1.2, 1),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
            detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1),
        ]
        filtered = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
        all_filtered = [ln for sub in filtered if sub is not None for ln in sub]
        final_filtered = filter_longest_horizontal_vertical(all_filtered, min_length=13)

        vert_line_x = None
        horiz_line_y = None
        mask_v = mask_h = None
        y_spot, x_spot = centers[i]

        if final_filtered is not None and len(final_filtered) == 2:
            line1, line2 = final_filtered
            horiz_line = line1[0]
            vert_line = line2[0]

            # get approximate line positions (axis coordinates)
            vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
            horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))

            mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)
            mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                  device=device, dtype=torch.float32)

        elif final_filtered is not None and len(final_filtered) == 1:
            line = final_filtered[0]
            angle = calculate_angle(*line[0])
            if 70 <= angle <= 110:
                vert_line = line[0]
                vert_line_x = int(round((vert_line[0] + vert_line[2]) / 2))
                mask_v = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)
            else:
                horiz_line = line[0]
                horiz_line_y = int(round((horiz_line[1] + horiz_line[3]) / 2))
                mask_h = torch.tensor(create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot),
                                      device=device, dtype=torch.float32)

        vert_line_positions.append(vert_line_x)
        horiz_line_positions.append(horiz_line_y)
        mask_v_list.append(mask_v)
        mask_h_list.append(mask_h)

    # --- vectorized mirroring using lines as symmetry axes ---
    B, H, W = theoretical_field_batch.shape
    x = torch.arange(W, device=device).unsqueeze(0).expand(B, W)
    y = torch.arange(H, device=device).unsqueeze(0).expand(B, H)

    # replace None by center if no line detected
    vert_line_x = torch.tensor([v if v is not None else W // 2 for v in vert_line_positions],
                               dtype=torch.long, device=device)
    horiz_line_y = torch.tensor([h if h is not None else H // 2 for h in horiz_line_positions],
                                dtype=torch.long, device=device)

    mirrored_x_all = 2 * vert_line_x[:, None] - x
    mirrored_y_all = 2 * horiz_line_y[:, None] - y
    mirrored_x_all = torch.clamp(mirrored_x_all, 0, W - 1)
    mirrored_y_all = torch.clamp(mirrored_y_all, 0, H - 1)

    # reflect the base field
    T_base = theoretical_field_batch
    T_mirrored_v = T_base.gather(2, mirrored_x_all[:, None, :].expand(B, H, W))
    T_mirrored_h = T_base.gather(1, mirrored_y_all[:, :, None].expand(B, H, W))
    theoretical_field_batch = T_base + T_mirrored_v + T_mirrored_h

    y_spot, _ = centers[i]

    # --- show the 2D fields ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(T_base[0].cpu(), cmap='hot')
    axes[0].set_title("Final (base + mirrors)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(T_mirrored_v[0].cpu(), cmap='hot')
    axes[1].set_title("Vertical mirror $T_v$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(T_mirrored_h[0].cpu(), cmap='hot')
    axes[2].set_title("Horizontal mirror $T_h$")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

    # --- normalize and compute residuals ---
    for i in range(B):
        image = batch_images[i, 0]
        otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
        otsu_mask = (image > otsu_threshold).float()

        nz = image[image > otsu_threshold]
        p55 = np.percentile(nz.detach().cpu().numpy(), 55) if nz.numel() > 0 else float(image.max().item())
        norm_threshold = torch.tensor(p55, dtype=image.dtype, device=image.device)
        experimental_norm = normalize_image_with_threshold(image, norm_threshold)

        tfield = theoretical_field_batch[i]
        tmin, tmax = tfield.min(), tfield.max()
        theoretical_norm = (tfield - tmin) / (tmax - tmin + 1e-8)
        theoretical_norm_mask = theoretical_norm * otsu_mask

        if mask_v_list[i] is not None:
            theoretical_norm_mask *= mask_v_list[i]
        if mask_h_list[i] is not None:
            theoretical_norm_mask *= mask_h_list[i]

        residual = experimental_norm - theoretical_norm
        if mask_h_list[i] is not None:
            residual *= mask_h_list[i]

        residuals.append(residual.unsqueeze(0).unsqueeze(0))
        theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0))
        theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0))
        experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0))

    residuals = torch.cat(residuals, dim=0)
    theoretical_norms = torch.cat(theoretical_norms, dim=0)
    theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
    experimental_norms = torch.cat(experimental_norms, dim=0)

    return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms


# Krapez integrand function
def krapez_integrand(Fo, x_star, y_star):
    r_s = radius / np.sqrt(2)
    Pe = (V * r_s) / D 
    Fo_safe = np.where(Fo <= 0, 1e-12, Fo)
    A = 1 + 8 * Fo_safe
    exp_term = -2 * ((x_star) ** 2 + (y_star + Pe * Fo_safe) ** 2) / A
    return np.exp(exp_term) / (A * np.sqrt(Fo_safe))

def green_depth_torch(Fo_e, nmax=200):
    device, dtype = Fo_e.device, Fo_e.dtype
    Fo_e_safe = torch.where(Fo_e <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo_e)
    n = torch.arange(1, nmax + 1, dtype=dtype, device=device).view(nmax, 1, 1, 1)
    exp_term = torch.exp(- (n ** 2) / Fo_e_safe.unsqueeze(0))  # [nmax, B, H, W]
    s = exp_term.sum(dim=0)
    g1 = 1.0 + 2.0 * s
    return g1

def residual_cracks_mirrored_depth(batch_images):
        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=500):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))
        def krapez_integrand_finite_depth_torch(Fo, x_star, y_star):
            device, dtype = Fo.device, Fo.dtype
            r_s_intern = r_s / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            Pe = (V * r_s_intern) / D  # Péclet number
            depth_star = depth / r_s_intern
            Fo_safe = torch.where(Fo <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo)
            A = 1.0 + 8.0 * Fo_safe
            exp_term = -2.0 * ((x_star**2) + (y_star + Pe * Fo_safe)**2) / A
            Fo_e = Fo_safe / (depth_star**2)
            g1 = green_depth_torch(Fo_e)
            integrand = (g1 * torch.exp(exp_term)) / (A * torch.sqrt(Fo_safe))
            integrand = torch.where(Fo > 0, integrand, torch.tensor(0.0, dtype=dtype, device=device))
            return integrand
        
        device = batch_images.device
        batch_size, _, height, width = batch_images.shape

        residuals = []
        theoretical_norms = []
        theoretical_norms_mask = []
        experimental_norms = []
        X_list, Y_list, centers = [], [], []
        for i in range(batch_size):
            img = batch_images[i, 0]
            y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
            centers.append((y_spot.item(), x_spot.item()))

            x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
            y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
            X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
            X_list.append(X)
            Y_list.append(Y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        X_star = X / r_s
        Y_star = Y / r_s
        # T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0)
        T_star = gauss_legendre_integral(krapez_integrand_finite_depth_torch, X_star, Y_star, tau_min=0, tau_max=10000, n_points=1000)
        const = (2 * alpha * P0) / (K * torch.sqrt(torch.tensor(torch.pi, dtype=X.dtype, device=X.device)) * 
                                    torch.tensor(torch.pi, dtype=X.dtype, device=X.device) * r_s)        
        theoretical_field_batch = T_star * const
        # theoretical_field_batch = torch.zeros(theoretical_field_batch.size(), device=device)

        for i in range(batch_size):
            image_for_lines = batch_images[i, 0].detach().cpu().numpy()
            image_uint8 = normalize_image(image_for_lines)
            clahe_blurred_img = preprocess_image(image_uint8)

            line_sets = [
                detect_lines_custom(image_uint8, 1.0, 1.2, 0),
                detect_lines_custom(image_uint8, 1.0, 1.2, 1),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
            ]
            filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
            all_filtered_lines = [line for sublist in filtered_lines if sublist is not None for line in sublist]
            final_filtered_lines = filter_longest_horizontal_vertical(all_filtered_lines, min_length=13)

            y_spot, x_spot = centers[i]
            sym_h = sym_v = None
            vert_line = horiz_line = None
            mask_v = mask_h = None

            if final_filtered_lines is not None and len(final_filtered_lines) == 2:
                line1, line2 = final_filtered_lines
                horiz_line = line1[0]
                vert_line = line2[0]
                sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                mask_v = torch.tensor(mask_v, device=device)  
                mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                mask_h = torch.tensor(mask_h, device=device)  
            elif final_filtered_lines is not None and len(final_filtered_lines) == 1:
                line = final_filtered_lines[0]
                angle = calculate_angle(*line[0])
                if 70 <= angle <= 110:
                    vert_line = line[0]
                    sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                    mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                    mask_v = torch.tensor(mask_v, device=device)  
                else:
                    horiz_line = line[0]
                    sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                    mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                    mask_h = torch.tensor(mask_h, device=device)  

            theoretical_field = theoretical_field_batch[i].clone()

            if sym_v is not None:
                x_meters = (torch.arange(width, device=device) - sym_v[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_v[1]) * pixel_size
                Xv, Yv = torch.meshgrid(y_meters, x_meters, indexing="ij")
                Xv_star = Xv / r_s
                Yv_star = Yv / r_s
                # T_sym_v = gauss_legendre_integral(salazar_integrand, Xv[None], Yv[None], tau_min=-500, tau_max=0)
                T_sym_v = gauss_legendre_integral(krapez_integrand_finite_depth_torch, Xv_star[None], Yv_star[None], tau_min=0, tau_max=10000, n_points=1000)
                theoretical_field += T_sym_v[0] * const
            if sym_h is not None:
                x_meters = (torch.arange(width, device=device) - sym_h[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_h[1]) * pixel_size
                Xh, Yh = torch.meshgrid(y_meters, x_meters, indexing="ij")
                Xh_star = Xh / r_s
                Yh_star = Yh / r_s                
                T_sym_h = gauss_legendre_integral(krapez_integrand_finite_depth_torch, Xh_star[None], Yh_star[None], tau_min=0, tau_max=10000, n_points=1000)
                theoretical_field += T_sym_h[0] * const

            image = batch_images[i, 0]
            otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
            otsu_mask = (image > otsu_threshold).float()
            norm_threshold = torch.tensor(np.percentile(image[image > otsu_threshold].detach().cpu().numpy(), 0), 
                     dtype=image.dtype, device=image.device)
            experimental_norm = normalize_image_with_threshold(image, norm_threshold)

            theoretical_norm = (theoretical_field - theoretical_field.min()) / (theoretical_field.max() - theoretical_field.min() + 1e-8)
            theoretical_norm_mask = theoretical_norm * otsu_mask
            if mask_v is not None:
                theoretical_norm_mask *= mask_v
            if mask_h is not None:
                theoretical_norm_mask *= mask_h

            residual =  experimental_norm - theoretical_norm
            # if mask_v is not None:
            #     residual *= mask_v
            if mask_h is not None:
                residual *= mask_h
            residuals.append(residual.unsqueeze(0).unsqueeze(0))
            theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            img_np = theoretical_norm_mask.squeeze().cpu().numpy()  # shape HxW
            img_np = exposure.adjust_gamma(img_np, 0.7)

            h, w = img_np.shape

        #     plt.imshow(img_np, cmap="hot",vmin=0,vmax=1)
        #     plt.axis('off')
        #     # plt.scatter(x_spot, y_spot, color='white', marker='o', s=100, label='Original spot')
        #     plt.scatter(sym_v[0], sym_v[1], color='cyan', marker='x', s=100, label='Crack virtual mirror spot')  
        #     plt.scatter(sym_h[0], sym_h[1], color='magenta', marker='x', s=100, label='Edge virtual mirror spot')
        #     # Optionally plot the detected lines
        #     # if vert_line is not None:
        #     #     x1, y1, x2, y2 = vert_line
        #     #     plot_infinite_line(x1, y1, x2, y2, (h, w), color="cyan", linewidth=2.5)
        #     # if horiz_line is not None:
        #     #     x1, y1, x2, y2 = horiz_line
        #     #     plot_infinite_line(x1, y1, x2, y2, (h, w), color="magenta", linewidth=2.5)

        #     plt.legend(
        #         loc="lower right",
        #         bbox_to_anchor=(0.62, 0.05),  # shift upward (increase second value)
        #         frameon=True,
        #         fontsize=10
        #     )            
        # plt.savefig("figures/symmetry_plot.svg", format="svg", bbox_inches="tight")            
        # plt.show()
        residuals = torch.cat(residuals, dim=0)              # [B, 1, H, W]
        theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
        theoretical_norms = torch.cat(theoretical_norms, dim=0)
        experimental_norms = torch.cat(experimental_norms, dim=0)
        return residuals, theoretical_norms,theoretical_norms_mask, experimental_norms

def residual_cracks_mirrored(batch_images):
        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))

        device = batch_images.device
        batch_size, _, height, width = batch_images.shape

        residuals = []
        theoretical_norms = []
        theoretical_norms_mask = []
        experimental_norms = []
        X_list, Y_list, centers = [], [], []

        for i in range(batch_size):
            img = batch_images[i, 0]
            y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
            centers.append((y_spot.item(), x_spot.item()))

            x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
            y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
            X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
            X_list.append(X)
            Y_list.append(Y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)

        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0)
        const = (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))
        theoretical_field_batch = T_star * const
        # theoretical_field_batch = torch.zeros(theoretical_field_batch.size(), device=device)

        for i in range(batch_size):
            image_for_lines = batch_images[i, 0].detach().cpu().numpy()
            image_uint8 = normalize_image(image_for_lines)
            clahe_blurred_img = preprocess_image(image_uint8)

            line_sets = [
                detect_lines_custom(image_uint8, 1.0, 1.2, 0),
                detect_lines_custom(image_uint8, 1.0, 1.2, 1),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
            ]
            filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
            all_filtered_lines = [line for sublist in filtered_lines if sublist is not None for line in sublist]
            final_filtered_lines = filter_longest_horizontal_vertical(all_filtered_lines, min_length=13)

            y_spot, x_spot = centers[i]
            sym_h = sym_v = None
            vert_line = horiz_line = None
            mask_v = mask_h = None

            if final_filtered_lines is not None and len(final_filtered_lines) == 2:
                line1, line2 = final_filtered_lines
                horiz_line = line1[0]
                vert_line = line2[0]
                sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                mask_v = torch.tensor(mask_v, device=device)  
                mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                mask_h = torch.tensor(mask_h, device=device)  
            elif final_filtered_lines is not None and len(final_filtered_lines) == 1:
                line = final_filtered_lines[0]
                angle = calculate_angle(*line[0])
                if 70 <= angle <= 110:
                    vert_line = line[0]
                    sym_v = reflect_point_across_line(*vert_line, x_spot, y_spot)
                    mask_v = create_vertical_line_mask(image_for_lines.shape, *vert_line, x_spot, y_spot)
                    mask_v = torch.tensor(mask_v, device=device)  
                else:
                    horiz_line = line[0]
                    sym_h = reflect_point_across_line(*horiz_line, x_spot, y_spot)
                    mask_h = create_vertical_line_mask(image_for_lines.shape, *horiz_line, x_spot, y_spot)
                    mask_h = torch.tensor(mask_h, device=device)  

            theoretical_field = theoretical_field_batch[i].clone()

            if sym_v is not None:
                x_meters = (torch.arange(width, device=device) - sym_v[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_v[1]) * pixel_size
                Xv, Yv = torch.meshgrid(y_meters, x_meters, indexing="ij")
                T_sym_v = gauss_legendre_integral(salazar_integrand, Xv[None], Yv[None], tau_min=-500, tau_max=0)
                theoretical_field += T_sym_v[0] * const
            if sym_h is not None:
                x_meters = (torch.arange(width, device=device) - sym_h[0]) * pixel_size
                y_meters = (torch.arange(height, device=device) - sym_h[1]) * pixel_size
                Xh, Yh = torch.meshgrid(y_meters, x_meters, indexing="ij")
                T_sym_h = gauss_legendre_integral(salazar_integrand, Xh[None], Yh[None], tau_min=-500, tau_max=0)
                theoretical_field += T_sym_h[0] * const

            image = batch_images[i, 0]
            otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
            otsu_mask = (image > otsu_threshold).float()
            norm_threshold = torch.tensor(np.percentile(image[image > otsu_threshold].detach().cpu().numpy(), 5), 
                     dtype=image.dtype, device=image.device)
            experimental_norm = normalize_image_with_threshold(image, norm_threshold)

            theoretical_norm = (theoretical_field - theoretical_field.min()) / (theoretical_field.max() - theoretical_field.min() + 1e-8)
            theoretical_norm_mask = theoretical_norm * otsu_mask
            if mask_v is not None:
                theoretical_norm_mask *= mask_v
            if mask_h is not None:
                theoretical_norm_mask *= mask_h

            residual =  experimental_norm - theoretical_norm
            # if mask_v is not None:
            #     residual *= mask_v
            if mask_h is not None:
                residual *= mask_h
            residuals.append(residual.unsqueeze(0).unsqueeze(0))
            theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            img_np = theoretical_norm_mask.squeeze().cpu().numpy()  # shape HxW
            img_np = exposure.adjust_gamma(img_np, 0.7)

            h, w = img_np.shape

        #     plt.imshow(img_np, cmap="hot",vmin=0,vmax=1)
        #     plt.axis('off')
        #     # plt.scatter(x_spot, y_spot, color='white', marker='o', s=100, label='Original spot')
        #     plt.scatter(sym_v[0], sym_v[1], color='cyan', marker='x', s=100, label='Crack virtual mirror spot')  
        #     plt.scatter(sym_h[0], sym_h[1], color='magenta', marker='x', s=100, label='Edge virtual mirror spot')
        #     # Optionally plot the detected lines
        #     # if vert_line is not None:
        #     #     x1, y1, x2, y2 = vert_line
        #     #     plot_infinite_line(x1, y1, x2, y2, (h, w), color="cyan", linewidth=2.5)
        #     # if horiz_line is not None:
        #     #     x1, y1, x2, y2 = horiz_line
        #     #     plot_infinite_line(x1, y1, x2, y2, (h, w), color="magenta", linewidth=2.5)

        #     plt.legend(
        #         loc="lower right",
        #         bbox_to_anchor=(0.62, 0.05),  # shift upward (increase second value)
        #         frameon=True,
        #         fontsize=10
        #     )            
        # plt.savefig("figures/symmetry_plot.svg", format="svg", bbox_inches="tight")            
        # plt.show()
        residuals = torch.cat(residuals, dim=0)              # [B, 1, H, W]
        theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
        theoretical_norms = torch.cat(theoretical_norms, dim=0)
        experimental_norms = torch.cat(experimental_norms, dim=0)
        return residuals, theoretical_norms,theoretical_norms_mask, experimental_norms

def residual_cracks(batch_images):
        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))

        device = batch_images.device
        batch_size, _, height, width = batch_images.shape
        residuals = []
        theoretical_norms = []
        theoretical_norms_mask = []
        experimental_norms = []
        X_list, Y_list, centers = [], [], []
        for i in range(batch_size):
            img = batch_images[i, 0]
            y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
            centers.append((y_spot.item(), x_spot.item()))
            x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
            y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
            X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
            X_list.append(X)
            Y_list.append(Y)
        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        const = (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))
        theoretical_field_batch = T_star * const
        mask_v_batch = torch.ones((batch_size, height, width), dtype=torch.float32, device=device)
        mask_h_batch = torch.ones((batch_size, height, width), dtype=torch.float32, device=device)
        for i in range(batch_size):
            image = batch_images[i, 0].detach().cpu().numpy()
            image_uint8 = normalize_image(image)
            clahe_blurred_img = preprocess_image(image_uint8)

            line_sets = [
                detect_lines_custom(image_uint8, 1.0, 1.2, 0),
                detect_lines_custom(image_uint8, 1.0, 1.2, 1),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
            ]
            filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
            all_filtered_lines = [line for sublist in filtered_lines if sublist is not None for line in sublist]
            final_filtered_lines = filter_longest_horizontal_vertical(all_filtered_lines, min_length=13)

            y_spot, x_spot = centers[i]
            vert_line = horiz_line = None

            if final_filtered_lines is not None and len(final_filtered_lines) == 2:
                line1, line2 = final_filtered_lines
                horiz_line = line1[0]
                vert_line = line2[0]
            elif final_filtered_lines is not None and len(final_filtered_lines) == 1:
                line = final_filtered_lines[0]
                angle = calculate_angle(*line[0])
                if 70 <= angle <= 110:
                    vert_line = line[0]
                else:
                    horiz_line = line[0]

            if vert_line is not None:
                mask_v = create_vertical_line_mask(image.shape, *vert_line, x_spot, y_spot)
                mask_v_batch[i] = torch.tensor(mask_v, device=device)
            if horiz_line is not None:
                mask_h = create_vertical_line_mask(image.shape, *horiz_line, x_spot, y_spot)
                mask_h_batch[i] = torch.tensor(mask_h, device=device)  
        for i in range(batch_size):
            image = batch_images[i, 0]
            otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
            otsu_mask = (image > otsu_threshold).float()
            norm_threshold = np.percentile(image[image > otsu_threshold].detach().cpu().numpy(), 30)

            experimental_norm = normalize_image_with_threshold(image, norm_threshold)

            theoretical = theoretical_field_batch[i]
            theoretical_norm = (theoretical - theoretical.min()) / (theoretical.max() - theoretical.min() + 1e-8)
            theoretical_norm_mask = theoretical_norm*otsu_mask
            if mask_v_batch[i] is not None:
                theoretical_norm_mask *= mask_v_batch[i]
            if mask_h_batch[i] is not None:
                theoretical_norm_mask *= mask_h_batch[i]
                
            # mask_v = mask_v_batch[i].detach().cpu().numpy()
            # plt.imshow(mask_v, cmap="gray")
            # plt.gca().xaxis.set_visible(False)
            # plt.gca().yaxis.set_visible(False)        
            # plt.savefig(f"figures/mask_v_{i}.png", bbox_inches="tight", dpi=300)
            # plt.show()

            residual =  experimental_norm - theoretical_norm
            # if mask_v_batch[i] is not None:
            #     residual *= mask_v_batch[i]
            if mask_h_batch[i] is not None:
                residual *= mask_h_batch[i]
            residuals.append(residual.unsqueeze(0).unsqueeze(0))
            theoretical_norms_mask.append(theoretical_norm_mask.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            theoretical_norms.append(theoretical_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
            experimental_norms.append(experimental_norm.unsqueeze(0).unsqueeze(0)) # [1, 1, H, W]
        residuals = torch.cat(residuals, dim=0)              # [B, 1, H, W]
        theoretical_norms_mask = torch.cat(theoretical_norms_mask, dim=0)
        theoretical_norms = torch.cat(theoretical_norms, dim=0)
        experimental_norms = torch.cat(experimental_norms, dim=0)
        return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms


def residual_cracks_fast(batch_images):
        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))

        device = batch_images.device
        batch_size, _, height, width = batch_images.shape
        X_list, Y_list, centers = [], [], []
        for i in range(batch_size):
            img = batch_images[i, 0]
            y_spot, x_spot = torch.unravel_index(torch.argmax(img), (height, width))
            centers.append((y_spot.item(), x_spot.item()))
            x_meters = (torch.arange(width, device=device) - x_spot) * pixel_size
            y_meters = (torch.arange(height, device=device) - y_spot) * pixel_size
            X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")
            X_list.append(X)
            Y_list.append(Y)
        X = torch.stack(X_list)
        Y = torch.stack(Y_list)
        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        const = (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))
        theoretical_field = T_star * const
        mask_v_batch = torch.ones((batch_size, height, width), dtype=torch.float32, device=device)
        mask_h_batch = torch.ones((batch_size, height, width), dtype=torch.float32, device=device)
        clipped_experimental = []

        for i in range(batch_size):
            img = batch_images[i, 0]  # (H, W), normalized to [0,1]
            otsu_thresh = torch.tensor(threshold_otsu(img.detach().cpu().numpy()),dtype=batch_images.dtype, device=batch_images.device)
            clipped_img = torch.clamp(img, min=otsu_thresh, max=img.max())   
            clipped_img = (clipped_img - otsu_thresh) / (img.max() - otsu_thresh + 1e-8)
            clipped_experimental.append(clipped_img)

            image = img.detach().cpu().numpy()
            image_uint8 = normalize_image(image)
            clahe_blurred_img = preprocess_image(image_uint8)

            line_sets = [
                detect_lines_custom(image_uint8, 1.0, 1.2, 0),
                detect_lines_custom(image_uint8, 1.0, 1.2, 1),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 0),
                detect_lines_custom(clahe_blurred_img, 1.0, 1.2, 1)
            ]
            filtered_lines = [filter_longest_horizontal_vertical(lines, min_length=13) for lines in line_sets]
            all_filtered_lines = [line for sublist in filtered_lines if sublist is not None for line in sublist]
            final_filtered_lines = filter_longest_horizontal_vertical(all_filtered_lines, min_length=13)

            y_spot, x_spot = centers[i]
            vert_line = horiz_line = None

            if final_filtered_lines is not None and len(final_filtered_lines) == 2:
                line1, line2 = final_filtered_lines
                horiz_line = line1[0]
                vert_line = line2[0]
            elif final_filtered_lines is not None and len(final_filtered_lines) == 1:
                line = final_filtered_lines[0]
                angle = calculate_angle(*line[0])
                if 70 <= angle <= 110:
                    vert_line = line[0]
                else:
                    horiz_line = line[0]

            if vert_line is not None:
                mask_v = create_vertical_line_mask(image.shape, *vert_line, x_spot, y_spot)
                mask_v_batch[i] = torch.tensor(mask_v, device=device)
            if horiz_line is not None:
                mask_h = create_vertical_line_mask(image.shape, *horiz_line, x_spot, y_spot)
                mask_h_batch[i] = torch.tensor(mask_h, device=device)  




        experimental_norm = torch.stack(clipped_experimental, dim=0)
        experimental_norm = experimental_norm.unsqueeze(1).to(device)
        otsu_mask = experimental_norm > 0

        theoretical_min = theoretical_field.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        theoretical_max = theoretical_field.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        theoretical_norm = (theoretical_field - theoretical_min) / (theoretical_max - theoretical_min)
        theoretical_norm = theoretical_norm.unsqueeze(1)
        theoretical_norm = theoretical_norm * otsu_mask.float() * mask_h_batch.float() * mask_v_batch.float()

        residual = experimental_norm - theoretical_norm
        masked_residual = residual * mask_h_batch.float() * mask_v_batch.float()
        return masked_residual

def residual_depth(batch_images):
        device = batch_images.device
        batch_size, _, height, width = batch_images.shape

        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def krapez_integrand_finite_depth_torch(Fo, x_star, y_star):
            device, dtype = Fo.device, Fo.dtype
            r_s_intern = r_s / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            Pe = (V * r_s_intern) / D  # Péclet number
            depth_star = depth / r_s_intern
            Fo_safe = torch.where(Fo <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo)
            A = 1.0 + 8.0 * Fo_safe
            exp_term = -2.0 * ((x_star**2) + (y_star + Pe * Fo_safe)**2) / A
            Fo_e = Fo_safe / (depth_star**2)
            g1 = green_depth_torch(Fo_e)
            integrand = (g1 * torch.exp(exp_term)) / (A * torch.sqrt(Fo_safe))
            integrand = torch.where(Fo > 0, integrand, torch.tensor(0.0, dtype=dtype, device=device))
            return integrand
        # Get the center of each image in the batch
        centers = torch.stack([
            torch.tensor(torch.unravel_index(torch.argmax(batch_images[i, 0]), (height, width)), device=device)
            for i in range(batch_size)
        ])

        # Create x and y coordinates for each image
        x_meters = [
            (torch.arange(width, device=device) - centers[i, 1]) * pixel_size for i in range(batch_size)
        ]
        y_meters = [
            (torch.arange(height, device=device) - centers[i, 0]) * pixel_size for i in range(batch_size)
        ]

        # Create meshgrid for each image
        X = torch.stack([torch.meshgrid(y_meters[i], x_meters[i], indexing="ij")[1] for i in range(batch_size)])
        Y = torch.stack([torch.meshgrid(y_meters[i], x_meters[i], indexing="ij")[0] for i in range(batch_size)])
        X_star = X / r_s
        Y_star = Y / r_s
        # Compute T_star and theoretical field
        # T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        # theoretical_field = T_star * (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))
        T_star = gauss_legendre_integral(krapez_integrand_finite_depth_torch, X_star, Y_star, tau_min=0, tau_max=10000, n_points=1000)
        const = (2 * alpha * P0) / (K * torch.sqrt(torch.tensor(torch.pi, dtype=X.dtype, device=X.device)) * 
                                    torch.tensor(torch.pi, dtype=X.dtype, device=X.device) * r_s)        
        theoretical_field = T_star * const
        # Normalize theoretical and experimental fields
        theoretical_min = theoretical_field.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        theoretical_max = theoretical_field.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        theoretical_norms = (theoretical_field - theoretical_min) / (theoretical_max - theoretical_min)
        theoretical_norms = theoretical_norms.unsqueeze(1)

        clipped_experimental = []
        residuals = []
        theoretical_norms_mask = []
        for i in range(batch_size):
            img = batch_images[i, 0]  # (H, W), normalized to [0,1]
            otsu_thresh = torch.tensor(threshold_otsu(img.detach().cpu().numpy()),dtype=batch_images.dtype, device=batch_images.device)
            otsu_mask = (img > otsu_thresh).float()  # 1 where img > threshold, 0 otherwise
            mask_v = otsu_mask.squeeze().detach().cpu().numpy()
            # plt.imshow(mask_v, cmap="gray")
            # plt.gca().xaxis.set_visible(False)
            # plt.gca().yaxis.set_visible(False)        
            # plt.savefig(f"figures/mask_v_{i}.png", bbox_inches="tight", dpi=300)
            # plt.show()
            norm_threshold = torch.tensor(np.percentile(img[img > otsu_thresh].detach().cpu().numpy(), 25), 
                     dtype=img.dtype, device=img.device)
            clipped_img = torch.clamp(img, min=norm_threshold, max=img.max())   
            clipped_img = (clipped_img - norm_threshold) / (img.max() - norm_threshold + 1e-8)
            clipped_experimental.append(clipped_img)   

            theoretical_norm_mask = theoretical_norms[i,0] * otsu_mask
            residual = theoretical_norm_mask - clipped_img
            theoretical_norms_mask.append(theoretical_norm_mask)
            residuals.append(residual)

        experimental_norms = torch.stack(clipped_experimental, dim=0)
        experimental_norms = experimental_norms.unsqueeze(1).to(device)
        theoretical_norms_mask = torch.stack(theoretical_norms_mask, dim=0)
        theoretical_norms_mask = theoretical_norms_mask.unsqueeze(1).to(device)
        residuals = torch.stack(residuals, dim=0)
        residuals = residuals.unsqueeze(1).to(device)
        return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms


def analytical_solution_uncracked_sample(batch_images):
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
        lambda Fo, x, y: krapez_integrand_finite_depth_torch(Fo, x, y),
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
                lambda Fo, x, y: krapez_integrand_finite_depth_torch(Fo, x, y),
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

def residual(batch_images):
        device = batch_images.device
        batch_size, _, height, width = batch_images.shape

        def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
            nodes, weights = roots_legendre(n_points)
            nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
            weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
            tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
            weights = weights * (tau_max - tau_min) / 2
            integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
            integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
            return integral

        def salazar_integrand(tau, x, y):
            exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
            return 1 / torch.sqrt(-tau) * torch.exp(exp_term) / (r_s**2 + 8 * D * (-tau))
        
        def krapez_integrand_finite_depth_torch(Fo, x_star, y_star):
            device, dtype = Fo.device, Fo.dtype
            r_s_intern = r_s / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            Pe = (V * r_s_intern) / D  # Péclet number
            depth_star = depth / r_s_intern
            Fo_safe = torch.where(Fo <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo)
            A = 1.0 + 8.0 * Fo_safe
            exp_term = -2.0 * ((x_star**2) + (y_star + Pe * Fo_safe)**2) / A
            Fo_e = Fo_safe / (depth_star**2)
            g1 = green_depth_torch(Fo_e)
            integrand = (g1 * torch.exp(exp_term)) / (A * torch.sqrt(Fo_safe))
            integrand = torch.where(Fo > 0, integrand, torch.tensor(0.0, dtype=dtype, device=device))
            return integrand
        # Get the center of each image in the batch
        centers = torch.stack([
            torch.tensor(torch.unravel_index(torch.argmax(batch_images[i, 0]), (height, width)), device=device)
            for i in range(batch_size)
        ])

        # Create x and y coordinates for each image
        x_meters = [
            (torch.arange(width, device=device) - centers[i, 1]) * pixel_size for i in range(batch_size)
        ]
        y_meters = [
            (torch.arange(height, device=device) - centers[i, 0]) * pixel_size for i in range(batch_size)
        ]

        # Create meshgrid for each image
        X = torch.stack([torch.meshgrid(y_meters[i], x_meters[i], indexing="ij")[1] for i in range(batch_size)])
        Y = torch.stack([torch.meshgrid(y_meters[i], x_meters[i], indexing="ij")[0] for i in range(batch_size)])
        # Compute T_star and theoretical field
        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        theoretical_field = T_star * (2 * alpha * P0) / ((K / torch.sqrt(torch.tensor(D, dtype=X.dtype, device=X.device))) * torch.sqrt(torch.tensor(torch.pi**3, dtype=X.dtype, device=X.device)))

        # Normalize theoretical and experimental fields
        theoretical_min = theoretical_field.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        theoretical_max = theoretical_field.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        theoretical_norms = (theoretical_field - theoretical_min) / (theoretical_max - theoretical_min)
        theoretical_norms = theoretical_norms.unsqueeze(1)

        clipped_experimental = []
        residuals = []
        theoretical_norms_mask = []
        for i in range(batch_size):
            img = batch_images[i, 0]  # (H, W), normalized to [0,1]
            otsu_thresh = torch.tensor(threshold_otsu(img.detach().cpu().numpy()),dtype=batch_images.dtype, device=batch_images.device)
            otsu_mask = (img > otsu_thresh).float()  # 1 where img > threshold, 0 otherwise
            mask_v = otsu_mask.squeeze().detach().cpu().numpy()
            # plt.imshow(mask_v, cmap="gray")
            # plt.gca().xaxis.set_visible(False)
            # plt.gca().yaxis.set_visible(False)        
            # plt.savefig(f"figures/mask_v_{i}.png", bbox_inches="tight", dpi=300)
            # plt.show()
            norm_threshold = torch.tensor(np.percentile(img[img > otsu_thresh].detach().cpu().numpy(), 25), 
                     dtype=img.dtype, device=img.device)
            clipped_img = torch.clamp(img, min=norm_threshold, max=img.max())   
            clipped_img = (clipped_img - norm_threshold) / (img.max() - norm_threshold + 1e-8)
            clipped_experimental.append(clipped_img)   

            theoretical_norm_mask = theoretical_norms[i,0] * otsu_mask
            residual = theoretical_norm_mask - clipped_img
            theoretical_norms_mask.append(theoretical_norm_mask)
            residuals.append(residual)

        experimental_norms = torch.stack(clipped_experimental, dim=0)
        experimental_norms = experimental_norms.unsqueeze(1).to(device)
        theoretical_norms_mask = torch.stack(theoretical_norms_mask, dim=0)
        theoretical_norms_mask = theoretical_norms_mask.unsqueeze(1).to(device)
        residuals = torch.stack(residuals, dim=0)
        residuals = residuals.unsqueeze(1).to(device)
        return residuals, theoretical_norms, theoretical_norms_mask, experimental_norms

# Load the first 4 images
# folder_path = r"synthetic_datasets_old_versions/synthesis_v_1.0_10-7_otsu_0.00075_dataset50"
# folder_path = r"synthetic_datasets_old_versions/synthesis_v_1.0_10-8_otsu_0.00075_dataset48"
# folder_path = r"synthetic_datasets_old_versions/synthesis_v_1.0_0.0_otsu_0.00075_dataset35"

# folder_path = r"/d/brahou/DDPM/denoising-diffusion-pytorch/synthetic_datasets_old_versions/cracks_0.0"
# folder_path = r"/d/brahou/DDPM/denoising-diffusion-pytorch/synthetic_datasets_old_versions/cracks_10-11"


folder_path = r"/d/brahou/data/flyd_frames_classification/data_flyd_frames_cropped/negative"

images = load_images_from_folder(folder_path)[0:10]
# Convert and normalize each image
batch_tensors = []
for _, img in images:  
    img_uint8 = normalize_image(img)  # Convert to uint8/float as needed
    img_tensor = torch.tensor(img_uint8, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    batch_tensors.append(img_tensor)

# Stack into one batch and move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = torch.cat(batch_tensors).to(device)  # Shape: [4, 1, H, W]
 
theoretical_norm_mask_batch, mask_h_batch = analytical_solution(batch)
theoretical_norm_mask_batch2, mask_h_batch = analytical_solution_uncracked_sample(batch)
residuals, theoretical_norms, theoretical_norms_masks, experimental_norms = residual_cracks_mirrored_depth(batch)

for i in range(30):
    theoretical_norm_mask_mirrored = theoretical_norm_mask_batch[i].squeeze(0).cpu().numpy()
    theoretical_norm_mask_mirrored2 = theoretical_norm_mask_batch2[i].squeeze(0).cpu().numpy()
    experimental_norm = experimental_norms[i].squeeze(0).cpu().numpy()
    theoretical_norm = theoretical_norms[i].squeeze(0).cpu().numpy()
    theoretical_norm_mask = theoretical_norms_masks[i].squeeze(0).cpu().numpy()
    res = torch.abs(residuals[i]).squeeze(0).cpu()
    # experimental_norm[110:120, 0:10] = 0
    # theoretical_norm_mask_mirrored = exposure.adjust_gamma(theoretical_norm_mask_mirrored, 0.7)
    # theoretical_norm_mask = exposure.adjust_gamma(theoretical_norm_mask, 0.7)
    # experimental_norm = exposure.adjust_gamma(experimental_norm, 0.7)

    # Spot x position for 1D plot
    y_spot, x_spot = np.unravel_index(np.argmax(experimental_norm), experimental_norm.shape)
    os.makedirs("figures", exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    fig = plt.figure(figsize=(12, 10))

    # --- 1) Experimental image ---
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    im_exp = ax1.imshow(experimental_norm, cmap='hot')
    ax1.set_xlabel("X (pixels)", fontsize=14)
    ax1.set_ylabel("Y (pixels)", fontsize=14)
    ax1.set_xticks([0, 30, 60, 90, 119])
    ax1.set_yticks([0, 30, 60, 90, 119])
    ax1.tick_params(axis='both', direction='out', length=6, labelsize=12, top=False, right=False)
    ax1.set_title("Experimental image", fontsize=16)
    plt.colorbar(im_exp, ax=ax1, pad=0.02, fraction=0.046)

    # --- 2) Analytical baseline ---
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    im0 = ax2.imshow(theoretical_norm_mask_mirrored2, cmap='hot')
    ax2.set_xlabel("X (pixels)", fontsize=14)
    ax2.set_ylabel("Y (pixels)", fontsize=14)
    ax2.set_xticks([0, 30, 60, 90, 119])
    ax2.set_yticks([0, 30, 60, 90, 119])
    ax2.tick_params(axis='both', direction='out', length=6, labelsize=12, top=False, right=False)
    ax2.set_title("Analytical baseline", fontsize=16)
    plt.colorbar(im0, ax=ax2, pad=0.02, fraction=0.046)

    # --- 3) New analytical version ---
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    im1 = ax3.imshow(theoretical_norm_mask_mirrored, cmap='hot')
    ax3.set_xlabel("X (pixels)", fontsize=14)
    ax3.set_ylabel("Y (pixels)", fontsize=14)
    ax3.set_xticks([0, 30, 60, 90, 119])
    ax3.set_yticks([0, 30, 60, 90, 119])
    ax3.tick_params(axis='both', direction='out', length=6, labelsize=12, top=False, right=False)
    ax3.set_title("New analytical version", fontsize=16)
    cbar = plt.colorbar(im1, ax=ax3, pad=0.02, fraction=0.046)
    cbar.set_label("Normalized temperature", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # --- 4) Combined line profile ---
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax4.plot(experimental_norm[y_spot, :], label='Experimental', color='black', linewidth=2)
    ax4.plot(theoretical_norm_mask[y_spot, :], label='Analytical baseline', color='tab:orange', linestyle='--', linewidth=2)
    ax4.plot(theoretical_norm_mask_mirrored[y_spot, :], label='New analytical version', color='tab:blue', linestyle='-.', linewidth=2)

    ax4.legend(fontsize=11, loc="upper right")
    ax4.grid(True, alpha=0.4)
    ax4.set_xlabel("X (pixels)", fontsize=14)
    ax4.set_ylabel("Normalized temperature", fontsize=14)
    ax4.tick_params(axis='both', labelsize=12, top=False, right=False)
    ax4.set_title(f"1D cross-section at Y = {y_spot}", fontsize=15)

    plt.tight_layout()
    plt.show()