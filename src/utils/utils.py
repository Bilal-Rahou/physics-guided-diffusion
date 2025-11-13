import numpy as np
import torch
import cv2
import math


def normalize_image(img):
    """Normalize an image to 0-255 uint8 range."""
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8


def preprocess_image(img_uint8):
    """Apply CLAHE + Gaussian blur preprocessing."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_uint8)
    clahe_blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    return clahe_blurred


def normalize_image_with_threshold(image, threshold):
    """Normalize an image between [threshold, max]."""
    image_clipped = torch.clamp(image, min=threshold, max=image.max())
    norm = (image_clipped - threshold) / (image_clipped.max() - threshold + 1e-8)
    return norm


def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=500):
    """Numerically integrate f(tau, x, y) using Gauss-Legendre quadrature."""
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n_points)
    nodes = torch.tensor(nodes, dtype=x.dtype, device=x.device)
    weights = torch.tensor(weights, dtype=x.dtype, device=x.device)
    tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
    weights = weights * (tau_max - tau_min) / 2
    integrand_values = f(tau_vals[:, None, None], x[:, None, :, :], y[:, None, :, :])
    integral = torch.tensordot(weights, integrand_values, dims=([0], [1]))
    return integral


def detect_lines_custom(img, scale, sigma_scale, refine):
    """Detect lines using OpenCV LSD with custom parameters."""
    lsd = cv2.createLineSegmentDetector(
        refine=refine,
        scale=scale,
        sigma_scale=sigma_scale,
        quant=2.0,
        ang_th=22.5,
        log_eps=0,
        density_th=0.7,
        n_bins=1024
    )
    lines, _, _, _ = lsd.detect(img)
    return lines


def calculate_angle(x1, y1, x2, y2):
    """Calculate the angle (in degrees) of a line segment."""
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 180


def filter_longest_horizontal_vertical(lines, min_length=15):
    """
    Select the longest horizontal and vertical lines from a set of lines.
    """
    longest_horizontal = None
    longest_vertical = None
    max_h_length = 0
    max_v_length = 0

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
    """
    Creates a mask for the vertical line extended infinitely.
    The mask has 1 on the side where the spot is located and 0 on the other.
    """
    dx = x2 - x1
    dy = y2 - y1

    # Line coefficients: Ax + By + C = 0
    A = dy
    B = -dx
    C = dx * y1 - dy * x1

    mask = np.zeros(image_shape, dtype=np.uint8)
    side_of_line = A * x_spot + B * y_spot + C

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            side = A * x + B * y + C
            if (side_of_line > 0 and side > 0) or (side_of_line < 0 and side < 0):
                mask[y, x] = 1
            else:
                mask[y, x] = 0
    return mask


def reflect_point_across_line(x1, y1, x2, y2, x0, y0):
    """
    Reflect a point (x0, y0) across a finite line segment (x1, y1)-(x2, y2).
    Returns (x_reflected, y_reflected) if projection lies within segment, else None.
    """
    dx = x2 - x1
    dy = y2 - y1
    norm_sq = dx * dx + dy * dy
    if norm_sq == 0:
        raise ValueError("The segment is just a point")

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
