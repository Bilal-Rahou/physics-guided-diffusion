import torch
import numpy as np
from skimage.filters import threshold_otsu
from .residual import analytical_solution_cracked_sample, analytical_solution_uncracked_sample
from .utils import normalize_image_with_threshold

def residual_mae(
    batch_images,
    alpha,
    P0,
    K,
    D,
    V,
    pixel_size,
    r_s,
    depth,
    mode,
    batch_size=8
):
    """
    Compute mean absolute difference (physics metric) between experimental and
    analytical normalized temperature fields, processing in sub-batches.

    Args:
        batch_images (torch.Tensor): [N, 1, H, W] all generated images
        alpha, P0, K, D, V, pixel_size, r_s, depth: physical parameters
        mode (str): "cracked" or "uncracked"
        batch_size (int): size of each sub-batch for processing

    Returns:
        float: overall mean absolute difference (physics metric)
    """

    # --- Split into sub-batches ---
    N = batch_images.shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    mean_absolute_differences = []

    for b in range(num_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, N)
        sub_batch = batch_images[start:end]

        # --- Compute analytical solution for this sub-batch ---
        if mode == "cracked":
            theoretical_norm_mask_batch, mask_h_batch = analytical_solution_cracked_sample(
                sub_batch,
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
                sub_batch,
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
                f"Invalid mode '{mode}'. Please choose either 'cracked' or 'uncracked'."
            )

        # --- Compute mean absolute error for each image in this sub-batch ---
        for i in range(sub_batch.shape[0]):
            image = sub_batch[i, 0]
            theoretical_norm = theoretical_norm_mask_batch[i]

            # Get corresponding horizontal mask (if available)
            mask_h = None
            if i < len(mask_h_batch) and mask_h_batch[i] is not None:
                mask_h = mask_h_batch[i].to(image.device).squeeze()
            else:
                # if no horizontal mask, default to full image mask (all ones)
                mask_h = torch.ones_like(image, device=image.device)

            # --- Normalize experimental image using Otsu threshold ---
            otsu_threshold = threshold_otsu(image.detach().cpu().numpy())
            otsu_threshold = torch.tensor(otsu_threshold, dtype=image.dtype, device=image.device)
            experimental_norm = normalize_image_with_threshold(image, otsu_threshold)

            # --- Build combined mask ---
            mask_otsu = (image >= otsu_threshold).float()
            combined_mask = mask_otsu * mask_h  # intersection of Otsu and horizontal mask

            # --- Compute masked absolute difference ---
            difference = torch.abs(experimental_norm - theoretical_norm)
            difference *= combined_mask  # apply both masks

            # --- Compute mean difference only in the valid region ---
            pixel_count = torch.sum(combined_mask)
            mean_diff = torch.sum(difference) / (pixel_count + 1e-8)
            mean_absolute_differences.append(mean_diff.item())

    # --- Combine results from all sub-batches ---
    overall_mean_abs_diff = float(np.mean(mean_absolute_differences))
    print(f"Overall mean absolute physics difference ({mode}): {overall_mean_abs_diff:.6f}")
    return overall_mean_abs_diff