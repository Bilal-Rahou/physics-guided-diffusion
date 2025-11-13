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

# Constants
P0 = 2  # Laser power (in watts)
K = 237  # Thermal conductivity (W/mK)
D = 1e-4  # Thermal diffusivity (m^2/s)
V = -0.0005  # Laser speed (m/s)
alpha = 0.38  # Efficiency
pixel_size = 0.0003  # Pixel size (in meters)
radius = 0.0015 # Gaussian radius (in meters at 1/e²)
depth = 0.002
epsilon = K / np.sqrt(D)

# Convert pixel position to meters
def pixel_to_meter(x, y, pixel_size_m=pixel_size):
    return x * pixel_size_m, y * pixel_size_m

# Salazar integrand function
def salazar_integrand(tau, x, y):
    r_s = radius
    exp_term = -2 * ((x) ** 2 + (y - V * tau) ** 2) / (r_s**2 + 8 * D * (-tau))
    return 1 / np.sqrt(-tau) * np.exp(exp_term) / (r_s**2 + 8 * D * (-tau))

# Krapez integrand function
def krapez_integrand(Fo, x_star, y_star):
    r_s = radius / np.sqrt(2)
    Pe = (V * r_s) / D 
    Fo_safe = np.where(Fo <= 0, 1e-12, Fo)
    A = 1 + 8 * Fo_safe
    exp_term = -2 * ((x_star) ** 2 + (y_star + Pe * Fo_safe) ** 2) / A
    return np.exp(exp_term) / (A * np.sqrt(Fo_safe))

def green_depth(Fo_e, nmax=200):
    Fo_e_safe = np.where(Fo_e <= 0, 1e-12, Fo_e)
    n = np.arange(1, nmax + 1, dtype=float)
    exp_term = np.exp(- (n[:, None, None, None]**2) / Fo_e_safe[None, :, :, :])
    s = exp_term.sum(axis=0)
    g1 = 1.0 + 2.0 * s
    return g1

def green_depth_torch(Fo_e, nmax=200):
    device, dtype = Fo_e.device, Fo_e.dtype
    Fo_e_safe = torch.where(Fo_e <= 0, torch.tensor(1e-12, dtype=dtype, device=device), Fo_e)
    n = torch.arange(1, nmax + 1, dtype=dtype, device=device).view(nmax, 1, 1, 1)
    exp_term = torch.exp(- (n ** 2) / Fo_e_safe.unsqueeze(0))  # [nmax, B, H, W]
    s = exp_term.sum(dim=0)
    g1 = 1.0 + 2.0 * s
    return g1

def krapez_integrand_finite_depth(Fo, x_star, y_star, nmax=200):
    r_s = radius / np.sqrt(2)
    Pe = (V * r_s) / D  # Péclet number
    depth_star = depth / r_s
    Fo_safe = np.where(Fo <= 0, 1e-12, Fo)
    A = 1.0 + 8.0 * Fo_safe
    exp_term = -2.0 * ( (x_star**2) + (y_star + Pe * Fo_safe)**2 ) / A
    Fo_e = Fo_safe / (depth_star**2)
    g1 = green_depth(Fo_e, nmax=nmax)
    integrand = (g1 * np.exp(exp_term)) / (A * np.sqrt(Fo_safe))
    integrand = np.where(Fo > 0, integrand, 0.0)
    return integrand

def krapez_integrand_finite_depth_torch(Fo, x_star, y_star):
    device, dtype = Fo.device, Fo.dtype
    r_s_intern = radius / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
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
    
# Gauss-Legendre integration
def gauss_legendre_integral(f, x, y, tau_min, tau_max, n_points=1000):
    nodes, weights = roots_legendre(n_points)
    tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
    weights = weights * (tau_max - tau_min) / 2
    integrand_values = f(tau_vals[:, None, None], x[None, :, :], y[None, :, :])
    integral = np.tensordot(weights, integrand_values, axes=(0, 0))
    return integral

def gauss_legendre_integral_torch(f, x, y, tau_min, tau_max, n_points=1000):
    """
    Compute Gauss-Legendre integral in torch for vectorized x,y grids.
    f must accept (tau, x, y) tensors and return matching shape.
    """
    # Use numpy to get nodes & weights once (they're just constants)
    nodes_np, weights_np = roots_legendre(n_points)
    device, dtype = x.device, x.dtype

    # Convert nodes & weights to torch
    nodes = torch.tensor(nodes_np, dtype=dtype, device=device)
    weights = torch.tensor(weights_np, dtype=dtype, device=device)

    # Map nodes from [-1, 1] to [tau_min, tau_max]
    tau_vals = tau_min + (nodes + 1) * (tau_max - tau_min) / 2
    weights = weights * (tau_max - tau_min) / 2

    # Evaluate integrand
    # f must broadcast: f(tau_vals[:,None,None], x[None,:,:], y[None,:,:])
    integrand_values = f(tau_vals[:, None, None], x[None, :, :], y[None, :, :])

    # Integrate along tau dimension
    integral = torch.tensordot(weights, integrand_values, dims=([0], [0]))

    return integral
# Temperature field calculation
def calculate_temperature_field(image_shape, center, model="salazar"):
    r_s = radius / np.sqrt(2)
    y_c, x_c = center
    height, width = image_shape
    x = np.arange(0, width) - x_c
    y = np.arange(0, height) - y_c
    x_meters, y_meters = pixel_to_meter(x, y)
    X, Y = np.meshgrid(x_meters, y_meters)

    if model == "salazar":
        T_star = gauss_legendre_integral(salazar_integrand, X, Y, tau_min=-500, tau_max=0, n_points=1000)
        T = T_star * (2 * alpha * P0) / (epsilon * np.sqrt(np.pi**3))
    elif model == "krapez":
        x_star = X / r_s
        y_star = Y / r_s
        T_star = gauss_legendre_integral(krapez_integrand, x_star, y_star, tau_min=0, tau_max=10000, n_points=1000)
        T = (2 * alpha * P0) / (K * np.sqrt(np.pi) * np.pi * r_s) * T_star
    elif model == "krapez_depth":
        x_star = X / r_s
        y_star = Y / r_s
        T_star = gauss_legendre_integral(krapez_integrand_finite_depth, x_star, y_star, tau_min=0, tau_max=10000, n_points=1000)
        T = (2 * alpha * P0) / (K * np.sqrt(np.pi) * np.pi * r_s) * T_star
    else:
        raise ValueError("Unknown model. Choose 'salazar' or 'krapez'.")
    
    return T

def calculate_temperature_field_torch(image_shape, center, model="salazar", device="cuda"):
    """
    Compute theoretical temperature field using the chosen model ('salazar', 'krapez', or 'krapez_depth').
    Fully GPU-compatible torch implementation.
    """
    # --- parameters and center ---
    r_s_torch = radius / torch.sqrt(torch.tensor(2.0, device=device))
    y_c, x_c = center
    height, width = image_shape

    # --- create spatial grid in meters ---
    x = torch.arange(0, width, device=device, dtype=torch.float32) - x_c
    y = torch.arange(0, height, device=device, dtype=torch.float32) - y_c
    x_meters, y_meters = pixel_to_meter(x, y)  # must return torch tensors
    X, Y = torch.meshgrid(y_meters, x_meters, indexing="ij")

    if model == "krapez_depth":
        x_star = X / r_s_torch
        y_star = Y / r_s_torch
        T_star = gauss_legendre_integral_torch(
            lambda Fo, x, y: krapez_integrand_finite_depth_torch(Fo, x, y),
            x_star, y_star, tau_min=0, tau_max=10000, n_points=1000
        )
        pi = torch.tensor(torch.pi, device=device)
        const = (2 * alpha * P0) / (K * pi**1.5 * r_s_torch)
        T = T_star * const

    else:
        raise ValueError("Unknown model. Choose 'salazar', 'krapez', or 'krapez_depth'.")

    return T
# Normalize image with threshold
def normalize_image_with_threshold(image, threshold):
    image_clipped = np.clip(image, threshold, np.max(image))
    normalized_image = rescale_intensity(image_clipped, in_range=(threshold, np.max(image)), out_range=(0, 1))
    return normalized_image

# Load experimental image and find center
def load_and_find_center(image):
    center = np.unravel_index(np.argmax(image), image.shape)[::-1]
    return image, center

def calculate_threshold(image, freq_threshold=20, intensity_threshold=30):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    for intensity, frequency in enumerate(histogram):
        if frequency > freq_threshold and intensity > intensity_threshold:
            return intensity  # Return the first intensity that meets the criteria
    return None

# Compare experimental and theoretical images
def compare_images(experimental_image, theoretical_image, otsu):
    # threshold = np.percentile(experimental_image, threshold_percentile)
    # threshold = calculate_threshold(experimental_image)
    # Propose default values for low and high thresholds
    # low_threshold = 0.1  # Low threshold for edge detection (normalized intensity scale: 0-1)
    # high_threshold = 0.3  # High threshold for edge detection (normalized intensity scale: 0-1)
    # experimental_image_normalized = (experimental_image - experimental_image.min()) / (experimental_image.max() - experimental_image.min())
    # edges = canny(experimental_image_normalized, low_threshold=low_threshold, high_threshold=high_threshold)
    # mask = edges > 0
    # threshold = experimental_image[mask].min() if mask.any() else experimental_image.min()
    if otsu:
        threshold = threshold_otsu(experimental_image)
        mask = experimental_image >= threshold
        experimental_norm = normalize_image_with_threshold(experimental_image, threshold)
    else:
        experimental_norm = rescale_intensity(experimental_image, out_range=(0, 1))

    theoretical_norm = rescale_intensity(theoretical_image, out_range=(0, 1))
    difference = np.abs(experimental_norm - theoretical_norm)

    # plt.hist(experimental_image.ravel(), bins=256, range=(0, 1))  # For each image
    # plt.title("Pixel Intensity Distribution")
    # plt.xlabel("Pixel Value")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(experimental_norm, cmap='gray')
    # plt.title("Experimental Image")
    # plt.colorbar(label="Pixel Intensity")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.subplot(1, 2, 2)
    # plt.imshow(experimental_image, cmap='gray')
    # plt.imshow(mask, cmap='Reds', alpha=0.3)  # Overlay the mask in red with transparency
    # plt.title("Experimental Image with Mask")
    # plt.colorbar(label="Pixel Intensity")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.tight_layout()
    # plt.show()
    if otsu:
        difference[~mask] = 0  # Set differences below the threshold to zero
        pixel_count = np.sum(mask)  # Number of pixels above the threshold
        mean_filtered_difference = np.sum(difference) / pixel_count
        return mean_filtered_difference
    return difference.mean()

# Process all images in a folder
import numpy as np
import matplotlib.pyplot as plt

def process_all_images(images, otsu=True):
    mean_absolute_differences = []
    first_five_residuals = []
    exp_images = []
    the_images = []
    
    first_five_profiles = []
    first_five_exp_images = []
    first_five_the_images = []

    for idx, image in enumerate(images[5:12]):
        experimental_image, center = load_and_find_center(image)
        threshold = threshold_otsu(experimental_image)
        threshold = 75
        image_shape = experimental_image.shape
        theoretical_field = calculate_temperature_field_torch(image_shape, (center[1], center[0]), model="krapez_depth")

        # mean_diff = compare_images(experimental_image, theoretical_field, otsu)
        # mean_absolute_differences.append(mean_diff)

        experimental_norm = normalize_image_with_threshold(experimental_image, threshold)
        theoretical_norm = rescale_intensity(theoretical_field.cpu().numpy(), out_range=(0, 1))

        # Extract the intensity profile at the spot level (center row)
        y_center = int(center[1])
        exp_profile = experimental_norm[y_center, :]
        theo_profile = theoretical_norm[y_center, :]

        if idx < 5:
            first_five_profiles.append((exp_profile, theo_profile))
            first_five_exp_images.append(experimental_norm)  # Store experimental image
            first_five_the_images.append(theoretical_norm)  # Store experimental image

        theoretical_field_krapez = calculate_temperature_field(image_shape, (center[1], center[0]), model="krapez")
        theoretical_norm_krapez = rescale_intensity(theoretical_field_krapez, out_range=(0, 1))

    plot_comparison_with_difference(theoretical_norm_krapez, theoretical_norm)
    # # Plot the profiles
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i, (exp_profile, theo_profile) in enumerate(first_five_profiles[:3]):
    #     axes[i].plot(exp_profile, label="Generated image", color="red")
    #     axes[i].plot(theo_profile, label="Analytical solution", color="blue", linestyle="dashed")
    #     axes[i].set_title(f"Profile {i+1}")
    #     axes[i].set_xlabel("Pixel Position")
    #     axes[i].set_ylabel("Normalized Temperature")
    #     axes[i].legend(loc="upper right")  # Legend positioned at the top right
    # plt.tight_layout()
    # plt.show()

    # # Plot the experimental images
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i, exp_img in enumerate(first_five_exp_images[:3]):
    #     axes[i].imshow(exp_img, cmap='hot')
    #     axes[i].axis("off")
    #     axes[i].set_title(f"Experimental Image")
    # plt.tight_layout()
    # plt.show()

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i, exp_img in enumerate(first_five_the_images[:3]):
    #     axes[i].imshow(exp_img, cmap='hot')
    #     axes[i].axis("off")
    #     axes[i].set_title(f"Analytical solution")
    # plt.tight_layout()
    # plt.show()
    return np.mean(mean_absolute_differences)  # Compute overall mean absolute difference




# Plot the first experimental and theoretical images
# Plot the first experimental and theoretical images along with their absolute difference
import matplotlib.gridspec as gridspec

def plot_comparison_with_difference(experimental_norm, theoretical_norm, save_path="comparison_with_difference.svg"):
    difference = np.abs(experimental_norm - theoretical_norm)

    # Define axis labels in millimeters
    height, width = experimental_norm.shape
    x_mm = np.arange(width) * pixel_size * 1000  # Convert to millimeters
    y_mm = np.arange(height) * pixel_size * 1000  # Convert to millimeters

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the experimental image
    im0 = axes[0].imshow(experimental_norm, cmap="hot", origin="lower", extent=[0, x_mm[-1], 0, y_mm[-1]])
    axes[0].set_title("Solution analytique")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    fig.colorbar(im0, ax=axes[0], orientation="vertical")

    # Plot the theoretical image
    im1 = axes[1].imshow(theoretical_norm, cmap="hot", origin="lower", extent=[0, x_mm[-1], 0, y_mm[-1]])
    axes[1].set_title("Solution analytique prenant en compte la profondeur")
    axes[1].set_xlabel("X (mm)")
    axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove Y-axis ticks
    fig.colorbar(im1, ax=axes[1], orientation="vertical")

    # Plot the absolute difference with threshold applied
    im2 = axes[2].imshow(difference, cmap="hot", origin="lower", extent=[0, x_mm[-1], 0, y_mm[-1]],vmax=1)
    axes[2].set_title("Différence absolue (seuil appliqué)")
    axes[2].set_xlabel("X (mm)")
    axes[2].tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove Y-axis ticks
    fig.colorbar(im2, ax=axes[2], orientation="vertical")

    # Adjust layout and save the plot
    plt.tight_layout()
    # plt.savefig(save_path, format="svg", bbox_inches="tight", facecolor="none")
    # print(f"Comparison with difference plot (threshold applied) saved as {save_path}")
    plt.show()

def plot_cross_section_temperatures(experimental_norm, theoretical_norm, save_path="cross_section_temperatures.svg"):
    """
    Plot 1D cross-section along the X-axis through the laser spot center
    comparing experimental and theoretical normalized temperature fields.
    """
    # --- locate spot center ---
    height, width = experimental_norm.shape
    y_spot, x_spot = np.unravel_index(np.argmax(experimental_norm), experimental_norm.shape)

    # --- define X axis in millimeters (relative to center) ---
    x_mm = (np.arange(width) - x_spot) * pixel_size * 1000  # convert to mm

    # --- extract profiles at the same Y position (spot center row) ---
    exp_profile = experimental_norm[y_spot, :]
    theo_profile = theoretical_norm[y_spot, :]

    # --- plot setup ---
    plt.figure(figsize=(7, 5))
    plt.plot(x_mm, exp_profile, label="Solution analytique", color="red", linewidth=2)
    plt.plot(x_mm, theo_profile, label="Solution analytique avec profondeur", color="blue", linestyle="--", linewidth=2)

    plt.xlabel("Position X (mm)", fontsize=13)
    plt.ylabel("Température normalisée", fontsize=13)
    plt.title("Comparaison des champs de température (coupe X au centre du spot)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    # --- optional save ---
    # plt.savefig(save_path, format="svg", bbox_inches="tight", facecolor="none")
    plt.show()


# # Main execution
# folder_path = r"/d/brahou/DDPM/denoising-diffusion-pytorch/negative_cropped"  # Your dataset folder path
folder_path = r"/d/brahou/data/flyd_frames_classification/data_flyd_frames_cropped/negative"
# folder_path = r"/d/brahou/data/pinn_dataset/cropped_dataset_wrapper/cropped_dataset"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-5_dataset_45000"  # Your dataset folder path
# folder_path = r"synthesis_noise_dataset_cropped_180000"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_0.0_dataset_45000"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-8_dataset_7"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-9_dataset_11"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-10_dataset_8"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-12_dataset_6"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-11_dataset_6"  # Your dataset folder path

# folder_path = r"synthesis_v_1.0_10-8_dataset_29"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-9_dataset_30"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-10_dataset_20"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-11_dataset_31"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-12_dataset_32"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-13_dataset_22"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_0.0_dataset_20"  # Your dataset folder path

# folder_path = r"synthesis_v_1.0_0.0_no_crop_dataset_27"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_0.0_nocrop_dataset_40"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_10-10_nocrop_dataset_15"  # Your dataset folder path
# folder_path = r"synthesis_v_1.0_0.0_cropped_dataset_40"

# folder_path = r"synthesis_v_1.0_10-10_otsu_0.00075_dataset44"
# folder_path = r"synthesis_v_1.0_10-10_otsu_0.00075_dataset15"

# folder_path = r"synthesis_v_1.0_10-8_otsu_0.00075_dataset46"
# folder_path = r"synthesis_v_1.0_10-8_otsu_0.00075_dataset34"
# folder_path = r"synthesis_v_1.0_10-8_otsu_0.00075_dataset48"

# folder_path = r"synthesis_v_1.0_10-7_otsu_0.00075_dataset50"

# folder_path = r"synthesis_v_1.0_10-9_otsu_0.00075_dataset46"
# folder_path = r"synthesis_v_1.0_10-9_otsu_0.00075_dataset42"
# folder_path = r"synthesis_v_1.0_10-9_otsu_0.00075_dataset43"
# folder_path = r"synthesis_v_1.0_10-9_otsu_0.00075_dataset49"


# folder_path = r"synthesis_v_1.0_0.0_otsu_0.00075_dataset35"


def load_images_from_folder(folder_path):
    generated_images = []
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".tif", ".jpeg")):
                image_files.append(os.path.join(root, file))
    print(f"Total images found: {len(image_files)}")
    for image_path in image_files:
        image = io.imread(image_path)
        if len(image.shape) == 3:  # Convert RGB to grayscale if necessary
            image = rgb2gray(image)
        generated_images.append(image)
    return generated_images


generated_images = load_images_from_folder(folder_path)
process_all_images(generated_images, otsu=True)