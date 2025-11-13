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
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import normalize_image
from src.utils.physics_metrics import residual_mae

folder_path = r"/d/brahou/data/flyd_frames_classification/data_flyd_frames_cropped/negative"
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


# Load and prepare images
images = load_images_from_folder(folder_path)
batch_tensors = []
for _, img in images:
    img_uint8 = normalize_image(img)
    img_tensor = torch.tensor(img_uint8, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    batch_tensors.append(img_tensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = torch.cat(batch_tensors).to(device)  # shape [10, 1, H, W]

# Compute physics-based metric
residual_error = residual_mae(
    batch_images=batch,
    alpha=0.38,
    P0=2,
    K=237,
    D=1e-4,
    V=-0.0005,
    pixel_size=0.0003,
    r_s=0.00075,
    depth=0.002,
    mode="uncracked",  # or "cracked"
    batch_size=8
)
