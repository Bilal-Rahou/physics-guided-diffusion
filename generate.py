import torch
from physics_ddpm import Unet, PhysicsInformedDiffusion
from torchvision import utils
import math
import os

# Reproducibility (same as before)
import numpy as np, random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def strip_prefix_from_state_dict(state_dict, prefix='online_model.'):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # Remove the prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

# Output path
results_folder = './generation_examples_ema'
os.makedirs(results_folder, exist_ok=True)

# Rebuild UNet model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    flash_attn=False
)

# Wrap into diffusion
diffusion = PhysicsInformedDiffusion(
    model=model,
    image_size=120,
    timesteps=1000,
    sampling_timesteps=5,
    objective="pred_v",
    c_data=1.0,
    c_residual=1e-10,
    P0=2,
    K=237,
    D=1e-4,
    V=-0.0005,
    alpha=0.38,
    pixel_size=0.0003,
    r_s=0.00065
)

checkpoint_path = 'physics_v_10-11_cracks_10%/model-1.pt'  # Path to your checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

# If using EMA weights (as is common in DDPM training), use 'ema'
# diffusion.load_state_dict(checkpoint['model'])
# diffusion.load_state_dict(checkpoint["ema"])
# diffusion.model.eval()

ema_state_dict = checkpoint.get('ema', None)
if ema_state_dict is not None:
    ema_model_state_dict = {
        k.replace("ema_model.model.", ""): v
        for k, v in ema_state_dict.items()
        if k.startswith("ema_model.model.")
    }
    diffusion.model.load_state_dict(ema_model_state_dict)
    diffusion.model.eval()
    print("✅ EMA weights loaded")
else:
    print("⚠️ No EMA weights found in checkpoint")

n_samples = 64
batch_size = 64  # You can adjust this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diffusion = diffusion.to(device)
model = model.to(device)

# Helper to split into batches
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    return [divisor] * groups + ([remainder] if remainder else [])

# Generate images
batches = num_to_groups(n_samples, batch_size)
all_images = []

with torch.inference_mode():
    for n in batches:
        samples = diffusion.sample(batch_size=n)
        all_images.append(samples)

all_images = torch.cat(all_images, dim=0)

# Create output folder
checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path))
sample_folder = os.path.join(results_folder, checkpoint_name)
os.makedirs(sample_folder, exist_ok=True)

# Save each image as an individual PNG
for i, img in enumerate(all_images):
    path = os.path.join(sample_folder, f'image_{i:03d}.png')  # zero-padded filenames
    utils.save_image(img, path)

print(f"Saved {len(all_images)} individual PNG images to: {sample_folder}")
