from PIL import Image
from physics_ddpm import Unet, Trainer
import torch
import numpy as np
import random
from physics_ddpm import PhysicsInformedDiffusion

# Set a global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # If using GPUs
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
torch.backends.cudnn.benchmark = False     # May reduce performance but ensures reproducibility

# Path to dataset
image_folder = r"/d/brahou/data/flyd_frames_classification/data_flyd_frames_cropped/positive"
test_folder = r"/d/brahou/data/flyd_frames_classification/data_flyd_frames_cropped/positive"
# Define the U-Net model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,  # Assuming grayscale images
    flash_attn=False
)

# Define the physics-informed diffusion process
diffusion = PhysicsInformedDiffusion(
    model=model,
    image_size=120,
    timesteps=1000,            # Number of diffusion timesteps
    sampling_timesteps=250,    # Number of sampling timesteps for inference
    objective="pred_v",       # Predicting x0 (image start)
    c_data=1.0,                # Weight for the data-driven loss
    c_residual=10**-11,          # Weight for the residual loss
    P0=2,                      # Laser power (in watts)
    K=237,                     # Thermal conductivity (W/mK)
    D=1e-4,                    # Thermal diffusivity (m^2/s)
    V=-0.0005,                 # Laser speed (m/s)
    alpha=0.38,                # Efficiency
    pixel_size=0.0003,         # Pixel size (in meters)
    r_s=0.00075,                 # Gaussian radius (in meters)
    mode="uncracked"
)

# Define the trainer
trainer = Trainer(
    diffusion,
    image_folder,
    test_folder=test_folder,
    train_batch_size=8,
    train_lr=5e-4,
    train_num_steps=400000,         # Total training steps
    gradient_accumulate_every=2,  # Gradient accumulation steps
    ema_decay=0.995,              # Exponential moving average decay
    amp=False,                     # Turn on mixed precision
    save_and_sample_every=5000,
    num_fid_samples=187,
    results_folder='./physics_v_10-11_cracks_10%',
    calculate_fid=True            # Whether to calculate FID during training
)

# Start training
trainer.train()
