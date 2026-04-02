# Physics-Guided Diffusion for Thermal Field Reconstruction and Defect Detection

This repository contains the official PyTorch implementation of our Physics-Informed Denoising Diffusion Probabilistic Model (DDPM). The framework is designed to bridge deep generative learning with physical laws by integrating analytical thermal physics directly into the diffusion process. It is primarily developed for active thermography and Non-Destructive Testing (NDT), specifically focusing on the generation and evaluation of thermal fields in cracked and crack-free specimens.

## Overview

While standard diffusion models rely purely on data distributions, this framework introduces a physics-guided regularization term. By penalizing deviations from established thermodynamic analytical solutions (such as the Salazar and Krapez formulations), the model ensures that the generated synthetic data remains physically consistent with real-world heat diffusion phenomena.

### Key Capabilities

* **Physics-Informed Regularization:** Computes a residual physics loss to quantify and minimize the deviation between the generated thermal fields and theoretical models.
* **Analytical Integration:** Implements GPU-accelerated Torch-based analytical integrators utilizing Gauss-Legendre quadrature for highly efficient batch evaluation.
* **Custom Noise Scheduling:** Utilizes a sigmoid noise schedule optimized for 120x120 spatial resolutions to ensure stable signal-to-noise progression during training.
* **Dual-Mode Generation:** Supports distinct training pipelines for both crack-free baseline samples and cracked samples.
* **Downstream Evaluation:** Includes integration with localization frameworks (e.g., YOLOv9) to validate the synthetic data's utility in downstream defect detection tasks.

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `src/` | Core model architectures, diffusion processes, and main training scripts. |
| `src/utils/` | Utility modules for image preprocessing, residual calculations, and performance metrics (FID, F1, mAP). |
| `notebooks/` | Jupyter notebooks containing inference demonstrations and visualization tools for generated samples. |
| `results/` | Default output directory for generated images, training logs, and model checkpoints. |
| `tests/` | Unit testing suite to ensure mathematical and structural integrity. |

## Installation

It is recommended to set up an isolated Python environment before installation.

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/physics-guided-diffusion.git
cd physics-guided-diffusion

# Install required dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
