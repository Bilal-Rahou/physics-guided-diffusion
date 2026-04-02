# Physics-Guided Diffusion for Thermal Field Reconstruction and Defect Detection

This repository contains a PyTorch implementation of our Physics-Informed Denoising Diffusion Probabilistic Model (DDPM). The framework is designed to combine deep generative learning with physical laws by integrating analytical thermal physics directly into the diffusion process. It is primarily developed for active thermography and Non-Destructive Testing (NDT), specifically focusing on the generation and evaluation of thermal fields in cracked and crack-free specimens.

## Overview

While standard diffusion models rely purely on data distributions, this framework introduces a physics-guided regularization term. By penalizing deviations from established thermal analytical solutions, the model ensures that the generated synthetic data remains physically consistent with real-world heat diffusion phenomena.

### Key features

* **Physics-Informed Regularization:** Computes a residual physics loss to quantify and minimize the deviation between the generated thermal fields and theoretical models.
* **Analytical Integration:** Implements analytical integrators utilizing Gauss-Legendre quadrature with batched processing
* **Dual-Mode Generation:** Supports distinct training pipelines for both crack-free baseline samples and cracked samples.

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `src/` | Core model architectures, diffusion processes, and main training scripts. |
| `src/utils/` | Utility modules for image preprocessing, residual calculations, and performance metrics (FID, F1, mAP). |
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
```

## Experimental sample specificity

This model utilizes experimental thermal sequences acquired via Flying Spot Thermography (FST). The base data is derived from the public FLYD dataset available at https://github.com/kevinhelvig/FLYD

Important Note on Residual Computation: The physics residual computation currently implemented in this repository is specifically tailored to the experimental conditions, geometry, and material properties of the FLYD dataset. If you intend to train this framework on different samples or alternative experimental setups, the residual computation logic (located in the source files) must be adapted to reflect your specific physical configurations.
