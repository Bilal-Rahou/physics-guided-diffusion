# 🧠 Physics-Informed Diffusion Model (DDPM)

This repository implements a **Physics-Informed Denoising Diffusion Probabilistic Model (DDPM)** designed to integrate *analytical thermal physics* into the diffusion process.  
It combines deep generative learning with thermodynamic consistency for **thermal field reconstruction and defect detection**.

---

## 🚀 Features

- **Physics-Guided Sampling:** Integrates analytical solutions (Salazar / Krapez models)
- **Cracked & Crack-Free Sample Modes**
- **Residual Physics Loss:** Quantifies deviation from physical laws
- **Batch Analytical Evaluation** for large-scale experiments
- **Torch-based Analytical Integrators** using Gauss-Legendre quadrature
- **Modular Utility Design** (`src/utils/`)

---

## 🧩 Repository Structure

| Folder | Description |
|---------|--------------|
| `src/` | Main model and training scripts |
| `src/utils/` | Utility functions for image processing, residuals, and metrics |
| `notebooks/` | Example notebooks for demo and visualization |
| `results/` | Model outputs, logs, and saved checkpoints |
| `tests/` | Unit tests for reproducibility |
| `requirements.txt` | Required dependencies |

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/physics-informed-ddpm.git
cd physics-informed-ddpm
pip install -e .