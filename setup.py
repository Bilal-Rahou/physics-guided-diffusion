from setuptools import setup, find_packages

setup(
    name="physics-guided-diffusion",
    version="0.1.0",
    author="Bilal Rahou",
    description="Physics-Informed Diffusion Model for Image synthesis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pillow",
        "ema-pytorch",
        "accelerate",
        "einops",
        "opencv-python",
        "scikit-image",
        "pytorch-fid",
    ],
)
