from setuptools import setup, find_packages
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='localmd',
    description="Method for compressing neuroimaging data using spatially localized low-rank matrix decompositions",
    author='Amol Pasarkar',
    version="0.0.4",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "tifffile",
        "torch",
        "scipy",
        "tqdm",
        "jax",
        "jaxlib",
        "plotly"
    ],
    extras_require={
        'vis': [
            'jupyterlab',  # Install JupyterLab when 'notebooks' is specified
            'plotly'  # Plotly is already in the main install_requires, but we can keep it here too
        ],
    },
    python_requires='>=3.8',
)