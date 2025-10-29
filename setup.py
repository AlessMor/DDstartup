from setuptools import setup, find_packages

setup(
    name="ddstartup",
    version="0.1.0",
    description="DD Startup: Fusion reactor tritium breeding and startup time analysis",
    author="Alessandro Morandi, Samuele Meschini, Gabriele Iob",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "h5py>=3.0.0",
        "numba>=0.54.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "SALib>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ddstartup=ddstartup.main:main",
            "ddstartup-postprocess=ddstartup.postprocessing.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Nuclear Fusion",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
