from setuptools import find_packages, setup

setup(
    name="civtensor",
    version="1.0.0",
    author="BIGAI-MAS",
    description="Freeciv Tensor Baseline",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard",
        "setproctitle",
        "tensorboardX@git+https://github.com/lanpa/tensorboardX.git@master",
    ],
    setup_requires=[
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
