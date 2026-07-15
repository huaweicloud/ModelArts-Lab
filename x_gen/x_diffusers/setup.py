import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    install_requires = []
    for line in f.read().splitlines():
        if line.strip() and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="x_diffusers",
    version="3.0.6",
    description="x diffusers",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
