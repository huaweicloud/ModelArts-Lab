import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    install_requires = []
    for line in f.read().splitlines():
        if line.strip() and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="x_base",
    version="3.0.6",
    description="x base.",
    packages=find_packages(),
    # 包含非 Python 文件（配置文件等）
    package_data={
        "x_base": [
            "config/*.yaml",  # 包含 config 目录下的所有 yaml 文件
        ]
    },
    # 确保包数据被包含
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
