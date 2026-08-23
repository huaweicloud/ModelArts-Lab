import os

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setup(
    name="ascend_vllm",
    version="6.5.935",
    packages=find_packages(include=("ascend_vllm", "ascend_vllm.*")),
    package_data={"ascend_vllm.middleware": ["validator_config.json"]},
    entry_points={
        "vllm.platform_plugins": ["ascend_vllm = ascend_vllm:register"],
        "vllm.general_plugins": [
            "modelarts_ascend_kv_connector = ascend_vllm:register_connector",
            "ascend_model_loader = ascend_vllm:register_model_loader",
            "ascend_service_profiling = ascend_vllm:register_service_profiling",
            "ascend_model = ascend_vllm:register_model",
            "modelarts_general_plugin_patch = ascend_vllm:register_general_plugin_patch",
        ],
    },
    install_requires=get_requirements(),
)
