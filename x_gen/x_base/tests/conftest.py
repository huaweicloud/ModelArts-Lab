"""
Pytest configuration and fixtures for x_base unit tests.
"""

# 添加项目根目录到路径
import pathlib
import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

_root_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_root_dir))


# ============================================================
# Pytest configuration
# ============================================================
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_torch: mark test as requiring torch")
    config.addinivalue_line("markers", "requires_diffusers: mark test as requiring diffusers")
    config.addinivalue_line("markers", "integration: mark test as integration test")


# ============================================================
# Helper functions
# ============================================================
def _create_mock_diffusion_pipeline_class():
    """Create a mock class that inherits from DiffusionPipeline."""
    try:
        from diffusers import DiffusionPipeline

        class MockDiffusionPipeline(DiffusionPipeline):
            def __init__(self, *args, **kwargs):
                pass

        return MockDiffusionPipeline
    except ImportError:
        return MagicMock


def create_mock_args(**overrides):
    """Create mock args with default values, allowing overrides."""
    defaults = {
        "model": "Wan2.1-T2V-1.3B",
        "task_type": "t2v",
        "pretrained_model_name_or_path": "Wan2.1-T2V-1.3B",
        "width": 832,
        "height": 480,
        "frames": 121,
        "save_fps": 16,
        "save_path": "./output.mp4",
        "ada_brighten": False,
        "frame_interpolation": False,
        "frame_model_path": "",
        "ten_second": False,
        "i2v_image_path": "",
        "num_inference_steps": 50,
        "turbo_mode": "faiz",
        "x": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def create_mock_pipe(pipeline_name="WanPipeline"):
    """Create a mock DiffusionPipeline with specified name."""
    MockPipelineClass = _create_mock_diffusion_pipeline_class()

    pipe = MockPipelineClass()
    pipe.__class__.__name__ = pipeline_name
    pipe.transformer = MagicMock()
    pipe.transformer.__class__ = MagicMock()
    pipe.transformer.config = MagicMock()
    pipe.transformer.config.patch_size = (1, 2, 2)
    pipe.scheduler = MagicMock()
    pipe.scheduler.timesteps = MagicMock()
    pipe.scheduler.config = MagicMock()
    pipe.scheduler.config.num_train_timesteps = 1000
    pipe.scheduler.set_timesteps = MagicMock()
    return pipe


# ============================================================
# Fixtures - Default model configurations
# ============================================================
@pytest.fixture
def mock_args():
    """Mock args namespace for Wan2.1-T2V-1.3B model."""
    return create_mock_args()


@pytest.fixture
def mock_args_wan14b():
    """Mock args for Wan2.1-T2V-14B model."""
    return create_mock_args(
        model="Wan2.1-T2V-14B",
        pretrained_model_name_or_path="Wan2.1-T2V-14B",
        turbo_mode="next_faiz",
    )


@pytest.fixture
def mock_args_wan22():
    """Mock args for Wan2.2 model."""
    return create_mock_args(
        model="Wan2.2-T2V-A14B",
        pretrained_model_name_or_path="Wan2.2-T2V-A14B",
        frames=81,
        turbo_mode="next_faiz",
    )


@pytest.fixture
def mock_args_i2v_480p():
    """Mock args for I2V 480p model."""
    return create_mock_args(
        model="Wan2.1-I2V-14B",
        task_type="i2v",
        pretrained_model_name_or_path="path/to/480p/model",
        i2v_image_path="input.jpg",
    )


@pytest.fixture
def mock_args_i2v_720p():
    """Mock args for I2V 720p model."""
    return create_mock_args(
        model="Wan2.1-I2V-14B",
        task_type="i2v",
        pretrained_model_name_or_path="path/to/720p/model",
        i2v_image_path="input.jpg",
    )


@pytest.fixture
def mock_args_hunyuan():
    """Mock args for HunyuanVideo model."""
    return create_mock_args(
        model="HunyuanVideo-T2V-13B",
        pretrained_model_name_or_path="HunyuanVideo-T2V-13B",
        frames=97,  # Must satisfy 4*k+1
        turbo_mode="faiz",
    )


# ============================================================
# Fixtures - Pipeline mocks
# ============================================================
@pytest.fixture
def mock_pipe():
    """Mock DiffusionPipeline for testing."""
    return create_mock_pipe()


@pytest.fixture
def mock_pipe_wan():
    """Mock WanPipeline."""
    return create_mock_pipe("WanPipeline")


@pytest.fixture
def mock_pipe_hunyuan():
    """Mock HunyuanVideoPipeline."""
    return create_mock_pipe("HunyuanVideoPipeline")


@pytest.fixture
def mock_pipe_cogvideox():
    """Mock CogVideoXPipeline."""
    return create_mock_pipe("CogVideoXPipeline")


# ============================================================
# Fixtures - Config data
# ============================================================
@pytest.fixture
def sample_cache_config():
    """Sample cache config for testing."""
    return {
        "mag_ratios": {
            "wan2.1-t2v-1.3b": [1.0, 0.99, 0.98, 0.97],
            "wan2.1-t2v-14b": [1.0, 0.99, 0.98, 0.97],
            "wan2.1-i2v-480p": [1.0, 0.99, 0.98, 0.97],
            "wan2.1-i2v-720p": [1.0, 0.99, 0.98, 0.97],
            "wan2.2-t2v-A14B": [1.0, 0.99, 0.98, 0.97],
            "wan2.2-i2v-A14B": [1.0, 0.99, 0.98, 0.97],
            "wan2.2-i2v-x": [0.9, 0.85],
            "wan2.2-t2v-x": [0.88, 0.82],
        }
    }


# 向后兼容别名
sample_turbo_config = sample_cache_config


# ============================================================
# Fixtures - Utility factories
# ============================================================
@pytest.fixture
def args_factory():
    """Factory fixture for creating custom mock args."""
    return create_mock_args


@pytest.fixture
def pipe_factory():
    """Factory fixture for creating custom mock pipes."""
    return create_mock_pipe
