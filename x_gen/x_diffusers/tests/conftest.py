"""
Pytest configuration and fixtures for x_diffusers unit tests.
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
# Test data constants
# ============================================================
SUPPORTED_MODELS = [
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B",
    "Wan2.1-T2V-1.3B",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
    "Wan2.1-VACE-14B",
    "Wan2.1-VACE-1.3B",
    "CogVideoX-5b",
    "HunyuanVideo-T2V-13B",
]

VALID_TASK_TYPES = ["t2v", "i2v", "t2i"]

VALID_VACE_TASKS = ["t2v", "i2v", "v2lf", "flf2v", "random2v", "inpaint", "outpaint", "openpose", "iwri"]

VALID_TURBO_MODES = ["default", "faiz", "next_faiz"]

# Guidance scale defaults by model
GUIDANCE_SCALE_MAP = {
    "Wan2.1-T2V-14B": (5.0, None),
    "Wan2.1-I2V-14B": (5.0, None),
    "Wan2.1-T2V-1.3B": (5.0, None),
    "CogVideoX-5b": (5.0, None),
    "HunyuanVideo-T2V-13B": (5.0, None),
    "Wan2.2-I2V-A14B": (3.5, 3.5),
    "Wan2.2-T2V-A14B": (3.0, 4.0),
}


# ============================================================
# Helper functions
# ============================================================
def create_mock_args(**overrides):
    """Create mock args with default values, allowing overrides."""
    defaults = {
        "model": "Wan2.1-T2V-14B",
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "task_type": "t2v",
        "prompt": "A cat and a dog baking a cake together",
        "negative_prompt": "Briqht tones, overexposed, static",
        "height": 480,
        "width": 720,
        "frames": 121,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
        "flow_shift": 3.0,
        "seed": 42,
        "save_fps": 16,
        "save_path": "./outputs/output.mp4",
        "i2v_image_path": "",
        "i2v_image_preprocess": "wan",
        # VACE args
        "vace_task": "",
        "image_path": "",
        "video_path": None,
        "first_frame_path": "",
        "last_frame_path": "",
        "image_path_list": None,
        "frame_indices": [0, 45, 70],
        "directions": ["left", "right"],
        "expand_ratio": 0.25,
        # Abilities args
        "sp": 1,
        "ulysses_degree": None,
        "ring_degree": None,
        "phaa_num": 1,
        "turbo_mode": "default",
        "vae_lightning": False,
        "frame_interpolation": False,
        "frame_interpolation_sr": False,
        "frame_model_path": str(
            pathlib.Path(__file__).parent.parent.parent.parent.parent / "weights" / "IFRNet_S_Vimeo90K.pth"
        ),
        "x": False,
        "x_model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers_x",
        "x_model_path_2": "Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2",
        "joint": False,
        "ten_second": False,
        "ten_second_model_id_t2v": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "ten_second_model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers_x",
        "ten_second_model_path_2": "Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2",
        "pusa_lora": "Wan-AI/pytorch_lora_weights_high.safetensors",
        "pusa_lora2": "Wan-AI/pytorch_lora_weights_low.safetensors",
        "joint_model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers_joint",
        "fsdp": None,
        "atten_a8w8": False,
        "atten_laser": False,
        "atten_rainfusion": False,
        "rainfusion_ratio": 0.375,
        "atten_ada_sparse": False,
        "ada_sparsity": 0.7,
        "matmul_a8w8": False,
        "matmul_a4w4": False,
        "conv3d_w8a8": False,
        "rope_fused": False,
        "lora_path_list": None,
        "lora_scale_weight_list": None,
        "lora_transformer_list": None,
        "fuse_lora": False,
        "server": False,
        "inf_vram_blocks_num": 0,
        "seedvr2_model_dir": "../weights/SeedVR2",
        "seedvr2_model_name": "seedvr2_ema_7b_fp16.safetensors",
        "adopt_sr": False,
        "resolution": 1080,
        "ada_brighten": False,
        "save_memory": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def create_mock_pipe(pipeline_name="WanPipeline"):
    """Create a mock DiffusionPipeline with specified name."""
    pipe = MagicMock()
    pipe.__class__.__name__ = pipeline_name
    pipe.transformer = MagicMock()
    pipe.transformer.__class__ = MagicMock()
    pipe.transformer.config = MagicMock()
    pipe.transformer.config.patch_size = (1, 2, 2)
    pipe.transformer.blocks = MagicMock()
    pipe.scheduler = MagicMock()
    pipe.scheduler.timesteps = MagicMock()
    pipe.scheduler.config = MagicMock()
    pipe.scheduler.config.num_train_timesteps = 1000
    pipe.scheduler.config.flow_shift = 3.0
    pipe.scheduler.set_timesteps = MagicMock()
    pipe.vae = MagicMock()
    pipe.vae_scale_factor_spatial = 8
    pipe.text_encoder = MagicMock()
    pipe.to = MagicMock(return_value=pipe)
    pipe.enable_sp = MagicMock()
    pipe.enable_vae_lightning = MagicMock()
    pipe.load_lora_weights = MagicMock()
    pipe.set_adapters = MagicMock()
    pipe.unload_lora_weights = MagicMock()
    pipe.fuse_lora = MagicMock()
    return pipe


# ============================================================
# Fixtures - Default model configurations
# ============================================================
@pytest.fixture
def mock_args():
    """Mock args namespace for Wan2.1-T2V-14B model."""
    return create_mock_args()


@pytest.fixture
def mock_args_wan13b():
    """Mock args for Wan2.1-T2V-1.3B model."""
    return create_mock_args(
        model="Wan2.1-T2V-1.3B",
        pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )


@pytest.fixture
def mock_args_wan22():
    """Mock args for Wan2.2 model."""
    return create_mock_args(
        model="Wan2.2-T2V-A14B",
        pretrained_model_name_or_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )


@pytest.fixture
def mock_args_i2v():
    """Mock args for I2V model."""
    return create_mock_args(
        model="Wan2.1-I2V-14B",
        task_type="i2v",
        pretrained_model_name_or_path="Wan-AI/Wan2.1-I2V-14B-Diffusers",
        i2v_image_path="input.jpg",
    )


@pytest.fixture
def mock_args_hunyuan():
    """Mock args for HunyuanVideo model."""
    return create_mock_args(
        model="HunyuanVideo-T2V-13B",
        pretrained_model_name_or_path="HunyuanVideo-T2V-13B",
    )


@pytest.fixture
def mock_args_cogvideox():
    """Mock args for CogVideoX model."""
    return create_mock_args(
        model="CogVideoX-5b",
        pretrained_model_name_or_path="CogVideoX-5b",
    )


@pytest.fixture
def mock_args_vace():
    """Mock args for VACE model."""
    return create_mock_args(
        model="Wan2.1-VACE-14B",
        vace_task="i2v",
        image_path="reference.jpg",
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


# ============================================================
# NPU Mock Fixtures
# ============================================================
# Import NPU mock fixtures for use in transformer/VAE tests
# These are defined in conftest_npu.py and imported here

import importlib.util  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402

_conftest_npu_path = os.path.join(os.path.dirname(__file__), "conftest_npu.py")
if os.path.exists(_conftest_npu_path):
    # 安全性检查：确保文件在当前目录下（路径白名单）
    _abs_path = os.path.abspath(_conftest_npu_path)
    _expected_dir = os.path.dirname(os.path.abspath(__file__))
    if not _abs_path.startswith(_expected_dir):
        raise ImportError(f"Security: conftest_npu.py must be in the same directory. Path: {_abs_path}")

    # 安全性检查：确保文件权限合理（不可被其他用户写入）
    _file_stat = os.stat(_conftest_npu_path)
    if _file_stat.st_mode & stat.S_IWOTH:
        raise ImportError(f"Security: conftest_npu.py has insecure permissions (world-writable). Path: {_abs_path}")

    _spec = importlib.util.spec_from_file_location("conftest_npu", _conftest_npu_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Security: Failed to create module spec for conftest_npu.py. Path: {_abs_path}")
    _conftest_npu = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_conftest_npu)

    # Expose key mock utilities
    mock_torch_npu = _conftest_npu.mock_torch_npu
    mock_x_base = _conftest_npu.mock_x_base
    mock_all_npu = _conftest_npu.mock_all_npu
    npu_mock_context = _conftest_npu.npu_mock_context
    MockNPUModule = _conftest_npu.MockNPUModule
    MockAttentionManager = _conftest_npu.MockAttentionManager
    MockRopeManager = _conftest_npu.MockRopeManager
    MockParallelManager = _conftest_npu.MockParallelManager
