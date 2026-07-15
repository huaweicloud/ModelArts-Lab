"""
Shared fixtures and utilities for framework tests.

This module provides:
- Direct module loading (bypassing x_diffusers/__init__.py)
- Common mock setup for diffusers, peft, and x_base
- Test isolation to prevent sys.modules pollution
"""

import abc
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ============================================================================
# Test Isolation
# ============================================================================

# Modules that are commonly mocked in framework tests
_ISOLATED_MODULES = [
    "diffusers",
    "diffusers.configuration_utils",
    "diffusers.loaders",
    "diffusers.loaders.peft",
    "diffusers.utils",
    "diffusers.utils.torch_utils",
    "diffusers.models",
    "diffusers.models.attention",
    "diffusers.models.transformers",
    "diffusers.models.autoencoders",
    "diffusers.models.transformers.transformer_wan",
    "diffusers.models.transformers.transformer_hunyuan_video",
    "diffusers.models.transformers.transformer_cogvideox",
    "diffusers.models.attention_processor",
    "diffusers.models.cache_utils",
    "diffusers.models.embeddings",
    "diffusers.models.modeling_outputs",
    "diffusers.models.modeling_utils",
    "diffusers.models.normalization",
    "diffusers.schedulers",
    "diffusers.schedulers.scheduling_utils",
    "diffusers.pipelines",
    "diffusers.pipelines.pipeline_utils",
    "diffusers.image_processor",
    "diffusers.video_processor",
    "diffusers.callbacks",
    "x_base",
    "x_base.utils",
    "x_base.utils.infer_info",
    "x_base.vae_parallelism",
    "x_base.vae_parallelism.vae_mgr",
    "x_base.vae_parallelism.utils",
    "x_base.attention_manager",
    "x_base.rope_manager",
    "peft",
    "peft.tuners",
    "peft.tuners.lora",
    "peft.tuners.lora.model",
    "peft.tuners.lora.layer",
    "peft.tuners.lora.aqlm",
    "peft.tuners.lora.awq",
    "peft.tuners.lora.eetq",
    "peft.tuners.lora.gptq",
    "peft.tuners.lora.hqq",
    "peft.tuners.lora.torchao",
    "peft.tuners.lora.tp_layer",
    "peft.tuners.tuners_utils",
    "torch_npu",
    "torch_npu.contrib",
]


def pytest_configure(config):
    """Register custom markers and save original sys.modules state."""
    # Save original modules state before any tests run
    config._framework_original_modules = {}
    for module_name in _ISOLATED_MODULES:
        if module_name in sys.modules:
            config._framework_original_modules[module_name] = sys.modules[module_name]

    # Register markers
    config.addinivalue_line("markers", "framework_test: mark test as a framework test that uses mocks")


def pytest_unconfigure(config):
    """Restore original sys.modules state after all tests complete."""
    if hasattr(config, "_framework_original_modules"):
        for module_name in _ISOLATED_MODULES:
            if module_name in config._framework_original_modules:
                sys.modules[module_name] = config._framework_original_modules[module_name]
            elif module_name in sys.modules:
                del sys.modules[module_name]


# Track whether we're running framework tests
_current_module_is_framework = False


def pytest_collection_modifyitems(session, config, items):
    """Identify framework tests and mark them."""
    global _current_module_is_framework

    framework_tests_dir = Path(__file__).parent

    for item in items:
        # Check if the test is in the framework directory
        test_path = Path(item.fspath)
        if framework_tests_dir in test_path.parents or test_path.parent == framework_tests_dir:
            item.add_marker(pytest.mark.framework_test)


@pytest.fixture(scope="module", autouse=True)
def isolate_framework_module(request):
    """Module-level fixture to isolate framework tests.

    Saves sys.modules state before each module and restores after.
    """
    # Check if this is a framework test module
    test_module = request.module
    if test_module is None:
        yield
        return

    module_file = getattr(test_module, "__file__", None)
    if module_file is None:
        yield
        return

    framework_tests_dir = Path(__file__).parent
    test_path = Path(module_file)

    is_framework_test = framework_tests_dir in test_path.parents or test_path.parent == framework_tests_dir

    if not is_framework_test:
        yield
        return

    # Save current state of isolated modules
    saved_modules = {}
    for module_name in _ISOLATED_MODULES:
        if module_name in sys.modules:
            saved_modules[module_name] = sys.modules[module_name]

    yield

    # Restore: remove any new mocked modules
    for module_name in _ISOLATED_MODULES:
        if module_name in saved_modules:
            sys.modules[module_name] = saved_modules[module_name]
        elif module_name in sys.modules:
            del sys.modules[module_name]


# ============================================================================
# Mock Classes
# ============================================================================


class MockBaseTunerLayer(abc.ABC):  # noqa: B024
    """Mock BaseTunerLayer with ABCMeta metaclass."""

    def __init__(self, base_layer):
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = {}
        self.lora_A = {}
        self.lora_B = {}
        self.lora_embedding_A = {}
        self.lora_embedding_B = {}
        self.disable_adapters = False
        self.merged = False

    def get_base_layer(self):
        return self.base_layer

    def update_layer(self, *args, **kwargs):  # noqa: B027
        pass


class MockLoraLayer(MockBaseTunerLayer):
    """Mock LoraLayer with ABCMeta metaclass."""

    adapter_layer_names = ("lora_A", "lora_B")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")


class MockWeightQuantLinearModule(torch.nn.Module):
    """Mock WeightQuantLinearModule class for isinstance checks."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class MockConfigMixin:
    """Mock ConfigMixin from diffusers."""

    def __init__(self, *args, **kwargs):
        pass


class MockSchedulerMixin:
    """Mock SchedulerMixin from diffusers."""

    pass


class MockBaseOutput:
    """Mock BaseOutput from diffusers.utils."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ============================================================================
# Module Loading
# ============================================================================


def load_module_directly(module_name: str, file_path: str):
    """
    Load a Python module directly from file, bypassing package __init__.py.

    Args:
        module_name: Name to register the module as in sys.modules
        file_path: Path to the Python file to load

    Returns:
        The loaded module object
    """
    path = Path(file_path)
    if not path.is_absolute():
        # Resolve relative to this conftest.py location
        conftest_dir = Path(__file__).parent
        path = conftest_dir.parent.parent / "x_diffusers" / file_path

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Mock Setup Functions
# ============================================================================


def setup_diffusers_mock():
    """
    Setup comprehensive diffusers mock with all necessary sub-modules.
    Returns the mock_diffusers object.
    """
    mock_diffusers = MagicMock()

    # configuration_utils
    mock_config_utils = MagicMock()
    mock_config_utils.ConfigMixin = MockConfigMixin
    mock_config_utils.register_to_config = lambda f: f
    mock_diffusers.configuration_utils = mock_config_utils

    # utils
    mock_utils = MagicMock()
    mock_utils.BaseOutput = MockBaseOutput
    mock_utils.logging = MagicMock()
    mock_utils.logging.get_logger = lambda x: MagicMock()
    mock_utils.is_ftfy_available = lambda: False
    mock_utils.is_torch_xla_available = lambda: False
    mock_utils.is_scipy_available = lambda: True
    mock_utils.replace_example_docstring = lambda f: f
    mock_diffusers.utils = mock_utils

    # utils.torch_utils
    mock_torch_utils = MagicMock()
    mock_torch_utils.randn_tensor = MagicMock()
    mock_diffusers.utils.torch_utils = mock_torch_utils

    # models
    mock_models = MagicMock()
    mock_models.attention = MagicMock()
    mock_models.attention.FeedForward = MagicMock
    mock_models.transformers = MagicMock()
    mock_models.autoencoders = MagicMock()
    mock_diffusers.models = mock_models

    # schedulers
    mock_schedulers = MagicMock()
    mock_schedulers.scheduling_utils = MagicMock()
    mock_schedulers.scheduling_utils.SchedulerMixin = MockSchedulerMixin
    mock_diffusers.schedulers = mock_schedulers

    # pipelines
    mock_pipelines = MagicMock()
    mock_pipelines.pipeline_utils = MagicMock()
    mock_pipelines.pipeline_utils.DiffusionPipeline = MagicMock
    mock_diffusers.pipelines = mock_pipelines

    # loaders
    mock_loaders = MagicMock()
    mock_loaders.peft = MagicMock()
    mock_loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING = {}
    mock_diffusers.loaders = mock_loaders

    # callbacks
    mock_diffusers.callbacks = MagicMock()
    mock_diffusers.callbacks.MultiPipelineCallbacks = MagicMock
    mock_diffusers.callbacks.PipelineCallback = MagicMock

    # image_processor
    mock_diffusers.image_processor = MagicMock()
    mock_diffusers.image_processor.PipelineImageInput = MagicMock

    # video_processor
    mock_diffusers.video_processor = MagicMock()
    mock_diffusers.video_processor.VideoProcessor = MagicMock

    # Register in sys.modules
    sys.modules["diffusers"] = mock_diffusers
    sys.modules["diffusers.configuration_utils"] = mock_config_utils
    sys.modules["diffusers.utils"] = mock_utils
    sys.modules["diffusers.utils.torch_utils"] = mock_torch_utils
    sys.modules["diffusers.models"] = mock_models
    sys.modules["diffusers.models.attention"] = mock_models.attention
    sys.modules["diffusers.models.transformers"] = mock_models.transformers
    sys.modules["diffusers.models.autoencoders"] = mock_models.autoencoders
    sys.modules["diffusers.schedulers"] = mock_schedulers
    sys.modules["diffusers.schedulers.scheduling_utils"] = mock_schedulers.scheduling_utils
    sys.modules["diffusers.pipelines"] = mock_pipelines
    sys.modules["diffusers.pipelines.pipeline_utils"] = mock_pipelines.pipeline_utils
    sys.modules["diffusers.loaders"] = mock_loaders
    sys.modules["diffusers.loaders.peft"] = mock_loaders.peft
    sys.modules["diffusers.callbacks"] = mock_diffusers.callbacks
    sys.modules["diffusers.image_processor"] = mock_diffusers.image_processor
    sys.modules["diffusers.video_processor"] = mock_diffusers.video_processor

    return mock_diffusers


def setup_peft_mock():
    """
    Setup peft mock with proper LoraLayer class (using abc.ABCMeta).
    Returns (mock_peft, mock_lora_model).
    """
    mock_peft = MagicMock()
    mock_lora_model = MagicMock(name="LoraModel")
    mock_lora_model._create_new_module = None

    mock_peft.tuners = MagicMock()
    mock_peft.tuners.lora = MagicMock()
    mock_peft.tuners.lora.model = MagicMock()
    mock_peft.tuners.lora.model.LoraModel = mock_lora_model
    mock_peft.tuners.tuners_utils = MagicMock()
    mock_peft.tuners.tuners_utils.BaseTunerLayer = MockBaseTunerLayer

    # Create individual dispatch function mocks
    for name in ["aqlm", "awq", "eetq", "gptq", "hqq", "torchao"]:
        module = MagicMock()
        setattr(mock_peft.tuners.lora, name, module)
        setattr(module, f"dispatch_{name}", MagicMock(return_value=None))

    # tp_layer special case
    tp_layer = MagicMock()
    tp_layer.dispatch_megatron = MagicMock(return_value=None)
    mock_peft.tuners.lora.tp_layer = tp_layer

    mock_peft.tuners.lora.layer = MagicMock()
    mock_peft.tuners.lora.layer.Conv2d = MagicMock(name="Conv2d")
    mock_peft.tuners.lora.layer.LoraLayer = MockLoraLayer
    mock_peft.tuners.lora.layer.dispatch_default = MagicMock(return_value=None)

    # Register in sys.modules
    sys.modules["peft"] = mock_peft
    sys.modules["peft.tuners"] = mock_peft.tuners
    sys.modules["peft.tuners.lora"] = mock_peft.tuners.lora
    sys.modules["peft.tuners.lora.model"] = mock_peft.tuners.lora.model
    sys.modules["peft.tuners.lora.aqlm"] = mock_peft.tuners.lora.aqlm
    sys.modules["peft.tuners.lora.awq"] = mock_peft.tuners.lora.awq
    sys.modules["peft.tuners.lora.eetq"] = mock_peft.tuners.lora.eetq
    sys.modules["peft.tuners.lora.gptq"] = mock_peft.tuners.lora.gptq
    sys.modules["peft.tuners.lora.hqq"] = mock_peft.tuners.lora.hqq
    sys.modules["peft.tuners.lora.layer"] = mock_peft.tuners.lora.layer
    sys.modules["peft.tuners.lora.torchao"] = mock_peft.tuners.lora.torchao
    sys.modules["peft.tuners.lora.tp_layer"] = mock_peft.tuners.lora.tp_layer
    sys.modules["peft.tuners.tuners_utils"] = mock_peft.tuners.tuners_utils

    return mock_peft, mock_lora_model


def setup_x_base_mock():
    """
    Setup x_base mock with proper WeightQuantLinearModule class.
    Returns (mock_x_base, mock_weight_quant).
    """
    mock_x_base = MagicMock()
    mock_x_base.WeightQuantLinearModule = MockWeightQuantLinearModule
    mock_x_base.enable_sp = MagicMock()
    mock_x_base.enable_vae_lightning = MagicMock()
    sys.modules["x_base"] = mock_x_base
    return mock_x_base, MockWeightQuantLinearModule


def setup_all_mocks():
    """
    Setup all mocks: diffusers, peft, and x_base.
    Returns (mock_diffusers, mock_peft, mock_lora_model, mock_x_base, mock_weight_quant).
    """
    mock_diffusers = setup_diffusers_mock()
    mock_peft, mock_lora_model = setup_peft_mock()
    mock_x_base, mock_weight_quant = setup_x_base_mock()
    return mock_diffusers, mock_peft, mock_lora_model, mock_x_base, mock_weight_quant


def cleanup_mocks():
    """Clean up all mocked modules from sys.modules."""
    for module_name in _ISOLATED_MODULES:
        if module_name in sys.modules:
            del sys.modules[module_name]


# ============================================================================
# Path Helpers
# ============================================================================


def get_x_diffusers_path():
    """Get the path to x_diffusers package."""
    return Path(__file__).parent.parent.parent / "x_diffusers"


def get_framework_module_path(module_rel_path: str):
    """
    Get the full path to a framework module.

    Args:
        module_rel_path: Relative path from x_diffusers/framework/, e.g., "lora/lora.py"

    Returns:
        Full path to the module file
    """
    return get_x_diffusers_path() / "framework" / module_rel_path
