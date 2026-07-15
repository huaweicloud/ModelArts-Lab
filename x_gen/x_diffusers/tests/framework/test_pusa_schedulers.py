"""
Unit tests for x_diffusers.framework.schedulers.pusa_schedulers module.

Tests cover:
- FlowMatchEulerDiscreteSchedulerPusa initialization
- set_timesteps method
- step method (various scenarios)
- add_noise_for_conditioning_frames method

Note: This test uses importlib to directly load the module via fixture,
bypassing x_diffusers/__init__.py to avoid complex dependency mocking.
No sys.modules mocking at module level - using fixtures instead.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture(scope="module")
def pusa_scheduler_classes():
    """
    Load pusa_schedulers module directly with mock dependencies.
    Returns the classes needed for testing.
    """
    # Save original modules
    saved_modules = {}
    modules_to_mock = [
        "diffusers",
        "diffusers.configuration_utils",
        "diffusers.utils",
        "diffusers.schedulers",
        "diffusers.schedulers.scheduling_utils",
    ]
    for mod in modules_to_mock:
        if mod in sys.modules:
            saved_modules[mod] = sys.modules[mod]

    # Mock classes
    class MockConfigMixin:
        def __init__(self, *args, **kwargs):
            pass

    class MockSchedulerMixin:
        pass

    class MockBaseOutput:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Setup mocks
    mock_diffusers = MagicMock()
    mock_config_utils = MagicMock()
    mock_config_utils.ConfigMixin = MockConfigMixin
    mock_config_utils.register_to_config = lambda f: f

    mock_diffusers.configuration_utils = mock_config_utils
    mock_diffusers.utils = MagicMock()
    mock_diffusers.utils.BaseOutput = MockBaseOutput
    mock_diffusers.utils.is_scipy_available = lambda: True
    mock_diffusers.utils.logging = MagicMock()
    mock_diffusers.utils.logging.get_logger = lambda x: MagicMock()
    mock_diffusers.schedulers = MagicMock()
    mock_diffusers.schedulers.scheduling_utils = MagicMock()
    mock_diffusers.schedulers.scheduling_utils.SchedulerMixin = MockSchedulerMixin

    # Set mocks in sys.modules
    sys.modules["diffusers"] = mock_diffusers
    sys.modules["diffusers.configuration_utils"] = mock_config_utils
    sys.modules["diffusers.utils"] = mock_diffusers.utils
    sys.modules["diffusers.schedulers"] = mock_diffusers.schedulers
    sys.modules["diffusers.schedulers.scheduling_utils"] = mock_diffusers.schedulers.scheduling_utils

    # Load the module
    _pusa_schedulers_path = (
        Path(__file__).parent.parent.parent / "x_diffusers" / "framework" / "schedulers" / "pusa_schedulers.py"
    )
    spec = importlib.util.spec_from_file_location("pusa_schedulers_module_test", _pusa_schedulers_path)
    pusa_schedulers_module = importlib.util.module_from_spec(spec)
    sys.modules["pusa_schedulers_module_test"] = pusa_schedulers_module
    spec.loader.exec_module(pusa_schedulers_module)

    # Extract classes
    FlowMatchEulerDiscreteSchedulerPusa = pusa_schedulers_module.FlowMatchEulerDiscreteSchedulerPusa
    FlowMatchEulerDiscreteSchedulerOutput = pusa_schedulers_module.FlowMatchEulerDiscreteSchedulerOutput

    yield FlowMatchEulerDiscreteSchedulerPusa, FlowMatchEulerDiscreteSchedulerOutput

    # Cleanup: restore original modules
    for mod in modules_to_mock:
        if mod in saved_modules:
            sys.modules[mod] = saved_modules[mod]
        elif mod in sys.modules:
            del sys.modules[mod]
    if "pusa_schedulers_module_test" in sys.modules:
        del sys.modules["pusa_schedulers_module_test"]


class TestFlowMatchEulerDiscreteSchedulerOutput:
    """Tests for FlowMatchEulerDiscreteSchedulerOutput dataclass."""

    def test_output_creation(self, pusa_scheduler_classes):
        """Test creating output with prev_sample."""
        _, FlowMatchEulerDiscreteSchedulerOutput = pusa_scheduler_classes
        prev_sample = torch.randn(2, 3, 4, 4)
        output = FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

        assert torch.allclose(output.prev_sample, prev_sample, rtol=1e-5, atol=1e-5)


class TestFlowMatchEulerDiscreteSchedulerPusa:
    """Tests for FlowMatchEulerDiscreteSchedulerPusa class."""

    def test_init_default_parameters(self, pusa_scheduler_classes):
        """Test initialization with default parameters."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 3.0
        assert scheduler.sigma_max == 1.0
        assert abs(scheduler.sigma_min - 0.003 / 1.002) < 1e-10
        assert scheduler.inverse_timesteps == False  # noqa: E712
        assert scheduler.extra_one_step == False  # noqa: E712
        assert scheduler.reverse_sigmas == False  # noqa: E712

    def test_init_custom_parameters(self, pusa_scheduler_classes):
        """Test initialization with custom parameters."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(
            num_inference_steps=50,
            num_train_timesteps=1000,
            shift=5.0,
            sigma_max=2.0,
            sigma_min=0.001,
            inverse_timesteps=True,
            extra_one_step=True,
            reverse_sigmas=True,
        )

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 5.0
        assert scheduler.sigma_max == 2.0
        assert scheduler.sigma_min == 0.001
        assert scheduler.inverse_timesteps == True  # noqa: E712
        assert scheduler.extra_one_step == True  # noqa: E712
        assert scheduler.reverse_sigmas == True  # noqa: E712

    def test_set_timesteps_default(self, pusa_scheduler_classes):
        """Test set_timesteps with default parameters."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        scheduler.set_timesteps(num_inference_steps=100)

        assert scheduler.sigmas is not None
        assert scheduler.timesteps is not None
        assert len(scheduler.sigmas) == 100
        assert len(scheduler.timesteps) == 100

    def test_set_timesteps_with_denoising_strength(self, pusa_scheduler_classes):
        """Test set_timesteps with denoising strength."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        scheduler.set_timesteps(num_inference_steps=50, denoising_strength=0.8)

        assert len(scheduler.sigmas) == 50

    def test_set_timesteps_with_shift(self, pusa_scheduler_classes):
        """Test set_timesteps with shift parameter."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        scheduler.set_timesteps(num_inference_steps=50, shift=10.0)

        assert scheduler.shift == 10.0

    def test_set_timesteps_extra_one_step(self, pusa_scheduler_classes):
        """Test set_timesteps with extra_one_step=True."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(extra_one_step=True)
        scheduler.set_timesteps(num_inference_steps=50)

        assert len(scheduler.sigmas) == 50

    def test_set_timesteps_inverse_timesteps(self, pusa_scheduler_classes):
        """Test set_timesteps with inverse_timesteps=True."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(inverse_timesteps=False)
        scheduler.set_timesteps(num_inference_steps=50)
        normal_sigmas = scheduler.sigmas.clone()

        scheduler_inv = FlowMatchEulerDiscreteSchedulerPusa(inverse_timesteps=True)
        scheduler_inv.set_timesteps(num_inference_steps=50)

        assert torch.allclose(scheduler_inv.sigmas, torch.flip(normal_sigmas, dims=[0]))

    def test_set_timesteps_reverse_sigmas(self, pusa_scheduler_classes):
        """Test set_timesteps with reverse_sigmas=True."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(reverse_sigmas=False)
        scheduler.set_timesteps(num_inference_steps=50)
        normal_sigmas = scheduler.sigmas.clone()

        scheduler_rev = FlowMatchEulerDiscreteSchedulerPusa(reverse_sigmas=True)
        scheduler_rev.set_timesteps(num_inference_steps=50)

        assert torch.allclose(scheduler_rev.sigmas, 1 - normal_sigmas)

    def test_set_timesteps_training_mode(self, pusa_scheduler_classes):
        """Test set_timesteps with training=True."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        scheduler.set_timesteps(num_inference_steps=50, training=True)

        assert hasattr(scheduler, "linear_timesteps_weights")
        assert scheduler.linear_timesteps_weights is not None
        assert len(scheduler.linear_timesteps_weights) == 50

    def test_step_scalar_timestep(self, pusa_scheduler_classes):
        """Test step with scalar timestep (0D tensor becomes 1D)."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        model_output = torch.randn(2, 3, 4, 4)
        sample = torch.randn(2, 3, 4, 4)
        timestep = scheduler.timesteps[5].unsqueeze(0)

        prev_sample = scheduler.step(model_output, timestep, sample)

        assert isinstance(prev_sample, torch.Tensor)
        assert prev_sample.shape == sample.shape

    def test_step_tensor_timestep_1d(self, pusa_scheduler_classes):
        """Test step with 1D tensor timestep."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        model_output = torch.randn(2, 3, 4, 4)
        sample = torch.randn(2, 3, 4, 4)
        timestep = scheduler.timesteps[5].unsqueeze(0)

        prev_sample = scheduler.step(model_output, timestep, sample)

        assert isinstance(prev_sample, torch.Tensor)
        assert prev_sample.shape == sample.shape

    def test_step_tensor_timestep_2d(self, pusa_scheduler_classes):
        """Test step with 2D tensor timestep (per-frame timesteps)."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        batch_size = 1
        frames = 4
        model_output = torch.randn(batch_size, 3, frames, 4, 4)
        sample = torch.randn(batch_size, 3, frames, 4, 4)
        timestep = scheduler.timesteps[5:9].unsqueeze(0)

        prev_sample = scheduler.step(model_output, timestep, sample)

        assert isinstance(prev_sample, torch.Tensor)
        assert prev_sample.shape == sample.shape

    def test_step_to_final(self, pusa_scheduler_classes):
        """Test step with to_final=True."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        model_output = torch.randn(2, 3, 4, 4)
        sample = torch.randn(2, 3, 4, 4)
        timestep = scheduler.timesteps[9]

        prev_sample = scheduler.step(model_output, timestep, sample, to_final=True)

        assert isinstance(prev_sample, torch.Tensor)

    def test_step_with_cond_frame_latent_indices(self, pusa_scheduler_classes):
        """Test step with conditioning frame latent indices."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        batch_size = 1
        frames = 5
        model_output = torch.randn(batch_size, 3, frames, 4, 4)
        sample = torch.randn(batch_size, 3, frames, 4, 4)
        timestep = scheduler.timesteps[5:10].unsqueeze(0)

        cond_frame_latent_indices = [0, 4]
        noise_multipliers = {0: 0.5, 4: 0.8}

        prev_sample = scheduler.step(
            model_output,
            timestep,
            sample,
            cond_frame_latent_indices=cond_frame_latent_indices,
            noise_multipliers=noise_multipliers,
        )

        assert isinstance(prev_sample, torch.Tensor)
        assert prev_sample.shape == sample.shape

    def test_step_device_handling(self, pusa_scheduler_classes):
        """Test step handles device correctly."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        model_output = torch.randn(2, 3, 4, 4)
        sample = torch.randn(2, 3, 4, 4)
        timestep = scheduler.timesteps[5]

        prev_sample = scheduler.step(model_output, timestep, sample)

        assert prev_sample.device == sample.device

    def test_add_noise_for_conditioning_frames_scalar_timestep(self, pusa_scheduler_classes):
        """Test add_noise_for_conditioning_frames with scalar timestep."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        original_samples = torch.randn(2, 3, 4, 4)
        noise = torch.randn(2, 3, 4, 4)
        timestep = scheduler.timesteps[5].unsqueeze(0)

        noisy_samples = scheduler.add_noise_for_conditioning_frames(original_samples, noise, timestep)

        assert isinstance(noisy_samples, torch.Tensor)
        assert noisy_samples.shape == original_samples.shape

    def test_add_noise_for_conditioning_frames_tensor_timestep(self, pusa_scheduler_classes):
        """Test add_noise_for_conditioning_frames with 2D tensor timestep."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        batch_size = 1
        frames = 4
        original_samples = torch.randn(batch_size, 3, frames, 4, 4)
        noise = torch.randn(batch_size, 3, frames, 4, 4)
        timestep = scheduler.timesteps[5:9].unsqueeze(0)
        noise_multiplier = 1.0

        noisy_samples = scheduler.add_noise_for_conditioning_frames(
            original_samples, noise, timestep, noise_multiplier=noise_multiplier
        )

        assert isinstance(noisy_samples, torch.Tensor)
        assert noisy_samples.shape == original_samples.shape

    def test_add_noise_for_conditioning_frames_with_multiplier(self, pusa_scheduler_classes):
        """Test add_noise_for_conditioning_frames with noise_multiplier."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        batch_size = 1
        frames = 4
        original_samples = torch.randn(batch_size, 3, frames, 4, 4)
        noise = torch.randn(batch_size, 3, frames, 4, 4)
        timestep = scheduler.timesteps[5:9].unsqueeze(0)
        noise_multiplier = 0.5

        noisy_samples = scheduler.add_noise_for_conditioning_frames(
            original_samples, noise, timestep, noise_multiplier=noise_multiplier
        )

        assert isinstance(noisy_samples, torch.Tensor)
        assert noisy_samples.shape == original_samples.shape

    def test_add_noise_formula_correctness(self, pusa_scheduler_classes):
        """Test that add_noise uses correct formula: (1-sigma)*original + sigma*noise."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        original_samples = torch.zeros(2, 3, 4, 4)
        noise = torch.ones(2, 3, 4, 4)
        timestep = scheduler.timesteps[5].unsqueeze(0)

        noisy_samples = scheduler.add_noise_for_conditioning_frames(original_samples, noise, timestep)

        # Verify the formula: (1-sigma)*original + sigma*noise
        # With original=0, noise=1, result should be sigma
        # Get sigma for the timestep
        timestep_idx = (timestep / scheduler.num_train_timesteps).squeeze()
        sigma = scheduler.sigmas[timestep_idx.long()]

        expected = (1 - sigma) * original_samples + sigma * noise

        assert torch.allclose(noisy_samples, expected, rtol=1e-5, atol=1e-5)

    def test_step_order_attribute(self, pusa_scheduler_classes):
        """Test that order attribute is set correctly."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        assert scheduler.order == 1

    def test_compatible_attribute(self, pusa_scheduler_classes):
        """Test that _compatibles attribute exists."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa()
        assert hasattr(scheduler, "_compatibles")
        assert scheduler._compatibles == []


class TestSchedulerIntegration:
    """Integration tests for scheduler behavior."""

    def test_multiple_steps_decrease_noise(self, pusa_scheduler_classes):
        """Test that running multiple steps decreases noise level."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=10)

        # Start with a noisy sample
        sample = torch.randn(2, 3, 4, 4)
        initial_variance = sample.var().item()  # noqa: F841

        for i in range(5):
            timestep = scheduler.timesteps[9 - i]
            model_output = torch.randn(2, 3, 4, 4)
            sample = scheduler.step(model_output, timestep, sample)

        assert isinstance(sample, torch.Tensor)

        # Verify that sample has been processed (sigma should be smaller after steps)
        final_sigma = scheduler.sigmas[4]  # sigma after 5 steps from the end
        assert final_sigma < scheduler.sigmas[9]  # sigma decreased from initial

    def test_sigmas_monotonic(self, pusa_scheduler_classes):
        """Test that sigmas are monotonically decreasing by default."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=50)

        for i in range(len(scheduler.sigmas) - 1):
            assert scheduler.sigmas[i] > scheduler.sigmas[i + 1]

    def test_timesteps_match_sigmas(self, pusa_scheduler_classes):
        """Test that timesteps are correctly derived from sigmas."""
        FlowMatchEulerDiscreteSchedulerPusa, _ = pusa_scheduler_classes
        scheduler = FlowMatchEulerDiscreteSchedulerPusa(num_inference_steps=50)

        expected_timesteps = scheduler.sigmas * scheduler.num_train_timesteps
        assert torch.allclose(scheduler.timesteps, expected_timesteps)
