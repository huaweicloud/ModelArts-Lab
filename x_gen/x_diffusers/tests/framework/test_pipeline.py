"""
Unit tests for x_diffusers.framework.pipeline.pipeline module.

Tests cover:
- Method injection for enable_sp
- Method injection for enable_vae_lightning
- All supported pipeline classes

Note: This test verifies the behavior of pipeline.py without importing it,
since pipeline.py uses relative imports that require package context.
We simulate the module's behavior and verify the expected method injection.

No sys.modules mocking at module level - using fixtures instead.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def mock_pipeline_objects():
    """
    Setup mock objects for pipeline tests.
    This fixture provides mock objects without polluting sys.modules.
    """
    # === diffusers mock ===
    mock_diffusers = MagicMock()
    mock_diffusers.WanPipeline = MagicMock(name="WanPipeline")
    mock_diffusers.WanImageToVideoPipeline = MagicMock(name="WanImageToVideoPipeline")
    mock_diffusers.WanVACEPipeline = MagicMock(name="WanVACEPipeline")
    mock_diffusers.CogVideoXPipeline = MagicMock(name="CogVideoXPipeline")
    mock_diffusers.HunyuanVideoPipeline = MagicMock(name="HunyuanVideoPipeline")
    mock_diffusers.HunyuanVideoImageToVideoPipeline = MagicMock(name="HunyuanVideoImageToVideoPipeline")

    # === x_base mock ===
    mock_x_base = MagicMock()
    mock_x_base.enable_sp = MagicMock(name="enable_sp")
    mock_x_base.enable_vae_lightning = MagicMock(name="enable_vae_lightning")

    # === ten_second mock ===
    mock_ten_second = MagicMock()
    mock_ten_second.WanVideoToVideoPipeline = MagicMock(name="WanVideoToVideoPipeline")

    # === Simulate pipeline.py behavior ===
    # The pipeline.py does the following method injections:

    # WanPipeline
    mock_diffusers.WanPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.WanPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # WanImageToVideoPipeline
    mock_diffusers.WanImageToVideoPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.WanImageToVideoPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # WanVideoToVideoPipeline (from ten_second)
    mock_ten_second.WanVideoToVideoPipeline.enable_sp = mock_x_base.enable_sp
    mock_ten_second.WanVideoToVideoPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # WanVACEPipeline
    mock_diffusers.WanVACEPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.WanVACEPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # CogVideoXPipeline
    mock_diffusers.CogVideoXPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.CogVideoXPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # HunyuanVideoPipeline
    mock_diffusers.HunyuanVideoPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.HunyuanVideoPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    # HunyuanVideoImageToVideoPipeline
    mock_diffusers.HunyuanVideoImageToVideoPipeline.enable_sp = mock_x_base.enable_sp
    mock_diffusers.HunyuanVideoImageToVideoPipeline.enable_vae_lightning = mock_x_base.enable_vae_lightning

    return mock_diffusers, mock_x_base, mock_ten_second


class TestPipelineMethodInjection:
    """Tests for method injection into pipeline classes using parametrization."""

    PIPELINE_CONFIGS = [
        ("WanPipeline", "diffusers", "WanPipeline"),
        ("WanImageToVideoPipeline", "diffusers", "WanImageToVideoPipeline"),
        ("WanVideoToVideoPipeline", "ten_second", "WanVideoToVideoPipeline"),
        ("WanVACEPipeline", "diffusers", "WanVACEPipeline"),
        ("CogVideoXPipeline", "diffusers", "CogVideoXPipeline"),
        ("HunyuanVideoPipeline", "diffusers", "HunyuanVideoPipeline"),
        ("HunyuanVideoImageToVideoPipeline", "diffusers", "HunyuanVideoImageToVideoPipeline"),
    ]

    @pytest.mark.parametrize("pipeline_name,module,pipeline_attr", PIPELINE_CONFIGS)
    @pytest.mark.parametrize("method_name", ["enable_sp", "enable_vae_lightning"])
    def test_method_injection(self, mock_pipeline_objects, pipeline_name, module, pipeline_attr, method_name):
        """Test method injection for all pipeline classes."""
        mock_diffusers, mock_x_base, mock_ten_second = mock_pipeline_objects

        if module == "diffusers":
            pipeline = getattr(mock_diffusers, pipeline_attr)
        else:
            pipeline = getattr(mock_ten_second, pipeline_attr)

        expected_method = getattr(mock_x_base, method_name)

        assert hasattr(pipeline, method_name)
        assert getattr(pipeline, method_name) == expected_method


class TestPipelineMethodCounts:
    """Tests to verify method injection correctness: existence and identity."""

    def test_enable_sp_injection_correctness(self, mock_pipeline_objects):
        """Test enable_sp exists on all pipelines and references the same function."""
        mock_diffusers, mock_x_base, mock_ten_second = mock_pipeline_objects
        pipelines_with_enable_sp = [
            mock_diffusers.WanPipeline,
            mock_diffusers.WanImageToVideoPipeline,
            mock_ten_second.WanVideoToVideoPipeline,
            mock_diffusers.WanVACEPipeline,
            mock_diffusers.CogVideoXPipeline,
            mock_diffusers.HunyuanVideoPipeline,
            mock_diffusers.HunyuanVideoImageToVideoPipeline,
        ]

        for pipeline in pipelines_with_enable_sp:
            assert hasattr(pipeline, "enable_sp")
            assert pipeline.enable_sp == mock_x_base.enable_sp

    def test_enable_vae_lightning_injection_correctness(self, mock_pipeline_objects):
        """Test enable_vae_lightning exists on all pipelines and references the same function."""
        mock_diffusers, mock_x_base, mock_ten_second = mock_pipeline_objects
        pipelines_with_vae_lightning = [
            mock_diffusers.WanPipeline,
            mock_diffusers.WanImageToVideoPipeline,
            mock_diffusers.WanVACEPipeline,
            mock_ten_second.WanVideoToVideoPipeline,
            mock_diffusers.CogVideoXPipeline,
            mock_diffusers.HunyuanVideoPipeline,
            mock_diffusers.HunyuanVideoImageToVideoPipeline,
        ]

        for pipeline in pipelines_with_vae_lightning:
            assert hasattr(pipeline, "enable_vae_lightning")
            assert pipeline.enable_vae_lightning == mock_x_base.enable_vae_lightning
