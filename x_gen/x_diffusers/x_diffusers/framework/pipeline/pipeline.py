from diffusers import (
    CogVideoXPipeline,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoPipeline,
    WanImageToVideoPipeline,
    WanPipeline,
    WanVACEPipeline,
)
from x_base import enable_sp, enable_vae_lightning

from .ten_second import WanVideoToVideoPipeline

WanPipeline.enable_sp = enable_sp
WanImageToVideoPipeline.enable_sp = enable_sp
WanVideoToVideoPipeline.enable_sp = enable_sp
WanVACEPipeline.enable_sp = enable_sp
WanPipeline.enable_vae_lightning = enable_vae_lightning
WanImageToVideoPipeline.enable_vae_lightning = enable_vae_lightning
WanVACEPipeline.enable_vae_lightning = enable_vae_lightning
WanVideoToVideoPipeline.enable_vae_lightning = enable_vae_lightning

CogVideoXPipeline.enable_sp = enable_sp
CogVideoXPipeline.enable_vae_lightning = enable_vae_lightning

HunyuanVideoPipeline.enable_sp = enable_sp
HunyuanVideoImageToVideoPipeline.enable_sp = enable_sp
HunyuanVideoPipeline.enable_vae_lightning = enable_vae_lightning
HunyuanVideoImageToVideoPipeline.enable_vae_lightning = enable_vae_lightning
