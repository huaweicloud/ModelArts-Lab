"""
Cache 加速模型实现

按模型拆分，每个模型文件包含 TeaCache 和 MagCache 的 forward 实现。
"""

from .cogvideox import (
    teacache_cogvideox_forward,
)
from .cogvideox import (
    teacache_init as teacache_cogvideox_init,
)
from .hunyuan import (
    teacache_hunyuan_forward,
)
from .hunyuan import (
    teacache_init as teacache_hunyuan_init,
)
from .wan import (
    magcache_calibration as magcache_wan_calibration_init,
)
from .wan import (
    magcache_init as magcache_wan_init,
)
from .wan import (
    magcache_wan_calibration,
    magcache_wan_forward,
    teacache_wan_forward,
    teacache_wan_vace_forward,
)
from .wan import (
    teacache_init as teacache_wan_init,
)

__all__ = [
    # Wan
    "teacache_wan_forward",
    "teacache_wan_vace_forward",
    "magcache_wan_forward",
    "magcache_wan_calibration",
    "teacache_wan_init",
    "magcache_wan_init",
    "magcache_wan_calibration_init",
    # Hunyuan
    "teacache_hunyuan_forward",
    "teacache_hunyuan_init",
    # CogVideoX
    "teacache_cogvideox_forward",
    "teacache_cogvideox_init",
]
