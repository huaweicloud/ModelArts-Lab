#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.abc
import sys

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply vllm-ascend's process-wide patches once per process."""
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from ascend_vllm.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the PATCH NPU platform."""
    return "ascend_vllm.platform.PatchNPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    _ensure_global_patch()

    from vllm_ascend.model_loader.netloader import register_netloader
    from vllm_ascend.model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from vllm_ascend.profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from vllm_ascend.models import register_model

    register_model()


# ---------------------------------------------------------------------------
# Meta-path import hook for reliable patch loading.
#
# Following the ascend-vllm (v6.5.306) pattern: the hook is installed at
# ``ascend_vllm`` import time (when vllm loads the platform plugin) and
# intercepts the import of ``vllm_ascend.ops``, which is imported naturally
# during vllm startup after vllm_ascend's own patches are applied.  Once that
# module is loaded, the hook triggers ``import ascend_vllm.patch.platform`` so
# all platform patches (including cloud_ops_turbo) are applied — regardless of
# whether ``pre_register_and_update`` is called or ``VLLM_PLUGINS`` filters
# general_plugins entry points.
# ---------------------------------------------------------------------------


class _OpsPatchLoader(importlib.abc.Loader):
    def __init__(self, original):
        self._original = original

    def create_module(self, spec):
        return self._original.create_module(spec)

    def exec_module(self, module):
        self._original.exec_module(module)
        if not _OpsPatchHook._done:
            _OpsPatchHook._done = True
            import ascend_vllm.patch.platform  # noqa: F401


class _OpsPatchHook(importlib.abc.MetaPathFinder):
    _target = "vllm_ascend.ops"
    _done = False

    def find_spec(self, name, path, target=None):
        if name == self._target and not self._done:
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    spec.loader = _OpsPatchLoader(spec.loader)
                    return spec
        return None


if not any(isinstance(f, _OpsPatchHook) for f in sys.meta_path):
    sys.meta_path.insert(0, _OpsPatchHook())
if _OpsPatchHook._target in sys.modules and not _OpsPatchHook._done:
    _OpsPatchHook._done = True
    import ascend_vllm.patch.platform  # noqa: F401
