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

from typing import TYPE_CHECKING, Optional

from vllm_ascend.ascend_config import get_ascend_config  # noqa: F401
from vllm_ascend.platform import NPUPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    VllmConfig = None
    FlexibleArgumentParser = None


class PatchNPUPlatform(NPUPlatform):
    @classmethod
    def pre_register_and_update(cls, parser: Optional[FlexibleArgumentParser] = None) -> None:  # noqa: UP045
        super().pre_register_and_update(parser)

        from ascend_vllm.utils import adapt_patch

        adapt_patch(is_global_patch=True)
