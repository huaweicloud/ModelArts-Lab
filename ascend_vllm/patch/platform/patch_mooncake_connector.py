# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.logger import logger
from vllm.v1.request import RequestStatus
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorScheduler,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


def request_finished(
    self,
    request: "Request",
    block_ids: BlockIds,
) -> tuple[bool, dict[str, Any] | None]:
    """
    Once a request is finished, determine whether request blocks
    should be freed now or will be sent asynchronously and freed later.
    """

    params = request.kv_transfer_params
    logger.debug("MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params)

    # adapt begin
    # Original code only allowed FINISHED_LENGTH_CAPPED, which dropped
    # kv_transfer_params for FINISHED_STOPPED requests. Allow both statuses
    # so that EOS / stop_token_ids on the first generated token still
    # forwards KV blocks to decode.
    if (
        params is None
        or not params.get("do_remote_decode")
        or request.status
        not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        )
    ):
        return False, None
    # adapt end

    num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
    computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
    computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
    computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
    delay_free_blocks = sum(computed_block_lens) > 0
    if delay_free_blocks:
        logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
        self._reqs_need_send[request.request_id] = time.time()

    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=computed_block_ids,
        remote_engine_id=self.engine_id,
        remote_request_id=request.request_id,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        remote_pcp_size=self.pcp_size,
        remote_dcp_size=self.dcp_size,
        remote_ptp_size=self.tp_size,
        last_token_id=request.output_token_ids[-1],
        remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
        num_prompt_blocks=num_prompt_blocks,
        remote_block_size=self.block_size,
    )


MooncakeConnectorScheduler.request_finished = request_finished
