import os
from enum import Enum

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from vllm.entrypoints.serve.instrumentator.health import engine_client, logger, router
from vllm.v1.engine.exceptions import EngineDeadError


def get_npu_status_path() -> str:
    return os.getenv("NPU_STATUS_FILE_PATH", "/opt/cloud/node/npu_status.yaml")


class NPUStatusInfo(Enum):
    """NPU 状态信息枚举"""

    HEALTH = 0  # 健康状态
    UNKNOWN = 1  # 未知状态
    NPU_ERROR = 2  # NPU 错误
    SWITCH_ERROR = 3  # 交换机错误


@router.get("/health", response_class=Response)
async def custom_health(raw_request: Request) -> Response:
    """
    Health check.
    Returns:
        - 200: All checks passed.
        - 500:
            - {"fault message": "unhealthy"} if engine is dead.
            - {"fault message": "npu error"} if npu errLevel is L3/L4/L5.
            - {"fault message": "L1-1520 err"} if L1-1520 errLevelName is PreSeparate/Separate.
    """
    try:
        # Step 1: Check engine health
        await engine_client(raw_request).check_health()

        # Step 2: Check npu_status.yaml
        npu_status = check_npu_status()
        if npu_status == NPUStatusInfo.NPU_ERROR:
            return JSONResponse(content={"fault message": "npu error"}, status_code=500)
        if npu_status == NPUStatusInfo.SWITCH_ERROR:
            return JSONResponse(content={"fault message": "L1-1520 err"}, status_code=500)

        return JSONResponse(content={"status": "healthy"}, status_code=200)
    except EngineDeadError:
        return JSONResponse(content={"status": "unhealthy"}, status_code=500)


def npu_error(npu):
    err_codes_str = ",".join(map(str, npu.get("errCodes", [])))
    err_messages_str = ",".join(map(str, npu.get("errMessages", [])))
    err_time = npu.get("errTime", "")
    logger.error(
        f"npu fault occurs, errCodes: {err_codes_str} "
        f"errLevel: {npu['errLevel']} "
        f"errLevelName: {npu.get('errLevelName', '')} "
        f"errMessages: {err_messages_str} "
        f"errTime: {err_time} "
        f"health: {npu.get('health', '')} "
        f"name: {npu.get('name', '')}"
    )
    return NPUStatusInfo.NPU_ERROR


def get_err_codes_set(npu):
    errCodes = list(map(str, npu.get("errCodes", [])))
    return {code.lower().strip() for code in errCodes if code and code.strip()}


def _load_npu_status_file(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, encoding="utf-8") as file:
            import yaml

            return yaml.safe_load(file)
    except Exception as e:
        logger.warning(f"check npu_status.yaml failed: {e}")
        return None


def _validate_npu_status_structure(npu_status):
    """
    验证 npu_status 数据结构是否符合预期格式
    """
    # 1. 顶层结构校验：必须是字典，且包含 resources 和 version
    if not isinstance(npu_status, dict):
        logger.warning("npu_status root is not a dictionary.")
        return False

    if "resources" not in npu_status or "version" not in npu_status:
        logger.warning("Missing 'resources' or 'version' field in root.")
        return False

    # 2. resources 字段校验：必须是非空列表
    resources = npu_status["resources"]
    if not isinstance(resources, list) or len(resources) == 0:
        logger.warning("'resources' must be a non-empty list.")
        return False

    # 3. 遍历 resources 列表，逐个校验内部元素
    for index, resource in enumerate(resources):
        # 确保每一项都是字典
        if not isinstance(resource, dict):
            logger.warning(f"Resource at index {index} is not a valid dictionary.")
            return False

        # 【强制拦截】每个资源必须包含 type 字段，且为字符串
        if "type" not in resource:
            logger.warning(f"Missing or invalid 'type' string in resource at index {index}.")
            return False

        resource_type = resource["type"]

        # 4. 根据 type 的类型，进行针对性的分支校验
        if resource_type == "NPU":
            # NPU 类型必须包含 status 字段
            if "status" not in resource:
                logger.warning(f"NPU resource at index {index} is missing 'status' field.")
                return False

            status_list = resource["status"]
            # status 必须是列表（数量不限，哪怕是空列表也符合格式要求）
            if not isinstance(status_list, list):
                logger.warning(f"'status' in NPU resource at index {index} is not a list.")
                return False

    return True


def _get_allow_lists():
    """获取白名单配置"""
    npu_health_level_allow_list = set()
    npu_health_level_allow_str: str = os.getenv("VLLM_HEALTH_NPU_ERRLEVEL_WHITELIST", "L0,L1").lower()
    if npu_health_level_allow_str != "-1":
        npu_health_level_allow_list = set(npu_health_level_allow_str.split(","))
        npu_health_level_allow_list = {
            npu_health_level.strip()
            for npu_health_level in npu_health_level_allow_list
            if npu_health_level and npu_health_level.strip()
        }

    error_code_allow_list = set()
    error_code_allow_str: str = os.getenv("VLLM_HEALTH_NPU_ERRCODE_WHITELIST", "").lower()
    if error_code_allow_str != "-1":
        error_code_allow_list = set(error_code_allow_str.split(","))
        error_code_allow_list = {
            error_code.strip() for error_code in error_code_allow_list if error_code and error_code.strip()
        }

    return npu_health_level_allow_list, error_code_allow_list


def _check_npu_resource(npu, npu_health_level_allow_list, error_code_allow_list):
    """检查单个 NPU 资源"""
    if "errLevel" not in npu or "errCodes" not in npu:
        logger.warning("[resources.status.errLevel/errCodes] does not exist in npu_status.yaml")
        return NPUStatusInfo.UNKNOWN

    npu_errLevel = npu["errLevel"].lower()
    if npu_errLevel in npu_health_level_allow_list:
        return NPUStatusInfo.HEALTH

    err_codes_set = get_err_codes_set(npu)
    if err_codes_set <= error_code_allow_list:
        return NPUStatusInfo.HEALTH

    return npu_error(npu)


def _check_l1_1520_resource(resource):
    """检查 L1-1520 资源"""
    err_level_name = resource.get("errLevelName")
    if err_level_name is None:
        logger.warning("errLevelName does not exist in L1-1520 resource")
        return NPUStatusInfo.UNKNOWN
    if err_level_name != "NotHandle":
        logger.error(f"L1-1520 fault occurs, errLevelName: {err_level_name}")
        return NPUStatusInfo.SWITCH_ERROR
    return NPUStatusInfo.HEALTH


def _check_all_resources(resources, npu_health_level_allow_list, error_code_allow_list):
    """检查所有资源"""
    for resource in resources:
        check_res = _single_resource_check(resource, npu_health_level_allow_list, error_code_allow_list)
        if check_res != NPUStatusInfo.HEALTH:
            return check_res
    return NPUStatusInfo.HEALTH


def _single_resource_check(resource, npu_health_level_allow_list, error_code_allow_list):
    if resource["type"].lower() == "npu":
        status = resource.get("status", [])
        for npu in status:
            res = _check_npu_resource(npu, npu_health_level_allow_list, error_code_allow_list)
            if res != NPUStatusInfo.HEALTH:
                return res
    elif resource["type"].lower() == "l1-1520":
        return _check_l1_1520_resource(resource)
    return NPUStatusInfo.HEALTH


def check_npu_status() -> int:
    file_path = get_npu_status_path()

    # 加载文件
    npu_status = _load_npu_status_file(file_path)
    if npu_status is None:
        return NPUStatusInfo.HEALTH if not os.path.exists(file_path) else NPUStatusInfo.UNKNOWN

    # 验证结构
    if not _validate_npu_status_structure(npu_status):
        return NPUStatusInfo.UNKNOWN

    # 获取白名单
    npu_health_level_allow_list, error_code_allow_list = _get_allow_lists()

    # 检查所有资源
    resources = npu_status.get("resources", [])
    return _check_all_resources(resources, npu_health_level_allow_list, error_code_allow_list)


def apply_health_patches() -> None:
    for route in router.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            if route.path == "/health" and "GET" in route.methods:
                # A. 替换处理函数
                route.endpoint = custom_health
                logger.info("Successfully patched /health route with custom NPU logic.")
                return

    logger.warning(" Target route /health not found. Patch failed.")


apply_health_patches()
