import json
import os
from collections import defaultdict
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse, OpenAIBaseModel
from vllm.logger import init_logger
from vllm_ascend import envs as envs_ascend

logger = init_logger("vllm.entrypoints.middleware")

TYPE_MAPPING = {"int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict}
DEFAULT_MAX_MODEL_LEN = 8192
ACTION = os.environ.get("ROLE", "")
NOT_ALLOWED_COMPLETIONS = os.environ.get("NOT_ALLOWED_COMPLETIONS", "")


def is_trace_log_enabled() -> bool:
    trace_log_enabled = getattr(envs_ascend, "ENABLE_TRACE_LOG", None)
    if trace_log_enabled is not None:
        return bool(trace_log_enabled)

    return os.getenv("ENABLE_TRACE_LOG", "0").lower() in ("1", "true", "yes", "on")


class BaseValidator:
    def __init__(self, param_name: str, error_msg: str | None = None):
        self.param_name = param_name
        self.error_msg = error_msg

    def validate(self, value: Any) -> str | None:
        pass

    def validate_json(self, value: Any) -> str | None:
        pass


class NestedBaseValidator(BaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: str | None = None,
        subfield: list[str] | None = None,
        checker_condition=None,
        checker: Callable[[str, Any], tuple[str | None, Any | None]] | None = None,
        skip_check_subfield: list | None = None,
    ):
        super().__init__(param_name, error_msg)
        self.subfield = [] if subfield is None else subfield
        self.checker_condition = checker_condition
        self.checker = checker
        self.skip_check_subfield = [] if skip_check_subfield is None else skip_check_subfield

    def validate(self, value):
        if not self.subfield:
            return None
        return self.check_field(value, self.param_name)

    def check_field(self, value, param_name: str) -> str | None:
        if isinstance(value, dict):
            return self.check_dict_subfield(value, param_name)
        if isinstance(value, list):
            return self.check_list_subfield(value, param_name)
        return None

    def check_dict_subfield(self, value, cur_param: str) -> str | None:
        if cur_param in self.skip_check_subfield:
            return None
        for name, val in list(value.items()):
            sub_cur_param = f"{cur_param}.{name}"
            if self.checker_condition and self.checker_condition(sub_cur_param):
                if self.checker:
                    if err_str := self.checker(sub_cur_param, value):
                        return err_str
            elif isinstance(val, list | dict):
                err_str = self.check_field(val, sub_cur_param)
                if err_str:
                    return err_str
        return None

    def check_list_subfield(self, value, cur_param: str) -> str | None:
        for val in value:
            if isinstance(val, dict):
                err_str = self.check_dict_subfield(val, cur_param)
                if err_str:
                    return err_str
        return None


class SupportedValidator(NestedBaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: str | None = None,
        subfield: list[str] | None = None,
        skip_check_subfield: list | None = None,
    ):
        def checker_condition(param_name: str):
            return param_name not in self.subfield

        def checker(param_name: str, value: Any):
            value.pop(param_name.split(".")[-1], None)
            return None

        super().__init__(param_name, error_msg, subfield, checker_condition, checker, skip_check_subfield)


class NestedValueValidator(NestedBaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: str | None = None,
        subfield: list[str] | None = None,
        target_values: list[str] | None = None,
    ):
        self.target_values = [] if target_values is None else target_values

        def checker_condition(param_name: str):
            return param_name in self.subfield

        def checker(param_name: str, value: Any):
            if value[param_name.split(".")[-1]] not in self.target_values:
                return f"{param_name} only support the value in {self.target_values}"
            return None

        super().__init__(param_name, error_msg, subfield, checker_condition, checker)


class IncompatibilityValidator(BaseValidator):
    def __init__(self, param_name: str, error_msg: str | None = None, subfield: list[str] | None = None):
        super().__init__(param_name, error_msg)
        self.subfield = [] if subfield is None else subfield

    def validate_json(self, request_json):
        for param_name in self.subfield:
            request_json.pop(param_name, None)
        return None


class RangeValidator(BaseValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: str | None = None,
        min_val: float | int | None = None,
        max_val: float | int | None = None,
        type_: type | None = None,
    ):
        super().__init__(param_name, error_msg)
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_

    def validate(self, value: Any) -> str | None:
        if self.type_:
            try:
                value_trans = self.type_(value)
            except (ValueError, TypeError):
                return (
                    self.error_msg
                    or f"The type of `{self.param_name}` must belong to {self.type_.__name__}, "
                    f"but got {type(value).__name__!r}"
                )
            if self.min_val is not None and value_trans < self.min_val:
                return (
                    self.error_msg or f"`{self.param_name}` must be greater than {self.min_val},but got {value_trans}."
                )
            if self.max_val is not None and value_trans > self.max_val:
                return (
                    self.error_msg or f"`{self.param_name}` must be smaller than {self.max_val},but got {value_trans}."
                )
        return None


class ValueValidator(SupportedValidator):
    def __init__(
        self,
        param_name: str,
        error_msg: str | None = None,
        subfield: list[str] | None = None,
        target_value: list | None = None,
    ):
        super().__init__(param_name, error_msg, subfield)
        self.target_value = [] if target_value is None else target_value
        self.error_msg = self.error_msg or f"`{self.param_name}` only support the value in {self.target_value}"

    def validate(self, value):
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            return self.error_msg
        return None

    def validate_json(self, request_json):
        value = request_json[self.param_name]
        if error := super().validate(value):
            return error
        if value not in self.target_value:
            request_json.pop(self.param_name, None)
        return None


def create_validator(param_name: str, config: dict[str, Any]) -> BaseValidator | None:
    validator_type = config.get("validator_type")

    if validator_type == "supported":
        return SupportedValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            skip_check_subfield=config.get("skip_check_subfield", []),
        )

    elif validator_type == "incompatibility":
        return IncompatibilityValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
        )

    elif validator_type == "value":
        return ValueValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            target_value=config.get("target_value", []),
        )

    elif validator_type == "range":
        type_str = config.get("type_")
        if type_str and type_str not in TYPE_MAPPING:
            raise ValueError(f"Only supported type: {TYPE_MAPPING.keys()}")

        if type_str is None:
            raise ValueError("`type_` attribute is required in RangeValidator.")

        return RangeValidator(
            param_name=config.get("param_name", param_name),
            min_val=config.get("min_val"),
            max_val=config.get("max_val"),
            type_=TYPE_MAPPING.get(type_str),
        )

    elif validator_type == "nested_value":
        return NestedValueValidator(
            param_name=config.get("param_name", param_name),
            error_msg=config.get("error_msg"),
            subfield=config.get("subfield", []),
            target_values=config.get("target_value", []),
        )

    else:
        raise ValueError(f"Unknown validator type: {validator_type}")


def load_validators_from_json(config_path: str) -> tuple[dict[str, BaseValidator], dict[str, BaseValidator]]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    validators = defaultdict(list)
    validators_json = defaultdict(list)

    # load validators
    for param_name, validator_config in config.get("validators", {}).items():
        if not isinstance(validator_config, list):
            validator_config = [validator_config]
        for cfg in validator_config:
            validator = create_validator(param_name, cfg)
            if validator:
                validators[param_name].append(validator)

    # load validators_json
    for param_name, validator_config in config.get("validators_json", {}).items():
        if not isinstance(validator_config, list):
            validator_config = [validator_config]
        for cfg in validator_config:
            validator = create_validator(param_name, cfg)
            if validator:
                validators_json[param_name].append(validator)

    return validators, validators_json


VALIDATORS, VALIDATORS_JSON = load_validators_from_json(os.getenv("VALIDATORS_CONFIG_PATH", ""))


class ValidateSamplingParams(BaseHTTPMiddleware):
    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ):
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(error=ErrorInfo(message=message, type=err_type, code=status_code.value)).model_dump(),
        )

    def replace_with_stars(self, text):
        return "*" * len(text)

    async def log_response_header_and_usage(self, request: Request, call_next):
        response: Response = await call_next(request)

        if not is_trace_log_enabled():
            return response

        x_span_id = request.headers.get("x-span-id", "")

        x_user_alias = request.headers.get("x-user-alias", "")

        request_metadata = getattr(request.state, "request_metadata", None)
        if request_metadata is None:
            return response
        else:
            if isinstance(response, OpenAIBaseModel):
                response.headers["x-span-id"] = x_span_id
                response.headers["x_user_alias"] = x_user_alias
                trace_id = request_metadata.request_id

                if request_metadata.final_usage_info is not None:
                    logger.info(
                        'x-span-id=%s|x-user-alias=%s|CompletionMetric:{"trace_id":"%s","num_prompt_tokens":%d,"num_output_tokens":%d,"ttft":%.2f,"tpot":%.2f,"latency":%.2f}',
                        x_span_id,
                        x_user_alias,
                        trace_id,
                        request_metadata.final_usage_info.prompt_tokens,
                        request_metadata.final_usage_info.completion_tokens,
                        request_metadata.final_usage_info.ttft,
                        request_metadata.final_usage_info.tpot,
                        request_metadata.final_usage_info.latency,
                    )
            else:
                # adapt stream
                async def log_streaming_response():
                    try:
                        async for chunk in response.body_iterator:
                            yield chunk
                    except Exception as e:
                        logger.warning("failed to obtain the element of response.body_iterator:%s", e)
                    finally:
                        # Always print trace_log, even when response.body_iterator cannot be obtained.
                        response.headers["x-span-id"] = x_span_id
                        response.headers["x_user_alias"] = x_user_alias
                        trace_id = request_metadata.request_id

                        if request_metadata.final_usage_info is not None:
                            logger.info(
                                'x-span-id=%s|x-user-alias=%s|CompletionMetric:{"trace_id":"%s","num_prompt_tokens":%d,"num_output_tokens":%d,"ttft":%.2f,"tpot":%.2f,"latency":%.2f}',
                                x_span_id,
                                x_user_alias,
                                trace_id,
                                request_metadata.final_usage_info.prompt_tokens,
                                request_metadata.final_usage_info.completion_tokens,
                                request_metadata.final_usage_info.ttft,
                                request_metadata.final_usage_info.tpot,
                                request_metadata.final_usage_info.latency,
                            )

                return StreamingResponse(
                    log_streaming_response(),
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type,
                )

        return response

    def validator_check(self, json_load):
        for param_name, value in list(json_load.items()):
            validators = VALIDATORS.get(param_name)
            if not validators:
                json_load.pop(param_name, None)
                logger.warning(
                    "%s is not supported right now. Ascend-vllm will ignore it and change to default value.",
                    param_name,
                )
                continue
            for validator in validators:
                if error_message := validator.validate(value):
                    return self.create_error_response(str(error_message))
            if validators := VALIDATORS_JSON.get(param_name):
                for validator in validators:
                    if error_message := validator.validate_json(json_load):
                        return self.create_error_response(str(error_message))
        return None

    async def dispatch(self, request: Request, call_next):
        if NOT_ALLOWED_COMPLETIONS and request.method == "POST" and request.url.path in "/v1/completions":
            error_message = (
                "The /v1/completions endpoint is not supported by this deployment. "
                "Please use /v1/chat/completions instead."
            )
            return self.create_error_response(error_message, "invalid_request_error", HTTPStatus.METHOD_NOT_ALLOWED)

        if request.method == "POST" and request.url.path in ("/v1/completions", "/v1/chat/completions"):
            body = await request.body()
            if not body:
                return await self.log_response_header_and_usage(request, call_next)

            try:
                json_load = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return await self.log_response_header_and_usage(request, call_next)

            if ACTION != "prefill":
                max_tokens = json_load.get("max_tokens", None)
                if max_tokens is None:
                    json_load["max_tokens"] = int(os.getenv("DEFAULT_MAX_TOKENS", DEFAULT_MAX_MODEL_LEN))

            if not VALIDATORS:
                return await self.log_response_header_and_usage(request, call_next)

            if error_message := self.validator_check(json_load):
                return error_message

            if ACTION == "prefill":
                json_load["max_tokens"] = 1
            request._body = json.dumps(json_load).encode("utf-8")

            headers = dict(request.headers)
            if "x-request-id" in headers:
                request_id = headers["x-request-id"]
                request_id = f"cmpl-{request_id}"
                if "chat" in request.url.path:
                    request_id = f"chat{request_id}"

                logger.info("[Begin %s] request_id: %s", ACTION, request_id)
                response = await self.log_response_header_and_usage(request, call_next)
                logger.info("[End %s] request_id: %s", ACTION, request_id)
                return response

        if request.method == "GET" and request.url.path == "/v1/models":
            response = await call_next(request)
            chunk = await anext(response.body_iterator)
            chunk_json = json.loads(chunk.decode("utf-8"))

            if chunk_json is not None and len(chunk_json.get("data", [])) > 0 and chunk_json.get("data")[0].get("root"):
                chunk_json.get("data")[0]["root"] = self.replace_with_stars(chunk_json.get("data")[0].get("root"))

            new_json_str = json.dumps(chunk_json, ensure_ascii=False)
            new_chunk = new_json_str.encode("utf-8")

            return Response(
                content=new_chunk, headers={"Content-Length": str(len(new_chunk)), "content-type": "application/json"}
            )

        return await self.log_response_header_and_usage(request, call_next)
