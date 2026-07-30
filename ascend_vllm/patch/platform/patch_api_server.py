from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai import api_server
from vllm.logger import init_logger

from ascend_vllm.patch.platform.patch_image import InvalidMediaInputError

logger = init_logger(__name__)

_original_build_app = api_server.build_app


async def invalid_media_input_exception_handler(
    request: Request,
    exc: InvalidMediaInputError,
) -> JSONResponse:
    logger.warning(
        "Invalid media input: path=%s, detail=%s",
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "invalid_media_input",
            }
        },
    )


def build_app(*args, **kwargs) -> FastAPI:
    app = _original_build_app(*args, **kwargs)
    app.add_exception_handler(
        InvalidMediaInputError,
        invalid_media_input_exception_handler,
    )
    return app


api_server.build_app = build_app
