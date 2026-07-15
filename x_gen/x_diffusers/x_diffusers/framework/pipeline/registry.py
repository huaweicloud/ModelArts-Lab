import json
import os

import torch
from diffusers.utils import logging

_CLASSNAME_TO_PIPELINE: dict[str, type] = {}

logger = logging.get_logger(__name__)


def _read_model_index(model_path: str) -> dict:
    fp = os.path.join(model_path, "model_index.json")
    if not os.path.exists(fp):
        return dict()
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def register_hf_pipeline_class(hf_class_name: str):
    """
    把 model_index.json 的 _class_name 映射到自定义 pipeline 类
    """

    def deco(cls):
        if hf_class_name in _CLASSNAME_TO_PIPELINE:
            raise ValueError(f"Duplicate registration for {hf_class_name}")
        _CLASSNAME_TO_PIPELINE[hf_class_name] = cls
        return cls

    return deco


def get_pipeline_cls_by_hf_class_name(model_id: str, torch_dtype: torch.dtype, **kwargs) -> type | None:
    model_index = _read_model_index(model_id)
    hf_class_name = model_index.get("_class_name", "")
    if not hf_class_name:
        logger.error(f"_class_name missing in {model_id}/model_index.json")  # noqa: G004
        raise ValueError(f"_class_name missing in {model_id}/model_index.json")

    pipeline = _CLASSNAME_TO_PIPELINE.get(hf_class_name)
    if not pipeline:
        logger.error(f"Pipeline class '{hf_class_name}' not registered")  # noqa: G004
        raise ValueError(f"Pipeline class '{hf_class_name}' not registered")

    pipeline = pipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    return pipeline


def is_registered_mappings(model_id):
    model_index = _read_model_index(model_id)
    hf_class_name = model_index.get("_class_name", "")
    if not hf_class_name:
        return False

    return hf_class_name in _CLASSNAME_TO_PIPELINE
