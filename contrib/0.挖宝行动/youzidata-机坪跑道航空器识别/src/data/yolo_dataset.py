# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import, division, print_function

import os

import numpy as np
from moxing.framework import file
from data.yolo_load.detection_dataset import Detection_dataset
from utils.read_image_to_list import get_image_list
from mxnet import gluon, io, nd


def _pad_arrs_to_max_length(arrs, max_gt_box_number, pad_axis=0, pad_val=-1):
    """Inner Implementation of the Pad batchify"""
    if not isinstance(arrs[0], (nd.NDArray, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    max_size = max_gt_box_number
    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)
    ret = nd.full(shape=ret_shape, val=pad_val, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            slices = [slice(i, i + 1)] + slices
            ret[tuple(slices)] = arr
    return ret


class _train_batchify_fn(object):
    def __init__(self, max_gt_box_number):
        self._max_gt_box_number = max_gt_box_number

    def __call__(self, data):
        """Collate train data into batch."""
        img_data = nd.stack(*[item[0] for item in data])
        center_targets = nd.stack(*[item[1] for item in data])
        scale_targets = nd.stack(*[item[2] for item in data])
        weights = nd.stack(*[item[3] for item in data])
        objectness = nd.stack(*[item[4] for item in data])
        class_targets = nd.stack(*[item[5] for item in data])
        gt_bboxes = _pad_arrs_to_max_length([item[6] for item in data],
                                            self._max_gt_box_number,
                                            pad_axis=0, pad_val=-1)
        batch_data = io.DataBatch(data=[img_data],
                                  label=[gt_bboxes, objectness, center_targets,
                                         scale_targets, weights, class_targets])
        return batch_data


class _val_batchify_fn(object):
    def __init__(self, max_gt_box_number):
        self._max_gt_box_number = max_gt_box_number

    def __call__(self, data):
        """Collate train data into batch."""
        img_data = nd.stack(*[item[0] for item in data])
        gt_bboxes = _pad_arrs_to_max_length([item[1] for item in data],
                                            self._max_gt_box_number,
                                            pad_axis=0, pad_val=-1)
        batch_data = io.DataBatch(data=[img_data],
                                  label=[gt_bboxes])
        return batch_data


def _get_provide_data(next_batch):
    next_data = next_batch.data
    return [io.DataDesc(name='data', shape=next_data[0].shape)]


def _get_provide_label(next_batch, gt_boxes_shape=(32, 56, 4), is_train=True):
    next_label = next_batch.label
    if is_train:
        provide_label = [io.DataDesc(name='gt_boxes',
                                     shape=next_label[0].shape),
                         io.DataDesc(name='obj_t', shape=next_label[1].shape),
                         io.DataDesc(name='centers_t',
                                     shape=next_label[2].shape),
                         io.DataDesc(name='scales_t',
                                     shape=next_label[3].shape),
                         io.DataDesc(name='weights_t',
                                     shape=next_label[4].shape),
                         io.DataDesc(name='clas_t', shape=next_label[5].shape)]
    else:
        provide_label = None
    return provide_label


def _reset():
    pass


def get_data_iter(data_path, train_file=None, val_file=None, split_spec=1,
                  hyper_train={}, hyper_val={}, **kwargs):
    train_set = None
    val_set = None
    train_list = None
    val_list = None
    if train_file is not None:
        assert file.exists(train_file), 'not found train file'
        train_path = file.read(train_file).split("\n")[0:-1]
        train_list = [path.replace('\r', '').split(' ') for path in train_path]
        train_list = [[os.path.join(data_path, path[0]),
                       os.path.join(data_path, path[1])] for path in train_list]
    if val_file is not None:
        assert file.exists(val_file), 'not found val file'
        val_path = file.read(val_file).split("\n")[0:-1]
        val_list = [path.replace('\r', '').split(' ') for path in val_path]
        val_list = [[os.path.join(data_path, path[0]),
                     os.path.join(data_path, path[1])] for path in val_list]
    if train_file is None and val_file is None:
        train_list, val_list, _ = get_image_list(data_path, split_spec)
    if 'anchors' not in kwargs:
        kwargs['anchors'] = [[116, 90, 156, 198, 373, 326],
                             [30, 61, 62, 45, 59, 119],
                             [10, 13, 16, 30, 33, 23]]
    if 'offsets' not in kwargs:
        kwargs['offsets'] = [(13, 13), (26, 26), (52, 52)]
    if train_list is not None and len(train_list) > 0:
        dataset = Detection_dataset(img_list=train_list,
                                    index_file=hyper_train.get(
                                        'index_file', None),
                                    width=hyper_train.get('width', 416),
                                    height=hyper_train.get('height', 416),
                                    is_train=True,
                                    ** kwargs)
        max_gt_box_number = max([len(item) for item in dataset.label_cache])
        batch_size = hyper_train.get('batch_size', 32)
        train_set = gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=hyper_train.get('shuffle', True),
            batchify_fn=_train_batchify_fn(max_gt_box_number),
            last_batch='rollover',
            num_workers=hyper_train.get('preprocess_threads', 4))
        next_data_batch = next(iter(train_set))
        setattr(train_set, 'reset', _reset)
        setattr(train_set, 'provide_data', _get_provide_data(next_data_batch))
        setattr(train_set, 'provide_label', _get_provide_label(
            next_data_batch, (batch_size, max_gt_box_number, 4), is_train=True))
    if val_list is not None and len(val_list) > 0:
        assert 'index_file' in hyper_val and file.exists(
            hyper_val['index_file']), 'not found label name file'
        dataset = Detection_dataset(img_list=val_list,
                                    index_file=hyper_val.get(
                                        'index_file'),
                                    width=hyper_val.get('width', 416),
                                    height=hyper_val.get('height', 416),
                                    is_train=False,
                                    ** kwargs)
        max_gt_box_number = max([len(item) for item in dataset.label_cache])
        batch_size = hyper_val.get('batch_size', 32)
        val_set = gluon.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=hyper_val.get('shuffle', True),
            batchify_fn=_val_batchify_fn(max_gt_box_number),
            last_batch='keep',
            num_workers=hyper_val.get('preprocess_threads', 4))
        next_data_batch = next(iter(val_set))
        setattr(val_set, 'reset', _reset)
        setattr(val_set, 'provide_data', _get_provide_data(next_data_batch))
        setattr(val_set, 'provide_label', _get_provide_label(
            next_data_batch, is_train=False))
    return train_set, val_set
