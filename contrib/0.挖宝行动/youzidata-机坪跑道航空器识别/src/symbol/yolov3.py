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

"""
Darknet53 as YOLO-V3 backbone network.
"""
from __future__ import absolute_import, division, print_function

import mxnet as mx
import numpy as np
from .bbox import BBoxBatchIOU
from .loss import get_loss

def _conv2d(input_data, channel, kernel, padding, stride, num_sync_bn_devices=-1, idx=1):
    conv_layer = mx.sym.Convolution(data=input_data, kernel=kernel, stride=stride,
                                    pad=padding, num_filter=channel,
                                    no_bias=True, name="conv_%s" % (idx))
    if num_sync_bn_devices < 1:
        bn_layer = mx.symbol.BatchNorm(
            data=conv_layer, eps=1e-5, momentum=0.9, fix_gamma=False,
            __wd_mult__=0.0, name="bn_%s" % (idx))
    else:
        bn_layer = mx.symbol.contrib.SyncBatchNorm(
            data=conv_layer, eps=1e-5, momentum=0.9, fix_gamma=False, __wd_mult__=0.0,
            key="key_%s" % (idx), ndev=num_sync_bn_devices, name="bn_%s" % (idx))
    active_layer = mx.symbol.LeakyReLU(
        data=bn_layer, act_type='leaky', slope=0.1, name='leaky_relu_%s' % (idx))
    return active_layer


def _res_block(input_data, channel, num_sync_bn_devices=-1, idx=1):
    shortcut = input_data
    conv_block_1 = _conv2d(input_data=input_data, channel=channel,
                           kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx)
    conv_block_2 = _conv2d(input_data=conv_block_1, channel=channel * 2,
                           kernel=(3, 3), padding=(1, 1), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 1)
    return conv_block_2 + shortcut


def _feature_block(input_data, channel, num_sync_bn_devices=-1, idx=1):
    conv_block_1 = _conv2d(input_data=input_data, channel=channel,
                           kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx)
    conv_block_2 = _conv2d(input_data=conv_block_1, channel=channel * 2,
                           kernel=(3, 3), padding=(1, 1), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 1)
    conv_block_3 = _conv2d(input_data=conv_block_2, channel=channel,
                           kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 2)
    conv_block_4 = _conv2d(input_data=conv_block_3, channel=channel * 2,
                           kernel=(3, 3), padding=(1, 1), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 3)
    conv_block_5 = _conv2d(input_data=conv_block_4, channel=channel,
                           kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 4)
    conv_block_6 = _conv2d(input_data=conv_block_5, channel=channel * 2,
                           kernel=(3, 3), padding=(1, 1), stride=(1, 1),
                           num_sync_bn_devices=num_sync_bn_devices, idx=idx + 5)
    return conv_block_5, conv_block_6


def get_feature(input_data, num_sync_bn_devices=-1):
    block_0 = _conv2d(input_data=input_data, channel=32,
                      kernel=(3, 3), padding=(1, 1), stride=(1, 1),
                      num_sync_bn_devices=num_sync_bn_devices, idx=0)

    block_1_0 = _conv2d(input_data=block_0, channel=64,
                        kernel=(3, 3), padding=(1, 1), stride=(2, 2),
                        num_sync_bn_devices=num_sync_bn_devices, idx=1)
    block_1_1 = _res_block(block_1_0, 32, num_sync_bn_devices, idx=2)

    block_2_0 = _conv2d(input_data=block_1_1, channel=128,
                        kernel=(3, 3), padding=(1, 1), stride=(2, 2),
                        num_sync_bn_devices=num_sync_bn_devices, idx=4)
    block_2_1 = _res_block(block_2_0, 64, num_sync_bn_devices, idx=5)
    block_2_2 = _res_block(block_2_1, 64, num_sync_bn_devices, idx=7)
    block_3_0 = _conv2d(input_data=block_2_2, channel=256,
                        kernel=(3, 3), padding=(1, 1), stride=(2, 2),
                        num_sync_bn_devices=num_sync_bn_devices, idx=9)
    block_3_1 = _res_block(block_3_0, 128, num_sync_bn_devices, idx=10)
    block_3_2 = _res_block(block_3_1, 128, num_sync_bn_devices, idx=12)
    block_3_3 = _res_block(block_3_2, 128, num_sync_bn_devices, idx=14)
    block_3_4 = _res_block(block_3_3, 128, num_sync_bn_devices, idx=16)
    block_3_5 = _res_block(block_3_4, 128, num_sync_bn_devices, idx=18)
    block_3_6 = _res_block(block_3_5, 128, num_sync_bn_devices, idx=20)
    block_3_7 = _res_block(block_3_6, 128, num_sync_bn_devices, idx=22)
    block_3_8 = _res_block(block_3_7, 128, num_sync_bn_devices, idx=24)

    block_4_0 = _conv2d(input_data=block_3_8, channel=512,
                        kernel=(3, 3), padding=(1, 1), stride=(2, 2),
                        num_sync_bn_devices=num_sync_bn_devices, idx=26)
    block_4_1 = _res_block(block_4_0, 256, num_sync_bn_devices, idx=27)
    block_4_2 = _res_block(block_4_1, 256, num_sync_bn_devices, idx=29)
    block_4_3 = _res_block(block_4_2, 256, num_sync_bn_devices, idx=31)
    block_4_4 = _res_block(block_4_3, 256, num_sync_bn_devices, idx=33)
    block_4_5 = _res_block(block_4_4, 256, num_sync_bn_devices, idx=35)
    block_4_6 = _res_block(block_4_5, 256, num_sync_bn_devices, idx=37)
    block_4_7 = _res_block(block_4_6, 256, num_sync_bn_devices, idx=39)
    block_4_8 = _res_block(block_4_7, 256, num_sync_bn_devices, idx=41)

    block_5_0 = _conv2d(input_data=block_4_8, channel=1024,
                        kernel=(3, 3), padding=(1, 1), stride=(2, 2),
                        num_sync_bn_devices=num_sync_bn_devices, idx=43)
    block_5_1 = _res_block(block_5_0, 512, num_sync_bn_devices, idx=44)
    block_5_2 = _res_block(block_5_1, 512, num_sync_bn_devices, idx=46)
    block_5_3 = _res_block(block_5_2, 512, num_sync_bn_devices, idx=48)
    block_5_4 = _res_block(block_5_3, 512, num_sync_bn_devices, idx=50)

    route_1, feature_1 = _feature_block(
        block_5_4, 512, num_sync_bn_devices, idx=52)

    route_1_1 = _conv2d(input_data=route_1, channel=256,
                        kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                        num_sync_bn_devices=num_sync_bn_devices, idx=58)
    route_1_2 = mx.sym.UpSampling(route_1_1, scale=2, sample_type='nearest')
    route_1_3 = mx.sym.concat(route_1_2, block_4_8, dim=1)
    route_2, feature_2 = _feature_block(
        route_1_3, 256, num_sync_bn_devices, idx=59)

    route_2_1 = _conv2d(input_data=route_2, channel=128,
                        kernel=(1, 1), padding=(0, 0), stride=(1, 1),
                        num_sync_bn_devices=num_sync_bn_devices, idx=65)
    route_2_2 = mx.sym.UpSampling(route_2_1, scale=2, sample_type='nearest')
    route_2_3 = mx.sym.concat(route_2_2, block_3_8, dim=1)
    _, feature_3 = _feature_block(route_2_3, 128, num_sync_bn_devices, idx=66)
    return (feature_1, feature_2, feature_3)


def dynamic_target(num_classes, ignore_iou_thresh, box_preds, gt_boxes):
    box_preds = mx.sym.reshape(data=box_preds, shape=(0, -1, 4))
    objness_t = mx.sym.zeros_like(mx.sym.slice_axis(
        data=box_preds, axis=-1, begin=0, end=1))
    center_t = mx.sym.zeros_like(
        mx.sym.slice_axis(data=box_preds, axis=-1, begin=0, end=2))
    scale_t = mx.sym.zeros_like(
        mx.sym.slice_axis(data=box_preds, axis=-1, begin=0, end=2))
    weight_t = mx.sym.zeros_like(
        mx.sym.slice_axis(data=box_preds, axis=-1, begin=0, end=2))
    class_t = mx.sym.ones_like(mx.sym.tile(
        data=objness_t, reps=(num_classes))) * -1
    batch_ious = BBoxBatchIOU(box_preds, gt_boxes)  # (B, N, M)
    ious_max = mx.sym.max(data=batch_ious, axis=-1, keepdims=True)  # (B, N, 1)
    objness_t = (ious_max > ignore_iou_thresh) * -1  # use -1 for ignored
    return objness_t, center_t, scale_t, weight_t, class_t


def target_generator(num_classes, ignore_iou_thresh, box_preds, gt_boxes,
                     obj_t, centers_t, scales_t, weights_t, clas_t, label_smooth=False):
    dynamic_t = dynamic_target(
        num_classes, ignore_iou_thresh, box_preds, gt_boxes)
    obj, centers, scales, weights, clas = zip(
        dynamic_t, [obj_t, centers_t, scales_t, weights_t, clas_t])
    mask = obj[1] > 0
    objectness = mx.sym.where(mask, obj[1], obj[0])
    mask2 = mx.sym.tile(data=mask, reps=(2,))
    center_targets = mx.sym.where(mask2, centers[1], centers[0])
    scale_targets = mx.sym.where(mask2, scales[1], scales[0])
    weights = mx.sym.where(mask2, weights[1], weights[0])
    mask3 = mx.sym.tile(data=mask, reps=(num_classes,))
    class_targets = mx.sym.where(mask3, clas[1], clas[0])
    smooth_weight = 1. / num_classes
    if label_smooth:
        smooth_weight = 1. / num_classes
        class_targets = mx.sym.where(
            class_targets > 0.5, class_targets - smooth_weight, class_targets)
        class_targets = mx.sym.where(
            class_targets < -0.5, class_targets, mx.sym.ones_like(class_targets) * smooth_weight)
    class_mask = mx.sym.tile(data=mask, reps=(
        num_classes,)) * (class_targets >= 0)
    return [mx.sym.stop_gradient(x) for x in [objectness, center_targets, scale_targets,
                                              weights, class_targets, class_mask]]


def pred_generator(result, nms_thresh=0.45, nms_topk=400, post_nms=100):
    if nms_thresh > 0 and nms_thresh < 1:
        mx_version = mx.__version__
        if mx_version >= '1.3.0':
            result = mx.sym.contrib.box_nms(
                result, overlap_thresh=nms_thresh, valid_thresh=0.01,
                topk=nms_topk, id_index=0, score_index=1,
                coord_start=2, force_suppress=False)
        else:
            result = mx.sym.contrib.box_nms(
                result, overlap_thresh=nms_thresh,
                topk=nms_topk, id_index=0, score_index=1,
                coord_start=2, force_suppress=False)
        if post_nms > 0:
            result = result.slice_axis(axis=1, begin=0, end=post_nms)
    ids = result.slice_axis(axis=-1, begin=0, end=1)
    scores = result.slice_axis(axis=-1, begin=1, end=2)
    bboxes = result.slice_axis(axis=-1, begin=2, end=None)
    return mx.sym.Group([ids, scores, bboxes])


def get_yolo_output(feature, anchors, offsets, num_classes, is_train=True):
    num_pred = 1 + 4 + num_classes
    num_anchors = 3
    all_pred = num_pred*num_anchors
    strdies = [32, 16, 8]
    all_box_centers = []
    all_box_scales = []
    all_objectness = []
    all_class_pred = []
    all_detections = []
    for i in range(num_anchors):
        pred = mx.sym.Convolution(data=feature[i], kernel=(1, 1), stride=(1, 1),
                                  pad=(0, 0), num_filter=all_pred,
                                  no_bias=True, name="conv_output_%s" % (i))
        pred = mx.sym.reshape(data=pred, shape=(0, num_anchors*num_pred, -1))
        pred = mx.sym.transpose(data=pred, axes=(0, 2, 1))
        pred = mx.sym.reshape(data=pred, shape=(0, -1, num_anchors, num_pred))
        raw_box_centers = mx.sym.slice_axis(data=pred, axis=-1, begin=0, end=2)
        raw_box_scales = mx.sym.slice_axis(data=pred, axis=-1, begin=2, end=4)
        objness = mx.sym.slice_axis(data=pred, axis=-1, begin=4, end=5)
        class_pred = mx.sym.slice_axis(data=pred, axis=-1, begin=5, end=None)
        box_centers = mx.sym.broadcast_add(mx.sym.sigmoid(
            data=raw_box_centers), offsets[i]) * strdies[i]
        box_scales = mx.sym.broadcast_mul(
            mx.sym.exp(data=raw_box_scales), anchors[i])
        confidence = mx.sym.sigmoid(data=objness)
        class_score = mx.sym.broadcast_mul(
            mx.sym.sigmoid(data=class_pred), confidence)
        wh = box_scales / 2.0
        bbox = mx.sym.concat(box_centers - wh, box_centers + wh, dim=-1)
        if is_train:
            all_box_centers.append(mx.sym.reshape(
                data=raw_box_centers, shape=(0, -3, -1)))
            all_box_scales.append(mx.sym.reshape(
                data=raw_box_scales, shape=(0, -3, -1)))
            all_objectness.append(mx.sym.reshape(
                data=objness, shape=(0, -3, -1)))
            all_class_pred.append(mx.sym.reshape(
                data=class_pred, shape=(0, -3, -1)))
            all_detections.append(mx.sym.reshape(data=bbox, shape=(0, -1, 4)))
        else:
            bboxes = mx.sym.tile(data=bbox, reps=(num_classes, 1, 1, 1, 1))
            scores = mx.sym.transpose(data=class_score, axes=(3, 0, 1, 2))
            scores = mx.sym.expand_dims(data=scores, axis=-1)
            ids = mx.sym.broadcast_add(
                scores * 0, mx.sym.reshape(data=mx.sym.arange(0, num_classes), shape=(0, 1, 1, 1, 1)))
            detections = mx.sym.concat(ids, scores, bboxes, dim=-1)
            detections = mx.sym.transpose(
                data=detections, axes=(1, 0, 2, 3, 4))
            detections = mx.sym.reshape(data=detections, shape=(0, -1, 6))
            all_detections.append(detections)
    if is_train:
        box_preds = mx.sym.concat(*all_detections, dim=1)
        all_preds = [mx.sym.concat(*p, dim=1) for p in [
            all_objectness, all_box_centers, all_box_scales, all_class_pred]]
        return box_preds, all_preds
    else:
        result = mx.sym.concat(*all_detections, dim=1)
        return None, result


def get_symbol(num_classes, dtype='float32', ignore_iou_thresh=0.7,
               nms_thresh=0.45, nms_topk=400, post_nms=100, label_smooth=False,
               num_sync_bn_devices=-1, is_train=True, img_shape=416,
               offsets=[(13, 13), (26, 26), (52, 52)],
               anchors = [[116, 90, 156, 198, 373, 326],
                          [30, 61, 62, 45, 59, 119],
                          [10, 13, 16, 30, 33, 23]], **kwargs):
    """
    Parameters
    ----------
    num_classes : int, default 1000
        Number of classification classes.
    dtype: str, float32 or float16
        Data precision.
    """
    data = mx.sym.Variable(name="data")
    offset_shape_0 = (1, offsets[0][0] * offsets[0][1], 1, 2)
    offset_shape_1 = (1, offsets[1][0] * offsets[1][1], 1, 2)
    offset_shape_2 = (1, offsets[2][0] * offsets[2][1], 1, 2)
    anchor_shape = (1, 1, len(anchors[0]) // 2, 2)
    offset_0 = mx.sym.Variable(name="offset_0_weight", shape=offset_shape_0, lr_mult=0.0)
    offset_1 = mx.sym.Variable(name="offset_1_weight", shape=offset_shape_1, lr_mult=0.0)
    offset_2 = mx.sym.Variable(name="offset_2_weight", shape=offset_shape_2, lr_mult=0.0)
    anchor_0 = mx.sym.Variable(name="anchors_0_weight", shape=anchor_shape, lr_mult=0.0)
    anchor_1 = mx.sym.Variable(name="anchors_1_weight", shape=anchor_shape, lr_mult=0.0)
    anchor_2 = mx.sym.Variable(name="anchors_2_weight", shape=anchor_shape, lr_mult=0.0)
    gt_boxes = mx.sym.Variable(name="gt_boxes")

    if dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)
    feature = get_feature(data, num_sync_bn_devices)
    feature = [mx.sym.Cast(data=feature[i], dtype=np.float32)
               for i in range(3)]
    box_preds, all_preds = get_yolo_output(feature,
                                           (anchor_0, anchor_1, anchor_2),
                                           (offset_0, offset_1, offset_2),
                                           num_classes=num_classes,
                                           is_train=is_train)
    if is_train:
        obj_t = mx.sym.Variable(name="obj_t")
        centers_t = mx.sym.Variable(name="centers_t")
        scales_t = mx.sym.Variable(name="scales_t")
        weights_t = mx.sym.Variable(name="weights_t")
        clas_t = mx.sym.Variable(name="clas_t")
        all_targets = target_generator(num_classes, ignore_iou_thresh, box_preds,
                                       gt_boxes, obj_t, centers_t, scales_t,
                                       weights_t, clas_t, label_smooth)
        denorm = (img_shape // 8)**2 + (img_shape // 16)**2 + (img_shape // 32)**2
        denorm = denorm * 3.0
        denorm_class = denorm * num_classes
        symbol = get_loss(*(all_preds + all_targets + [denorm, denorm_class]))
    else:
        symbol = pred_generator(all_preds, nms_thresh, nms_topk, post_nms)
    return symbol
