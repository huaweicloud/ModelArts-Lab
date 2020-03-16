# -*- coding: utf-8 -*-
# !/usr/bin/python3
# utils for mask detection

import cv2
import math
import numpy as np


# 检测模型输入尺寸
net_h = 352
net_w = 640

# 检测模型的类别
class_names = ["face", "mask", "person"]
class_num   = len(class_names)

# 检测模型的anchors，用于解码出检测框
stride_list = [8, 16, 32]
anchors_1   = np.array([[10,13],   [16,30],   [33,23]])   / stride_list[0]
anchors_2   = np.array([[30,61], [62,45],   [59,119]])   / stride_list[1]
anchors_3   = np.array([[116,90], [156,198], [163,326]]) / stride_list[2]
anchor_list = [anchors_1, anchors_2, anchors_3]

# 检测框的输出阈值、NMS筛选阈值和人形/人脸区域匹配阈值
conf_threshold   = 0.3
iou_threshold    = 0.4
cover_threshold  = 0.8


# 图片预处理：缩放到模型输入尺寸
def preprocess(img_data):
    h, w, c   = img_data.shape
    new_image = cv2.resize(img_data, (net_w, net_h))
    return new_image, w, h
    
def overlap(x1, x2, x3, x4):
    left  = max(x1, x3)
    right = min(x2, x4)
    return right - left

# 计算两个矩形框的IOU
def cal_iou(box1, box2):
    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
    return inter_area * 1.0 / union_area

# 计算两个矩形框的IOU与box2区域的比值
def cover_ratio(box1, box2):
    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    small_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area * 1.0 / small_area

# 使用NMS筛选检测框
def apply_nms(all_boxes, thres):
    res = []
    
    for cls in range(class_num):        
        cls_bboxes   = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i+1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

# 从模型输出的特征矩阵中解码出检测框的位置、类别、置信度等信息
def decode_bbox(conv_output, anchors, img_w, img_h):

    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
    
    _, h, w = conv_output.shape    
    pred    = conv_output.transpose((1,2,0)).reshape((h * w, 3, 5+class_num))
    
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0]  = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1,0))) / w
    pred[..., 1]  = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1,0))) / h
    pred[..., 2]  = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1,0)) / w
    pred[..., 3]  = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1,0)) / h
                  
    bbox          = np.zeros((h * w, 3, 4))
    bbox[..., 0]  = np.maximum((pred[..., 0] - pred[..., 2] / 2.0) * img_w, 0)     # x_min
    bbox[..., 1]  = np.maximum((pred[..., 1] - pred[..., 3] / 2.0) * img_h, 0)     # y_min
    bbox[..., 2]  = np.minimum((pred[..., 0] + pred[..., 2] / 2.0) * img_w, img_w) # x_max
    bbox[..., 3]  = np.minimum((pred[..., 1] + pred[..., 3] / 2.0) * img_h, img_h) # y_max
    
    pred[..., :4] = bbox
    pred          = pred.reshape((-1, 5+class_num))
    pred[:, 4]    = pred[:, 4] * pred[:, 5:].max(1)    # 类别
    pred          = pred[pred[:, 4] >= conf_threshold]
    pred[:, 5]    = np.argmax(pred[:, 5:], axis=-1)    # 置信度
    
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4]-1].append(box)

    return all_boxes

# 从模型输出中得到检测框
def get_result(model_outputs, img_w, img_h):

    num_channel = 3 * (class_num + 5)    
    all_boxes   = [[] for ix in range(class_num)]
    for ix in range(3):
        pred      = model_outputs[2-ix].reshape((num_channel, net_h // stride_list[ix], net_w // stride_list[ix]))
        anchors   = anchor_list[ix]
        boxes     = decode_bbox(pred, anchors, img_w, img_h)        
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_num)]
    
    res = apply_nms(all_boxes, iou_threshold)    
    return res

# 在图中画出检测框，输出类别信息
def draw_boxes(img_data, bboxes):
    thickness      = 2
    font_scale     = 1
    text_font      = cv2.FONT_HERSHEY_SIMPLEX
    for bbox in bboxes:
        label = int(bbox[4])
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        score = bbox[5]
        
        if label == 2: # person
            face_flag = False
            face_box  = None
            for bbox2 in bboxes:
                if int(bbox2[4]) == 0 and cover_ratio(bbox, bbox2) >= cover_threshold:  # face
                    face_flag = True
                    face_box  = bbox2
            if not face_flag:
                cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (255,255,0), thickness)
                cv2.putText(img_data, 'unknown', (x_min, y_min-20), text_font, font_scale, (255,255,0), thickness)
            else:
                has_mask = False
                for bbox3 in bboxes:
                    if int(bbox3[4]) == 1 and cover_ratio(face_box, bbox3) >= cover_threshold:  # mask 
                        has_mask = True
                if has_mask:
                    cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (0,255,0), thickness)
                    cv2.putText(img_data, 'has_mask', (x_min, y_min-20), text_font, font_scale, (0,255,0), thickness)
                else:
                    cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (255,0,0), thickness)
                    cv2.putText(img_data, 'no_mask', (x_min, y_min-20), text_font, font_scale, (255,0,0), thickness)
        elif label == 1: # mask
            cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), (0,255,255), thickness)

    return img_data