from __future__ import absolute_import, division, print_function

import logging
import os
import xml.etree.ElementTree as ET

import h5py
import mxnet.image as mx_img
import numpy as np
from moxing.framework import file

from .transform import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform

INDEX_FILE_NAME = './index'


class Detection_dataset(object):
    def __init__(self, img_list, index_file=None,
                 width=416, height=416, is_train=True, **kwargs):
        assert isinstance(img_list, list), 'img_list must be a python list'
        self.img_list = img_list
        self.classes = self._get_label_names(index_file)
        self._im_shapes = {}
        self.label_cache = self._preload_labels()
        if is_train:
            self.yolo_transform = YOLO3DefaultTrainTransform(
                width=width,
                height=height,
                num_classes=len(self.classes),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                mixup=False,
                **kwargs)
        else:
            self.yolo_transform = YOLO3DefaultValTransform(
                width=width,
                height=height,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))

    def __str__(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx][0]
        label = self.label_cache[idx]
        img = file.read(img_path, binary=True)
        img = mx_img.imdecode(img, 1)
        return self.yolo_transform(img, label)

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        anno_path = self.img_list[idx][1]
        tree = ET.ElementTree()
        parser = ET.XMLParser(target=ET.TreeBuilder())
        parser.feed(file.read(anno_path, binary=True))
        tree._root = parser.close()
        root = tree.getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                self.classes.append(cls_name)
            cls_id = self.classes.index(cls_name)
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = 0 if xmax < 0 else xmax
            ymax = 0 if ymax < 0 else ymax
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError(
                    "Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(
            width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(
            height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(
            width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(
            height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.info("Preloading labels into memory...")
        return [self._load_label(idx) for idx in range(len(self.img_list))]

    def _get_label_names(self, index_file):
        classes = []
        if index_file is not None and file.exists(index_file):
            label_file = h5py.File(index_file, 'r')
            classes_name = label_file['labels_list'][:]
            label_file.close()
            classes = [name.decode('utf-8') for name in classes_name]
        else:
            for data_file in self.img_list:
                annotation_file = data_file[1]
                tree = ET.ElementTree()
                parser = ET.XMLParser(target=ET.TreeBuilder())
                parser.feed(file.read(annotation_file, binary=True))
                tree._root = parser.close()
                objs = tree.findall('object')
                non_diff_objs = [obj for obj in objs if
                                 int(obj.find('difficult').text) == 0]
                objs = non_diff_objs
                for obj in objs:
                    class_name = obj.find('name').text.lower().strip()
                    if class_name not in classes:
                        classes.append(class_name)
            if index_file is not None:
                index_file_path = index_file
            else:
                index_file_path = INDEX_FILE_NAME
            if file.exists(index_file_path):
                file.remove(index_file_path)
            label_file = h5py.File(index_file, 'w')
            label_file.create_dataset('labels_list', data=[item.encode() for item in classes])
            label_file.close()
        return classes
