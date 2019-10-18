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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from moxing.framework import file
import random

def get_image_list(data_path, split_spec):
    """get image list
    [[image_path, label_path]]
    :param data_path: data store url
    :param split_spec: split train percent if data doesn't have evaluation data
    Returns:
        train_data_list,
        eval_data_list,
    """
    image_list_train = []
    image_list_eval = []
    class_name = None
    file_list = file.list_directory(data_path)
    donot_have_directory = True
    if 'cache' in file_list:
        file_list.remove('cache')
    for i in file_list:
        if file.is_directory(os.path.join(data_path, i)):
            donot_have_directory = False
            break
    if 'Images' and 'Annotations' in file_list:
        image_list_train, image_list_eval, class_name = \
            get_image_images_annotation(data_path, split_spec)
    elif 'train' and 'eval' in file_list:
        file_list = file.list_directory(os.path.join(data_path, 'train'))
        is_raw = True
        if 'cache' in file_list:
            file_list.remove('cache')
        for i in file_list:
            if file.is_directory(os.path.join(data_path, 'train', i)):
                is_raw = False
                break
        if 'Images' and 'Annotations' in file_list:
            image_list_train, image_list_eval = get_image_train_eval(data_path)
        elif 'image_to_annotation.csv' in file_list:
            image_list_train, image_list_eval = get_image_csv(data_path)
        elif is_raw:
            image_list_train, image_list_eval = \
                get_image_train_eval_raw(data_path)
        else:
            image_list_train, image_list_eval, class_name = \
                get_image_classese_train_eval(data_path)

    elif donot_have_directory:
        image_list_train, image_list_eval, class_name = get_image_raw_txt(data_path, split_spec)
    else:
        image_list_train, image_list_eval, class_name = get_image_classese_raw(data_path, split_spec)
    return image_list_train, image_list_eval, class_name

def get_image_csv(data_path):
    file_list = file.list_directory(os.path.join(data_path, 'train'))
    image_list_train = []
    image_list_eval = []
    for i in file_list:
        if '.txt' in file.list_directory(os.path.join(data_path, 'train', i)):
            image_list_train = os.path.join(data_path, 'train', 'image_to_annotation.csv')
            image_list_eval = os.path.join(data_path, 'eval', 'image_to_annotation.csv')
            break
        elif '.xml' in file.list_directory(os.path.join(data_path, 'train', i)):
            with file.File(os.path.join(data_path, 'train', 'image_to_annotation.csv'), 'r') as f:
                for line in f.readlines()[1:]:
                    image_path, image_label = line.strip().split(',')
                    image_list_train.append([os.path.join(data_path, image_path),
                                             os.path.join(data_path, image_label)])
            with file.File(os.path.join(data_path, 'eval', 'image_to_annotation.csv'), 'r') as f:
                for line in f.readlines()[1:]:
                    image_path, image_label = line.strip().split(',')
                    image_list_eval.append([os.path.join(data_path, image_path),
                                             os.path.join(data_path, image_label)])
            break
    return image_list_train, image_list_eval

def get_image_images_annotation(data_path, split_spec):
    """get image list when data struct is
   {
   |-- data_url
       |-- Images
           |-- a.jpg
           |-- b.jpg
           ...
       |-- Annotations
           |-- a.txt (or a.xml)
           |-- b.txt (or b.xml)
           ...
       |-- label_map_dict (optional)
   }
   :param data_path: data store url
   :param split_spec: split train percent if data doesn't have evaluation data
   Returns:
       train_data_list,
       eval_data_list,
   """
    image_set = []
    label_dict = {}
    label_num = 0
    class_name = []
    # get all labeled data
    image_list_set = file.list_directory(os.path.join(data_path, 'Images'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if file.exists(os.path.join(data_path, 'Annotations', os.path.splitext(i)[0] + '.xml')):
            image_set.append([os.path.join(data_path, 'Images', i),
                              os.path.join(data_path, 'Annotations', os.path.splitext(i)[0] + '.xml')])
        elif file.exists(os.path.join(data_path, 'Annotations', os.path.splitext(i)[0] + '.txt')):
            label_name = file.read(os.path.join(data_path, 'Annotations',
                                                os.path.splitext(i)[0] + '.txt'))
            if label_name not in label_dict.keys():
                label_dict[label_name] = label_num
                class_name.append(label_name)
                label_num = label_num + 1
            image_set.append([os.path.join(data_path, 'Images', i),
                             label_dict[label_name]])

    # split data to train and eval
    num_examples = len(image_set)
    train_num = int(num_examples * split_spec)
    shuffle_list = list(range(num_examples))
    random.shuffle(shuffle_list)
    image_list_train = []
    image_list_eval = []
    for idx, item in enumerate(shuffle_list):
        if idx < train_num:
            image_list_train.append(image_set[item])
        else:
            image_list_eval.append(image_set[item])
    return image_list_train, image_list_eval, class_name

def get_image_raw_txt(data_path, split_spec):
    """get image list when data struct is
    {
    |-- data_url
        |-- a.jpg
        |-- a.txt (or a.xml)
        |-- b.jpg
        |-- b.txt (or b.xml)
        ...
        |-- label_map_dict (optional)
    }
    :param data_path: data store url
    :param split_spec: split train percent if data doesn't have evaluation data
    Returns:
        train_data_list,
        eval_data_list,
    """
    image_list_set = []
    image_set = []

    # get all labeled data
    image_list_set = file.list_directory(data_path)
    label_dict = {}
    label_num = 0
    class_name = []
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if not '.xml' in i and not '.txt' in i:
            if file.exists(os.path.join(data_path, os.path.splitext(i)[0] + '.xml')):
                image_set.append([os.path.join(data_path, i),
                                  os.path.join(data_path, os.path.splitext(i)[0] + '.xml')])
            elif file.exists(os.path.join(data_path, os.path.splitext(i)[0] + '.txt')):
                label_name = file.read(os.path.join(data_path,
                                                    os.path.splitext(i)[0] + '.txt'))
                if label_name not in label_dict.keys():
                    label_dict[label_name] = label_num
                    class_name.append(label_name)
                    label_num = label_num + 1
                image_set.append([os.path.join(data_path, i), label_dict[label_name]])

    # split data to train and eval
    num_examples = len(image_set)
    train_num = int(num_examples * split_spec)
    shuffle_list = list(range(num_examples))
    random.shuffle(shuffle_list)
    image_list_train = []
    image_list_eval = []
    for idx, item in enumerate(shuffle_list):
        if idx < train_num:
            image_list_train.append(image_set[item])
        else:
            image_list_eval.append(image_set[item])
    return image_list_train, image_list_eval, class_name

def get_image_train_eval(data_path):
    """get image list when data struct is
    {
    |-- data_url
        |-- train
            |-- Images
                |-- a.jpg
                ...
            |-- Annotations
                |-- a.txt (or a.xml)
            |-- label_map_dict (optional)
        |-- eval
            |-- Images
                |-- b.jpg
                ...
            |-- Annotations
                |-- b.txt (or b.xml)
                ...
            |-- label_map_dict (optional)
        |-- label_map_dict (optional)
    }
    :param data_path: data store url
    Returns:
      train_data_list,
      eval_data_list,
    """
    image_list_train = []
    # get all labeled train data
    image_list_set = file.list_directory(os.path.join(data_path, 'train', 'Images'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if file.exists(os.path.join(data_path, 'train', 'Annotations', os.path.splitext(i)[0] + '.xml')):
            image_list_train.append([os.path.join(data_path, 'train', 'Images', i),
                                     os.path.join(data_path, 'train', 'Annotations', os.path.splitext(i)[0] + '.xml')])
        elif file.exists(os.path.join(data_path, 'train', 'Annotations', os.path.splitext(i)[0] + '.txt')):
            image_list_train.append([os.path.join(data_path, 'train', 'Images', i),
                                     file.read(os.path.join(data_path, 'train',
                                                            'Annotations',
                                                            os.path.splitext(i)[0] + '.txt'))])
    # get all labeled eval data
    image_list_eval = []
    image_list_set = []
    image_list_set = file.list_directory(os.path.join(data_path, 'eval', 'Images'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if file.exists(os.path.join(data_path, 'eval', 'Annotations', os.path.splitext(i)[0] + '.xml')):
            image_list_eval.append([os.path.join(data_path, 'eval', 'Images', i),
                                    os.path.join(data_path, 'eval', 'Annotations', os.path.splitext(i)[0] + '.xml')])
        elif file.exists(os.path.join(data_path, 'eval', 'Annotations', os.path.splitext(i)[0] + '.txt')):
            image_list_eval.append([os.path.join(data_path, 'eval', 'Images', i),
                                    file.read(os.path.join(data_path, 'eval',
                                                           'Annotations',
                                                           os.path.splitext(i)[0] + '.txt'))])

    return image_list_train, image_list_eval


def get_image_train_eval_raw(data_path):
    """get image list when data struct is
    {
    |-- data_url
        |-- train
            |-- a.jpg
            |-- a.txt (or a.xml)
            ...
            |-- label_map_dict (optional)
        |-- eval
            |-- b.jpg
            |-- b.txt (or b.xml)
            ...
            |-- label_map_dict (optional)
    }
    :param data_path: data store url
    Returns:
      train_data_list,
      eval_data_list,
    """
    image_list_train = []
    # get all labeled train data
    image_list_set = file.list_directory(os.path.join(data_path, 'train'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if not '.xml' in i and not '.txt' in i:
            if file.exists(os.path.join(data_path, 'train', os.path.splitext(i)[0] + '.xml')):
                image_list_train.append([os.path.join(data_path, 'train', i),
                                         os.path.join(data_path, 'train', os.path.splitext(i)[0] + '.xml')])
            elif file.exists(os.path.join(data_path, 'train', os.path.splitext(i)[0] + '.txt')):
                image_list_train.append([os.path.join(data_path, 'train', i),
                                         file.read(os.path.join(data_path,
                                                                'train',
                                                                os.path.splitext(i)[0] + '.txt'))])

    # get all labeled eval data
    image_list_eval = []
    image_list_set = []
    image_list_set = file.list_directory(os.path.join(data_path, 'eval'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if not '.xml' in i and not '.txt' in i:
            if file.exists(os.path.join(data_path, 'eval', os.path.splitext(i)[0] + '.xml')):
                image_list_eval.append([os.path.join(data_path, 'eval', i),
                                        os.path.join(data_path, 'eval', os.path.splitext(i)[0] + '.xml')])
            elif file.exists(os.path.join(data_path, 'eval', os.path.splitext(i)[0] + '.txt')):
                image_list_eval.append([os.path.join(data_path, 'eval', i),
                                        file.read(os.path.join(data_path, 'eval',
                                                               os.path.splitext(i)[0] + '.txt'))])

    return image_list_train, image_list_eval

def get_image_classese_raw(data_path, split_spec):
    """get image list when data struct is
    {
    |-- data_url
        |-- class_1
            |-- a.jpg
            |-- b.jpg
        |-- class_2
            |-- c.jpg
            |-- d.jpg
            ...
        |-- label_map_dict (optional)
    }
    :param data_path: data store url
    Returns:
      train_data_list,
      eval_data_list,
    """
    image_set = []
    class_name = []
    # get all labeled train data
    image_list_set = file.list_directory(data_path)
    for i in image_list_set:
        if not file.is_directory(os.path.join(data_path, i)):
            image_list_set.remove(i)
    assert not image_list_set == [], 'there is no file in data url'
    label_index = 0
    for i in image_list_set:
        if file.is_directory(os.path.join(data_path, i)):
            img_list = file.list_directory(os.path.join(data_path, i))
            for j in img_list:
                label = label_index
                class_name.append(i)
                if not '.xml' in j and not '.txt' in j:
                    image_set.append([os.path.join(data_path, i, j), label])
            label_index += 1

    # split to train and eval
    image_list_train = []
    image_list_eval = []
    start_examples = 0
    for i in image_list_set:
        image_list_set = file.list_directory(os.path.join(data_path, i))
        num_examples = len(image_list_set)
        train_num = int(num_examples * split_spec)
        shuffle_list = list(range(start_examples, start_examples + num_examples))
        random.shuffle(shuffle_list)
        for idx, item in enumerate(shuffle_list):
            if idx < train_num:
                image_list_train.append(image_set[item])
            else:
                image_list_eval.append(image_set[item])
        start_examples += num_examples
    return image_list_train, image_list_eval, class_name

def get_image_classese_train_eval(data_path):
    """get image list when data struct is
    {
    |-- data_url
        |-- train
            |-- class_1
                |-- a.jpg
                ...
            |-- class_2
                |-- b.jpg
                ...
            ...
        |-- eval
            |-- class_1
                |-- c.jpg
                ...
            |-- class_2
                |-- d.jpg
    }
    :param data_path: data store url
    Returns:
      train_data_list,
      eval_data_list,
    """
    image_label_name = {}
    image_list_train = []
    label_index = 0
    class_name = []
    # get all labeled train data
    image_list_set = file.list_directory(os.path.join(data_path, 'train'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if file.is_directory(os.path.join(data_path, 'train', i)):
            img_list = file.list_directory(os.path.join(data_path, 'train', i))
            for j in img_list:
                label = label_index
                class_name.append(i)
                if not '.xml' in j and not '.txt' in j:
                    image_list_train.append([os.path.join(data_path, 'train', i, j), label])
            image_label_name[i] = label_index
            label_index += 1

    # get all labeled eval data
    image_list_eval = []
    image_list_set = file.list_directory(os.path.join(data_path, 'eval'))
    assert not image_list_set == [], 'there is no file in data url'
    for i in image_list_set:
        if file.is_directory(os.path.join(data_path, 'eval', i)):
            img_list = file.list_directory(os.path.join(data_path, 'eval', i))
            for j in img_list:
                label = image_label_name[i]
                if not '.xml' in j and not '.txt' in j:
                    image_list_eval.append([os.path.join(data_path, 'eval', i, j), label])

    return image_list_train, image_list_eval, class_name
