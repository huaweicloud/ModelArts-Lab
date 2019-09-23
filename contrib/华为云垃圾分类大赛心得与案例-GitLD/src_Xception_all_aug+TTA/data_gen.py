# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image

from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)
        img = img.resize((self.img_size[0], self.img_size[0]))
        img = img.convert('RGB')
        img = np.array(img)
        img = img[:, :, ::-1]
        return img

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)


def data_flow(train_data_dir, batch_size, num_classes, input_size):  # need modify
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    labels = np_utils.to_categorical(labels, num_classes)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size=0.2, random_state=0)
    print('total samples: %d, training samples: %d, validation samples: %d' % (len(img_paths), len(train_img_paths), len(validation_img_paths)))

    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size])
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size])
    # # 构造多进程的数据流生成器
    # train_enqueuer = OrderedEnqueuer(train_sequence, use_multiprocessing=True, shuffle=True)
    # validation_enqueuer = OrderedEnqueuer(validation_sequence, use_multiprocessing=True, shuffle=True)
    #
    # # 启动数据生成器
    # n_cpu = multiprocessing.cpu_count()
    # train_enqueuer.start(workers=int(n_cpu * 0.7), max_queue_size=10)
    # validation_enqueuer.start(workers=1, max_queue_size=10)
    # train_data_generator = train_enqueuer.get()
    # validation_data_generator = validation_enqueuer.get()

    # return train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator
    return train_sequence, validation_sequence


if __name__ == '__main__':
    # train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator = data_flow(dog_cat_data_path, batch_size)
    # for i in range(10):
    #     train_data_batch = next(train_data_generator)
    # train_enqueuer.stop()
    # validation_enqueuer.stop()
    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size)
    batch_data, bacth_label = train_sequence.__getitem__(5)
    label_name = ['cat', 'dog']
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_2_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_3_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    print('end')
