##################################################################
#为了提高模型识别率，对数据的特定通道进行一系列的预处理包括：
#GSG：高斯滤波+锐化+高斯滤波
#GS：高斯滤波+锐化
##################################################################

import pandas as pd
import numpy as np
import h5py
from PIL import Image
import cv2

def GSG(img):
    """
    对单通道图像先后进行：高斯滤波+锐化+高斯滤波操作
    :param img: 待处理的图像
    :return: 处理后的图像
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    step_0 = cv2.GaussianBlur(img,(5,5),0)
    step_1 = cv2.filter2D(step_0, -1, kernel=kernel)
    output = cv2.GaussianBlur(step_1,(5,5),0)
    return output

def GS(img):
    """
    对单通道图像先后进行：高斯滤波+锐化
    :param img: 待处理的图像
    :return: 处理后的图像
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    step_0 = cv2.GaussianBlur(img,(5,5),0)
    output = cv2.filter2D(step_0, -1, kernel=kernel)
    return output

def pre_gsg(data, channel_list):
    """
        对data中的指定通道进行gsg预处理
    :param data: 输入待处理的数据
    :param channel_list: 指定待处理的信道列表
    :return: 处理后的array，2维
    """
    if data.shape[1] == 10257:
        """
        这时处理的是测试数据或验证数据，最后17列是one-hot形式的label
        Format:(None,10257)
        """
        imgs = data[:, :-17].reshape((-1, 32, 32, 10))
        label = data[:, -17:]

        process_imgs = np.zeros(imgs.shape)

        for i in range(imgs.shape[0]):
            temp = imgs[i]
            for j in range(10):
                img = temp[:, :, j].reshape((32, 32))
                if j in channel_list:
                    pro_img = GSG(img)
                    process_imgs[i, :, :, j] = pro_img
                else:
                    process_imgs[i, :, :, j] = img

        x = process_imgs.reshape((-1,10240))
        output = np.hstack((x, label))
        return output

    elif data.shape[1] == 32:
        """
        这时处理的是测试数据，测试数据是4维矩阵
        Format:(None,32,32,10)
        """
        process_imgs = np.zeros(data.shape)

        for i in range(data.shape[0]):
            temp = data[i]
            for j in range(10):
                img = temp[:, :, j].reshape((32, 32))
                if j in channel_list:
                    pro_img = GSG(img)
                    process_imgs[i, :, :, j] = pro_img
                else:
                    process_imgs[i, :, :, j] = img

        output = process_imgs.reshape((-1,10240))
        return output
    else:
        print('Input error:I just preprocess s2 channel!')


def pre_gs(data, channel_list):
    """
        对data中的指定通道进行gsg预处理
    :param data: 输入待处理的数据
    :param channel_list: 指定待处理的信道列表
    :return: 处理后的array，2维
    """
    if data.shape[1] == 10257:
        """
        这时处理的是测试数据或验证数据，最后17列是one-hot形式的label
        Format:(None,10257)
        """
        imgs = data[:, :-17].reshape((-1, 32, 32, 10))
        label = data[:, -17:]

        process_imgs = np.zeros(imgs.shape)

        for i in range(imgs.shape[0]):
            temp = imgs[i]
            for j in range(10):
                img = temp[:, :, j].reshape((32, 32))
                if j in channel_list:
                    pro_img = GS(img)
                    process_imgs[i, :, :, j] = pro_img
                else:
                    process_imgs[i, :, :, j] = img

        x = process_imgs.reshape((-1,10240))
        output = np.hstack((x, label))
        return output

    elif data.shape[1] == 32:
        """
        这时处理的是测试数据，测试数据是4维矩阵
        Format:(None,32,32,10)
        """
        process_imgs = np.zeros(data.shape)

        for i in range(data.shape[0]):
            temp = data[i]
            for j in range(10):
                img = temp[:, :, j].reshape((32, 32))
                if j in channel_list:
                    pro_img = GS(img)
                    process_imgs[i, :, :, j] = pro_img
                else:
                    process_imgs[i, :, :, j] = img

        output = process_imgs.reshape((-1,10240))
        return output
    else:
        print('Input error:I just preprocess s2 channel!')