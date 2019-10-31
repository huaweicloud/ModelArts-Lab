import pandas as pd
import numpy as np
import h5py

def comb_img(img_a,img_b):
    """
    将实部虚部结合
    :param img_a: 实部   Format:(32,32)
    :param img_b: 虚部   Format:(32,32)
    :return: 二者平方和开根号得到的图像    Format:(32,32)
    """
    output = np.zeros(img_a.shape)
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            output[i][j] = np.sqrt(img_a[i][j]*img_a[i][j] + img_b[i][j]*img_b[i][j])
    return output


def comb(data, a, b):
    """
    对于一个数据集，将两个通道进行结合
    :param data: 待处理的数据集      Format:(None,32,32,8)
    :param a: 实部通道index          int
    :param b: 虚部通道index          int
    :return: 结合后的单通道数据集     Format:(None,32,32)
    """
    output = np.zeros((data.shape[0], 32, 32))
    for i in range(data.shape[0]):
        sample = data[i]
        s_a = sample[:, :, a]
        s_b = sample[:, :, b]
        output[i] = comb_img(s_a, s_b)

    return output
