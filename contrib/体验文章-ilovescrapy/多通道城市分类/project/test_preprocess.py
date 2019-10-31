import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import preprocess

data = np.load('E:/Alibaba German AI Challenge/data_process/s2/s2_data_shuffle.npy')
filename = 'E:/Alibaba German AI Challenge/origin_DATA/round1_test_a_20181109.h5'
f = h5py.File(filename,'r')
test = np.array(f['sen2'])
print('load up!')

channel_list = [3,4,5,7,8,9]

####################################################################################
#
#channel_list = [4,7,8,9]
#_sample = data[1,:10240].reshape(32,32,10)
#plt.figure(figsize = (20,20))
#cc = 0
#for i in channel_list:
#    plt.subplot(241+cc)
#    cc += 1
#    plt.imshow(_sample[:,:,i].reshape((32,32)),cmap=plt.cm.get_cmap('gray'))
#    plt.colorbar()
#    plt.title('Original-Sentinel-2-%d'%i)
#
#pre_data = preprocess.pre_gs(data,channel_list)
#print('preprocess up!')
#
#sample = pre_data[1,:10240].reshape(32,32,10)
#
#for i in channel_list:
#    plt.subplot(241+cc)
#    cc += 1
#    plt.imshow(sample[:,:,i].reshape((32,32)),cmap=plt.cm.get_cmap('gray'))
#    plt.colorbar()
#    plt.title('Sentinel-2-%d'%i)
#plt.show()
#
####################################################################################

pre_data = preprocess.pre_gs(data,channel_list)
pre_test = preprocess.pre_gs(test,channel_list)
print('preprocess up!')
print('pre_data shape is ',pre_data.shape)
print('pre_test shape is ',pre_test.shape)

np.save('s2_data_shuffle_gs_345789.npy', pre_data)
np.save('s2_test_shuffle_gs_345789.npy', pre_test)