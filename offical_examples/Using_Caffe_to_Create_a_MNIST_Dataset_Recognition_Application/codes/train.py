#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import os
import argparse
import moxing as mox
import multiprocessing
import time
import logging

TRAIN_BASE_PATH = "/home/work/user-job-dir/"

def save_inter_model(src_url, dst_url, interval=10):
    save_num = 1
    while True:
        time.sleep(interval * 60)
        model_dst_url = os.path.join(dst_url, str(save_num))
        print("copy inter model from {} to {}".format(str(src_url), str(model_dst_url)))
        mox.file.copy_parallel(src_url=src_url, dst_url=model_dst_url)
        save_num = save_num + 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default='s3://obs--jnn/datasets/caffe_mnist/',
                        help='the obs url to save data')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='num of gpus')
    parser.add_argument('--train_url', type=str, default='s3://obs--jnn/save_model/caffe_mnist/',
                        help='the obs url to save model')
    parser.add_argument('--data_path_suffix', type=str, default='caffe_mnist',
                        help='the name of folder that save train file')
    parser.add_argument('--data_local_path', type=str, default='/home/work/dataset',
                        help='the local path to save dataset')
    parser.add_argument('--model_local_path', type=str, default='/home/work/lenet_mnist',
                        help='the local path to save model')
    

    args = parser.parse_args()
    
    # data local url
    
    local_dataset_url = args.data_local_path
    if not os.path.exists(local_dataset_url):
        os.makedirs(local_dataset_url)
    print('local_dataset_url: ' + local_dataset_url)

    #data obs url
    data_url = args.data_url
    print('data_url:' + data_url)
    gpu_nums = args.num_gpus
    
    
    try:
        if mox.file.exists(data_url):
            #copy data from obs to local
            print("data obs url exists")
            mox.file.copy_parallel(src_url=data_url, dst_url=local_dataset_url)

            #converts the mnist data into lmdb/leveldb format
            DATA = '/home/work/dataset/'
            BUILD = '/home/work/caffe/bin'
            BACKEND = 'lmdb'
            cmd = '{BUILD}/convert_mnist_data.bin {DATA}/train-images-idx3-ubyte {DATA}/train-labels-idx1-ubyte {DATA}/mnist_train_{BACKEND} --backend={BACKEND}'.format(BUILD = BUILD, DATA = DATA, BACKEND = BACKEND)
            os.system(cmd)
            cmd = '{BUILD}/convert_mnist_data.bin {DATA}/t10k-images-idx3-ubyte {DATA}/t10k-labels-idx1-ubyte {DATA}/mnist_test_{BACKEND} --backend={BACKEND}'.format(BUILD = BUILD, DATA = DATA, BACKEND = BACKEND)
            os.system(cmd)
            
            # model save path
            model_local_output = args.model_local_path
            if not os.path.exists(model_local_output):
                os.makedirs(model_local_output)
            print("model_local_output: " + model_local_output)
            model_obs_output = args.train_url

            #Training
            solver_name = 'lenet_solver.prototxt'
            train_file_path = os.path.join(TRAIN_BASE_PATH, args.data_path_suffix)
            solver_file = os.path.join(train_file_path, solver_name)
            
            #Timing synchronization model from local to obs
            inter_save_process = multiprocessing.Process(target=save_inter_model, args=(model_local_output, model_obs_output, 1))
            inter_save_process.start()
                
            cmd = '/home/work/caffe/bin/caffe.bin train  -solver {}'.format(solver_file)
            if gpu_nums:
                gpus = ','.join('%s' %id for id in range(gpu_nums))
                cmd += ' -gpu {}'.format(gpus)
            print('cmd: ' + cmd)
            os.system(cmd)
            
            inter_save_process.terminate()
            # copy final model from local to obs
            model_obs_output = os.path.join(model_obs_output, "final")
            print("model_obs_output: " + model_obs_output)
            if not mox.file.exists(model_obs_output):
                mox.file.make_dirs(model_obs_output)
            mox.file.copy_parallel(src_url=model_local_output, dst_url=model_obs_output)

            print("task done")
    except BaseException as err:
        print(err)


