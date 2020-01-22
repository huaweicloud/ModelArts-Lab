#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import os
import argparse
import moxing as mox
import shutil

# get current running path
PWD = os.getcwd()
PYTHONPATH = os.environ.get('PYTHONPATH', './').split(':')
# find caffe path
for line in PYTHONPATH:
  if 'caffe' in line:
    CAFFE_PATH = line[:-7]
    break
assert CAFFE_PATH is not None

# get /cache path
CACHE_DIR = os.environ.get('DLS_LOCAL_CACHE_PATH', './')
# caffe bin
BUILD = os.path.join(CAFFE_PATH, 'bin')
# dataset path
DATA = os.path.join(CACHE_DIR, 'dataset')
# dataset backend
BACKEND = 'lmdb'
# model name in inference service
OUTPUT_MODEL_NAME = 'lenet.caffemodel'
# deploy name in inference service
DEPLOY_PROTO_NAME = 'lenet_deploy.prototxt'
# inference script in inference service
SERVICE_FILE_PATH = "customize_service.py"
# config in inference service
CONFIG_PATH = "config.json"
# solver name in training
SOLVER_NAME = 'lenet_solver.prototxt'
# net prototxt in training
NET_NAME = 'lenet_train_test.prototxt'
# local model path
MODLE_LOCAL_PATH = os.path.join(CACHE_DIR, 'lenet_mnist')
# local export path
EXPORT_PATH = os.path.join(CACHE_DIR, 'mnist_model')
# local export model path
EXPORT_MODEL_PATH = os.path.join(EXPORT_PATH, 'model')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default='/data/Mnist-Data-Set/',
                        help='the obs url to save data')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='num of gpus')
    parser.add_argument('--train_url', type=str, default='./tmp/test_mnist',
                        help='the obs url to save model')
    parser.add_argument('--data_path_suffix', type=str, default='caffe_mnist',
                        help='the name of folder that save train file')
    args = parser.parse_args()
    return args


def generate_model_package(model_obs_output=None, src_path=None):
    def conver2iter(iter_str1, iter_str2):
        x = int(iter_str1.split('iter_')[-1].split('.caffemodel')[0])
        y = int(iter_str2.split('iter_')[-1].split('.caffemodel')[0])
        if x > y:
            return 1
        else:
            return -1
    model_path_list = [x for x in os.listdir(MODLE_LOCAL_PATH) if x.endswith('.caffemodel')]
    model_path_list.sort(cmp=conver2iter)
    final_model_name = os.path.join(MODLE_LOCAL_PATH, model_path_list[-1])
    if os.path.exists(EXPORT_MODEL_PATH):
        shutil.rmtree(EXPORT_MODEL_PATH)
    os.mkdir(EXPORT_MODEL_PATH)
    # copy to local path
    shutil.copyfile(os.path.join(src_path, SERVICE_FILE_PATH),
                    os.path.join(EXPORT_MODEL_PATH, SERVICE_FILE_PATH))
    shutil.copyfile(final_model_name,
                    os.path.join(EXPORT_MODEL_PATH, OUTPUT_MODEL_NAME))
    shutil.copyfile(os.path.join(src_path, DEPLOY_PROTO_NAME),
                    os.path.join(EXPORT_MODEL_PATH, DEPLOY_PROTO_NAME))
    shutil.copyfile(os.path.join(src_path, CONFIG_PATH),
                    os.path.join(EXPORT_MODEL_PATH, CONFIG_PATH))

    # copy final model from local to obs
    model_obs_output = os.path.join(model_obs_output, "model")
    print("model_obs_output: " + model_obs_output)
    mox.file.copy_parallel(src_url=EXPORT_MODEL_PATH, dst_url=model_obs_output)


def train(gpu_nums, model_obs_output, src_path):
    # converts the mnist data into lmdb/leveldb format
    cmd = '{BUILD}/convert_mnist_data.bin {DATA}/train-images-idx3-ubyte \
                   {DATA}/train-labels-idx1-ubyte {DATA}/mnist_train_{BACKEND} \
                   --backend={BACKEND}'.format(BUILD=BUILD,
                                               DATA=DATA,
                                               BACKEND=BACKEND)
    ret1 = os.system(cmd)
    cmd = '{BUILD}/convert_mnist_data.bin {DATA}/t10k-images-idx3-ubyte \
                   {DATA}/t10k-labels-idx1-ubyte {DATA}/mnist_test_{BACKEND} \
                   --backend={BACKEND}'.format(BUILD=BUILD,
                                               DATA=DATA,
                                               BACKEND=BACKEND)
    ret2 = os.system(cmd)
    print(cmd)
    assert ret1 == 0 and ret2 == 0, 'There are some bugs in the building dataset.'
    
    # training
    solver_file = os.path.join(src_path, SOLVER_NAME)
    cmd = '{}/caffe.bin train  -solver {}'.format(BUILD, solver_file)
    if gpu_nums:
        gpus = ','.join('%s' % id for id in range(gpu_nums))
        cmd += ' -gpu {}'.format(gpus)
    print('cmd: ' + cmd)
    ret = os.system(cmd)
    if ret == 0:
        # Export model
        generate_model_package(model_obs_output, src_path)
        print("task done")
    else:
        raise Exception("There are some bugs in the training.")


if __name__ == "__main__":
    # get argparse
    args = get_args()
    
    # data local url
    os.makedirs(DATA)
    print('local_dataset_url: ' + DATA)
    # data obs url
    assert args.data_url is not None, 'data path is empty'
    print('data_url:' + args.data_url)
    # copy data from obs to local
    mox.file.copy_parallel(src_url=args.data_url, dst_url=DATA)
    # model save path
    os.makedirs(MODLE_LOCAL_PATH)
    os.makedirs(EXPORT_PATH)
    
    # copy net to cache
    src_path = os.path.join(PWD, args.data_path_suffix, 'src')
    net_path = os.path.join(src_path, NET_NAME)
    mox.file.copy(net_path, os.path.join(CACHE_DIR, NET_NAME))
    # train
    train(args.num_gpus, args.train_url, src_path)
