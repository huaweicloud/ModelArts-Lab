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

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import os
import sys
sys.path.append(".")
import numpy as np

import mxnet as mx
from data.yolo_dataset import get_data_iter
from symbol import yolov3
from utils.yolo_metric import YoloLoss, YoloVOCMApMetric
from utils.lr_schedular import WarmUpMultiFactorScheduler
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
OFFSETS = [(13, 13), (26, 26), (52, 52)]
ANCHORS = [[116, 90, 156, 198, 373, 326],
           [30, 61, 62, 45, 59, 119],
           [10, 13, 16, 30, 33, 23]]


def add_parameter():
    parser = argparse.ArgumentParser(
        description='train faster rcnn',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
    parser.add_argument('--data_url', type=str, help='root data path')
    parser.add_argument('--train_file_path', type=str,
                        help='train sample list file', default=None)
    parser.add_argument('--val_file_path', type=str,
                        help='validation sample list file', default=None)
    parser.add_argument('--index_file', type=str, default='./index',
                        help='label map dict file')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='the number of classes')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--checkpoint_url', type=str, default='yolov3_darknet53',
                        help='prefix of trained model file')
    parser.add_argument('--load_epoch', type=int,
                        help='load the model on epoch use checkpoint_url')
    parser.add_argument('--pretrained', type=str, default='darknet_53',
                        help='pretrained model prefix')
    parser.add_argument('--pretrained_epoch', type=int, default=0,
                        help='pretrained model epoch')
    parser.add_argument('--data_shape', type=int, default=416,
                        help='input data shape')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lr_steps', type=str, default='60, 80',
                        help='show progress for every n batches')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='initial weight decay')
    parser.add_argument('--warm_up_epochs', type=int, default=5,
                        help='warm up learning rate in the fist number epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num_sync_bn_devices', type=int, default=-1,
                        help='sync multi gpu batch_norm')
    parser.add_argument('--resume', action='store_true',
                        help='continue training')
    parser.add_argument('--num_epoch', type=int, default=100,
                        help='max num of epochs')
    parser.add_argument('--kv_store', type=str, default='device')
    parser.add_argument('--disp_batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--ignore_iou_thresh', type=float, default=0.7,
                        help='iou thresh')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='how many epochs to save model')
    parser.add_argument('--eval_frequency', type=int, default=1,
                        help='how many epochs to do validation')
    parser.add_argument('--preprocess_threads', type=int, default=0,
                        help='how many threads to read data')
    parser.add_argument('--num_examples', type=int, default=16551,
                        help='number of images in train data set')
    args, _ = parser.parse_known_args()
    return args


class Yolo_eval(object):
    def __init__(self, sym, val_data, period=1):
        self.val_data = val_data
        self.period = period
        num_gpus = args.num_gpus
        devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [
            mx.gpu(int(i)) for i in range(num_gpus)]
        self.model = mx.mod.Module(
            context=devs,
            symbol=sym,
            data_names=['data'],
            label_names=None)
        self.model.bind(for_training=False,
                        data_shapes=val_data.provide_data,
                        label_shapes=val_data.provide_label)
        class_names = val_data._dataset.classes
        self.eval_metric = YoloVOCMApMetric(
            iou_thresh=0.5, class_names=class_names)

    def __call__(self, iter_no, sym, arg, aux):
        """The validation function."""
        if (iter_no + 1) % self.period != 0:
            return
        self.model.set_params(arg, aux)
        self.eval_metric.reset()
        mx.nd.waitall()
        for batch in self.val_data:
            if self.model._data_shapes[0].shape != batch.data[0].shape:
                continue
            self.model.forward(batch, is_train=False)
            pred = self.model.get_outputs()
            gt_difficults = batch.label[0].slice_axis(
                axis=-1, begin=5, end=6) if batch.label[0].shape[-1] > 5 else None
            self.eval_metric.update(
                pred_bboxes=pred[2], pred_labels=pred[0], pred_scores=pred[1],
                gt_bboxes=batch.label[0].slice_axis(axis=-1, begin=0, end=4),
                gt_labels=batch.label[0].slice_axis(axis=-1, begin=4, end=5),
                gt_difficults=gt_difficults)
        logging.info(self.eval_metric.get_name_value())


def get_data():
    data_set = get_data_iter(data_path=args.data_url,
                             train_file=None,
                             val_file=None,
                             split_spec=0.8,
                             hyper_train={'width': args.data_shape,
                                          'height': args.data_shape,
                                          'batch_size': args.batch_size,
                                          'index_file': args.index_file,
                                          'shuffle': True,
                                          'preprocess_threads': args.preprocess_threads},
                             hyper_val={'width': args.data_shape,
                                        'height': args.data_shape,
                                        'batch_size': args.batch_size,
                                        'index_file': args.index_file,
                                        'shuffle': False,
                                        'preprocess_threads': args.preprocess_threads},
                             anchors=ANCHORS,
                             offsets=OFFSETS)
    return data_set


def get_optimizer_params():
    num_workers = int(os.environ.get('DMLC_NUM_WORKER', 1))
    epoch_size = args.num_examples // args.batch_size
    epoch_size = epoch_size // num_workers
    step_epochs = [float(l) for l in args.lr_steps.split(',')]
    begin_epoch = args.load_epoch if args.load_epoch is not None else 0
    steps = [int(epoch_size * (x - begin_epoch)) for x in step_epochs
                if x - begin_epoch > 0]
    warmup_steps = int(epoch_size * (args.warm_up_epochs - begin_epoch))
    lr_scheduler = WarmUpMultiFactorScheduler(step=steps,
                                              factor=0.1,
                                              base_lr=args.lr,
                                              warmup_steps=warmup_steps,
                                              warmup_begin_lr=0.0,
                                              warmup_mode='linear')
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / (num_workers * args.batch_size)),
                        'multi_precision': True,
                        'momentum': 0.9}
    return optimizer_params


def get_symbol(is_train=True):
    if is_train:
        if args.num_sync_bn_devices != -1 and args.num_sync_bn_devices != args.num_gpus:
            logging.info('num_sync_bn_devices must equal to num_gpus')
            args.num_sync_bn_devices = args.num_gpus
        net = yolov3.get_symbol(num_classes=args.num_classes,
                                dtype=args.dtype,
                                ignore_iou_thresh=args.ignore_iou_thresh,
                                label_smooth=False,
                                num_sync_bn_devices=args.num_sync_bn_devices,
                                is_train=True)
    else:
        net = yolov3.get_symbol(num_classes=args.num_classes,
                                dtype=args.dtype,
                                ignore_iou_thresh=args.ignore_iou_thresh,
                                label_smooth=False,
                                num_sync_bn_devices=-1,
                                is_train=False)
    return net


def get_model():
    num_gpus = args.num_gpus
    devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [
        mx.gpu(int(i)) for i in range(num_gpus)]
    model = mx.mod.Module(
        context=devs,
        symbol=get_symbol(),
        data_names=['data'],
        label_names=['gt_boxes', 'obj_t', 'centers_t',
                     'scales_t', 'weights_t', 'clas_t'])
    return model

def get_anchors(anchors):
    anchor_list = []
    for item in anchors:
        anchor = mx.nd.array(item)
        anchor = anchor.reshape(1, 1, -1, 2)
        anchor_list.append(anchor)
    return anchor_list

def get_offsets(offsets):
    offset_list = []
    for item in offsets:
        grid_x = np.arange(item[1])
        grid_y = np.arange(item[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        offset = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        offset = np.expand_dims(np.expand_dims(offset, axis=0), axis=0)
        offset = mx.nd.array(offset).reshape(1, -1, 1, 2)
        offset_list.append(offset)
    return offset_list

if __name__ == '__main__':
    args = add_parameter()
    import moxing as mox
    mox.file.copy_parallel(args.data_url, '/cache/local_data')
    args.data_url = '/cache/local_data'
    print ('copy success')
    data_set = get_data()
    offset_data = get_offsets(OFFSETS)
    anchor_data = get_anchors(ANCHORS)
    initializer = mx.init.Mixed(['offset_0_weight', 'offset_1_weight', 'offset_2_weight',
                                 'anchors_0_weight', 'anchors_1_weight', 'anchors_2_weight',
                                 '.*'],
                                [mx.init.Constant(offset_data[0]),
                                 mx.init.Constant(offset_data[1]),
                                 mx.init.Constant(offset_data[2]),
                                 mx.init.Constant(anchor_data[0]),
                                 mx.init.Constant(anchor_data[1]),
                                 mx.init.Constant(anchor_data[2]),
                                 mx.init.Xavier(factor_type="in", magnitude=2.34)])
    metrics = [YoloLoss(name='ObjLoss', index=0),
               YoloLoss(name='BoxCenterLoss', index=1),
               YoloLoss(name='BoxScaleLoss', index=2),
               YoloLoss(name='ClassLoss', index=3)]
    epoch_end_callback = []
    if args.checkpoint_url is not None:
        epoch_end_callback = [mx.callback.do_checkpoint(
            args.checkpoint_url, args.save_frequency)]

    if data_set[1] is not None:
        eval_sym = get_symbol(is_train=False)
        epoch_end_callback.append(
            Yolo_eval(eval_sym, data_set[1], args.eval_frequency))
    prefix = args.checkpoint_url if args.resume else args.pretrained
    epoch = args.load_epoch if args.resume else args.pretrained_epoch
    sym, arg, aux = mx.model.load_checkpoint(r"./youzi/model/" + prefix, epoch)
    optimizer_params = get_optimizer_params()
    batch_end_callback = [mx.callback.Speedometer(
        args.batch_size, args.disp_batches, auto_reset=False)]
    model=get_model()
    model.fit(train_data=data_set[0],
              optimizer='sgd',
              kvstore=args.kv_store,
              optimizer_params=optimizer_params,
              batch_end_callback=batch_end_callback,
              epoch_end_callback=epoch_end_callback,
              initializer=initializer,
              arg_params=arg,
              aux_params=aux,
              eval_metric=metrics,
              num_epoch=args.num_epoch,
              allow_missing=True)
