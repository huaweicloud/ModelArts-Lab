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
import os

import moxing.mxnet as mox
import mxnet as mx
import numpy as np

OFFSETS = [(13, 13), (26, 26), (52, 52)]
ANCHORS = [[116, 90, 156, 198, 373, 326],
           [30, 61, 62, 45, 59, 119],
           [10, 13, 16, 30, 33, 23]]

logging.info(os.getcwd())
logging.info(os.listdir(os.getcwd()))


def add_parameter():
    parser = argparse.ArgumentParser(description='train faster rcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_file_path', type=str,
                        help='train sample list file')
    parser.add_argument('--val_file_path', type=str,
                        help='validation sample list file')
    parser.add_argument('--index_file', type=str, default='./index',
                        help='label map dict file')
    parser.add_argument('--network', type=str, default='yolov3',
                        help='name of network')
    parser.add_argument('--num_classes', type=int, default=1,
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
    parser.add_argument('--resume', action='store_true',
                        help='continue training')
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
    parser.add_argument('--num_epoch', type=int, default=100,
                        help='max num of epochs')
    parser.add_argument('--kv_store', type=str, default='device')
    parser.add_argument('--disp_batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--ignore_iou_thresh', type=float, default=0.7,
                        help='iou thresh')
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='how many epochs to save model')
    parser.add_argument('--eval_frequency', type=int, default=1,
                        help='how many epochs to do validation')
    parser.add_argument('--preprocess_threads', type=int, default=0,
                        help='how many threads to read data')
    parser.add_argument('--num_examples', type=int, default=16551,
                        help='number of images in train data set')
    parser.add_argument('--export_model', type=bool, default=True,
                        help='change train url to model,metric.json')
    parser.add_argument('--train_url', type=str, help='the path model saved')
    args, _ = parser.parse_known_args()
    print(args.num_classes)
    return args


class Yolo_eval(object):
    def __init__(self, sym, val_data, period=1):
        self.val_data = val_data
        self.period = period
        num_gpus = mox.get_hyper_parameter('num_gpus')
        devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [mx.gpu(0)]
        self.model = mx.mod.Module(
            context=devs,
            symbol=sym,
            data_names=['data'],
            label_names=None)
        self.model.bind(
            for_training=False,
            data_shapes=val_data.provide_data,
            label_shapes=val_data.provide_label)
        class_names = self.val_data._dataset.classes
        for i in range(len(class_names)):
            class_names[i] = str(class_names[i])
        self.eval_metric = mox.contrib_metrics.YoloVOCMApMetric(
            iou_thresh=0.5, class_names=class_names, train_url=args.train_url)

    def __call__(self, iter_no, sym, arg, aux):
        """The validation function."""
        if (iter_no + 1) % self.period != 0:
            return
        self.model.set_params(arg, aux)
        self.eval_metric.reset()
        mx.nd.waitall()
        for batch in self.val_data:
            if self.model._data_shapes[0].shape != batch.data[0].shape:
                break
            self.model.forward(batch, is_train=False)
            pred = self.model.get_outputs()
            gt_difficults = batch.label[0].slice_axis(
                axis=-1, begin=5,
                end=6) if batch.label[0].shape[-1] > 5 else None
            self.eval_metric.update(
                pred_bboxes=pred[2],
                pred_labels=pred[0],
                pred_scores=pred[1],
                gt_bboxes=batch.label[0].slice_axis(axis=-1, begin=0, end=4),
                gt_labels=batch.label[0].slice_axis(axis=-1, begin=4, end=5),
                gt_difficults=gt_difficults)
        logging.info(self.eval_metric.get_name_value())


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


def get_data():
    data_set = mox.get_data_iter(
        'yolo_data',
        train_file=args.train_file_path,
        val_file=args.val_file_path,
        split_spec=0.8,
        hyper_train={'width': args.data_shape,
                     'height': args.data_shape,
                     'batch_size': args.batch_size,
                     'index_file': args.index_file,
                     'shuffle': True,
                     'preprocess_threads': args.preprocess_threads},
        hyper_val={'width': args.data_shape,
                   'height': args.data_shape,
                   'batch_size': 1,
                   'index_file': args.index_file,
                   'shuffle': False,
                   'preprocess_threads': args.preprocess_threads},
        anchors=ANCHORS,
        offsets=OFFSETS)
    return data_set


def get_optimizer_params():
    optimizer_params = mox.get_optimizer_params(
        num_examples=args.num_examples,
        lr=args.lr,
        batch_size=args.batch_size,
        lr_scheduler_mode='WarmUpMultiFactor',
        warmup_epochs=args.warm_up_epochs,
        warmup_begin_lr=0.0,
        lr_step_epochs=args.lr_steps,
        num_epoch=args.num_epoch)
    return optimizer_params


def get_symbol(is_train=True):
    if is_train:
        if args.num_sync_bn_devices != -1 and \
                args.num_sync_bn_devices != mox.get_hyper_parameter('num_gpus'):
            logging.info('num_sync_bn_devices must equal to num_gpus')
            args.num_sync_bn_devices = mox.get_hyper_parameter('num_gpus')
        net = mox.get_model(
            'object_detection',
            'yolov3',
            num_classes=args.num_classes,
            dtype=args.dtype,
            ignore_iou_thresh=args.ignore_iou_thresh,
            label_smooth=False,
            num_sync_bn_devices=args.num_sync_bn_devices,
            is_train=True)
    else:
        net = mox.get_model(
            'object_detection',
            'yolov3',
            num_classes=args.num_classes,
            dtype=args.dtype,
            ignore_iou_thresh=args.ignore_iou_thresh,
            label_smooth=False,
            num_sync_bn_devices=-1,
            is_train=False)
    return net


def get_model():
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [
        mx.gpu(int(i)) for i in range(num_gpus)
    ]
    model = mx.mod.Module(
        context=devs,
        symbol=get_symbol(),
        data_names=['data'],
        label_names=['gt_boxes', 'obj_t', 'centers_t', 'scales_t', 'weights_t', 'clas_t'])
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    return model


def get_predict_file_url(train_url, num_classes, class_name):
    with mox.file.File('./youzi/src/yolov3_service.py', 'r') as file:
        predict = file.readlines()
    predict[14] = 'INPUT_SHAPE =  ' + str(args.data_shape) + '\n'
    predict[16] = 'CLASS_NAMES =  ' + str(class_name) + '\n'
    with mox.file.File(
            os.path.join(train_url, 'customize_service.py'), 'w+') as file:
        for i in predict:
            file.write(i)


def export_model():
    # change train url to {model, metric.json}
    mox.export_model(args.train_url)
    symbol = get_symbol(is_train=False)
    symbol.save(
        os.path.join(args.train_url, 'model', 'fine_tune-symbol.json'))
    for i in mox.file.list_directory(
            os.path.join(args.train_url, 'model')):
        if 'params' in i:
            mox.file.rename(
                os.path.join(args.train_url, 'model', i),
                os.path.join(args.train_url, 'model', 'fine_tune-0000.params'))


def train_model():
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
    metrics = [
        mox.contrib_metrics.YoloLoss(name='ObjLoss', index=0),
        mox.contrib_metrics.YoloLoss(name='BoxCenterLoss', index=1),
        mox.contrib_metrics.YoloLoss(name='BoxScaleLoss', index=2),
        mox.contrib_metrics.YoloLoss(name='ClassLoss', index=3)
    ]
    epoch_end_callbacks = []
    if args.train_url is not None:
        worker_id = mox.get_hyper_parameter('worker_id')
        save_path = args.train_url if worker_id == 0 \
            else "%s-%d" % (args.train_url, worker_id)
        epoch_end_callbacks.append(
            mx.callback.do_checkpoint(
                os.path.join(save_path, 'fine_tune'), args.save_frequency))
        get_predict_file_url(args.train_url, args.num_classes,
                             data_set[0]._dataset.classes)
    logging.info("XXX:" + str(data_set[1]))
    if data_set[1] is not None:
        eval_sym = get_symbol(is_train=False)
        epoch_end_callbacks.append(
            Yolo_eval(eval_sym, data_set[1], args.eval_frequency))
    if args.resume:
        assert args.checkpoint_url is not None and args.load_epoch is not None
        prefix = args.checkpoint_url
        epoch = args.load_epoch
    else:
        assert args.pretrained_epoch is not None and args.pretrained is not None
        prefix = args.pretrained
        epoch = args.pretrained_epoch
    _, arg, aux = mox.load_model(r"./youzi/model/" + prefix, epoch)
    optimizer_params = get_optimizer_params()
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False)]

    mox.run(data_set=(data_set[0], None),
            optimizer='sgd',
            optimizer_params=optimizer_params,
            run_mode=mox.ModeKeys.TRAIN,
            model=get_model(),
            batch_end_callbacks=batch_end_callbacks,
            epoch_end_callbacks=epoch_end_callbacks,
            initializer=initializer,
            batch_size=args.batch_size,
            params_tuple=(arg, aux),
            metrics=metrics,
            num_epoch=args.num_epoch)
    if mox.get_hyper_parameter('worker_id') is 0 and args.export_model:
        export_model()


if __name__ == '__main__':
    args = add_parameter()
    train_model()
