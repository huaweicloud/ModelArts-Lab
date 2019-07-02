import ctypes
ctypes.CDLL('./codes/libimageaugdefault.so', ctypes.RTLD_LOCAL)
import mxnet as mx
import argparse
import logging
import os
from codes.symbol.resnet import get_symbol
import time

# load data
def get_mnist_iter(args):
    train_data_path = os.path.join(args.data_url, 'Caltech101_train.rec')
    eval_data_path = os.path.join(args.data_url, 'Caltech101_val.rec')

    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_data_path,
        data_shape=(3, 224, 224),
        data_name='data',
        label_name='softmax_label',
        label_width=1,
        batch_size=args.batch_size,
        shuffle=True,
        rand_crop=0,
        # rand_mirror=1,
        pca_noise=0.1,
        random_resized_crop=1,
        min_random_area=0.08,
        max_random_aspect_ratio=4./3.,
        min_random_aspect_ratio=3./4.,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        std_r=58.395,
        std_g=57.12,
        std_b=57.375,
        pad=0,
        fill_value=127,
        preprocess_threads=4,
        seed=int(time.time() - 0),
    )

    val_iter = mx.io.ImageRecordIter(
        path_imgrec=eval_data_path,
        data_shape=(3, 224, 224),
        data_name='data',
        label_name='softmax_label',
        label_width=1,
        batch_size=args.batch_size,
        mean_r=123.68,
        mean_g=116.779,
        mean_b=103.939,
        std_r=58.395,
        std_g=57.12,
        std_b=57.375,
        fill_value=127,
        preprocess_threads=4,
    )
    return train_iter, val_iter


def fit(args):
    # create kvstore
    kv = mx.kvstore.create(args.kv_store)
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    # get train data
    train, val = get_mnist_iter(args)
    # create checkpoint
    checkpoint = mx.callback.do_checkpoint(
        args.train_url if kv.rank == 0 else "%s-%d" % (
            args.train_url, kv.rank))
    # create callbacks after end of every batch
    batch_end_callbacks = [
        mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    # get the created network
    network = get_symbol(num_layers=args.num_layers, num_classes=args.num_classes,
                         image_shape='3,224,224')
    # create context
    devs = mx.cpu() if args.num_gpus == 0 else [mx.gpu(int(i)) for i in
                                                range(args.num_gpus)]
    # create model
    model = mx.mod.Module(context=devs, symbol=network)
    # create an initialization method
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                 magnitude=2)
    # create params of optimizer
    lr_step = []
    lr_step_input = args.lr_step.split(',')
    for i in lr_step_input:
        lr_step.append(int(i) * args.num_examples / args.batch_size)
    optimizer_params = {'multi_precision': True, 'learning_rate': args.lr,
                        'momentum': 0.9, 'wd': 0.0001, 'clip_gradient': 5,
                        'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(
                            step=lr_step,
                            factor=args.lr_factor)}
    metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy(), mx.metric.TopKAccuracy(5)]
    # run
    model.fit(train_data=train,
              eval_data=val,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_metric=metrics,
              arg_params={},
              aux_params={},
              kvstore=kv,
              optimizer='nag',
              optimizer_params=optimizer_params,
              initializer=initializer,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True)

    if args.export_model == 1 and args.train_url is not None and len(
            args.train_url):
        import moxing.mxnet as mox
        end_epoch = args.num_epochs
        save_path = args.train_url if kv.rank == 0 else "%s-%d" % (
            args.train_url, kv.rank)
        params_path = '%s-%04d.params' % (save_path, end_epoch)
        json_path = ('%s-symbol.json' % save_path)
        logging.info(params_path + 'used to predict')
        mox.file.make_dirs(os.path.join(args.train_url, 'model'))
        pred_params_path = os.path.join(args.train_url, 'model',
                                        'pred_model-0000.params')
        pred_json_path = os.path.join(args.train_url, 'model',
                                      'pred_model-symbol.json')
        mox.file.copy(params_path, pred_params_path)
        mox.file.copy(json_path, pred_json_path)
        for i in range(1, args.num_epochs + 1, 1):
            mox.file.remove('%s-%04d.params' % (save_path, i))
        mox.file.remove(json_path)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_classes', type=int, default=102,
                        help='the number of classes')
    parser.add_argument('--num_examples', type=int, default=7316,
                        help='the number of training examples,9145 in total')

    parser.add_argument('--data_url', type=str,
                        default='/home/chenyi00451803/data_set/data_caltech/',
                        help='the training data')
    parser.add_argument('--num_layers', type=int, default=34,
                        help='layers of resnet, support 18 34 50')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type=str, default='16, 24, 27',
                        help='change lr epoch')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='learning rate change factor')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='max num of epochs')
    parser.add_argument('--disp_batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--kv_store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--train_url', type=str,
                        default='/home/chenyi00451803/train_url/',
                        help='the path model saved')
    parser.add_argument('--num_gpus', type=int, default='1',
                        help='number of gpus')
    parser.add_argument('--export_model', type=int, default=1, help='1: export model for predict job \
                                                                     0: not export model')
    args, unkown = parser.parse_known_args()

    fit(args)
