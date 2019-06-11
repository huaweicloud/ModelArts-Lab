import mxnet as mx
import argparse
import logging
import os
# load data
def get_mnist_iter(args):
    train_image = os.path.join(args.data_url + 'train-images-idx3-ubyte')
    train_label = os.path.join(args.data_url + 'train-labels-idx1-ubyte')
    try:
        import moxing.mxnet as mox
    except:
        assert os.path.exists(train_image), 'file train-images-idx3-ubyte is not exist,please check your data url'
        assert os.path.exists(train_label), 'file train-labels-idx1-ubyte is not exist,please check your data url'
    else:
        assert mox.file.exists(train_image), 'file train-images-idx3-ubyte is not exist,please check your data url'
        assert mox.file.exists(train_image), 'file train-labels-idx1-ubyte is not exist,please check your data url'

    train = mx.io.MNISTIter(image=train_image,
                            label=train_label,
                            data_shape=(1, 28, 28),
                            batch_size=args.batch_size,
                            shuffle=True,
                            flat=False,
                            silent=False,
                            seed=10)
    return train

# create network
def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def fit(args):
    # create kvstore
    kv = mx.kvstore.create(args.kv_store)
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    # get train data
    train = get_mnist_iter(args)
    # create checkpoint
    checkpoint = mx.callback.do_checkpoint(args.train_url if kv.rank == 0 else "%s-%d" % (
        args.train_url, kv.rank))
    # create callbacks after end of every batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    # get the created network 
    network = get_symbol(num_classes=args.num_classes)
    # create context
    devs = mx.cpu() if args.num_gpus == 0 else [mx.gpu(int(i)) for i in range(args.num_gpus)]
    # create model
    model = mx.mod.Module(context=devs, symbol=network)
    # create an initialization method
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    # create params of optimizer
    optimizer_params = {'learning_rate': args.lr, 'wd' : 0.0001}
    # run
    model.fit(train,
              begin_epoch=0,
              num_epoch=args.num_epochs,
              eval_data=None,
              eval_metric=['accuracy'],
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=None,
              aux_params=None,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=True)

    if args.export_model == 1 and args.train_url is not None and len(args.train_url):
        import moxing.mxnet as mox
        end_epoch = args.num_epochs
        save_path = args.train_url if kv.rank == 0 else "%s-%d" % (args.train_url, kv.rank)
        params_path = '%s-%04d.params' % (save_path, end_epoch)
        json_path = ('%s-symbol.json' % save_path)
        logging.info(params_path + 'used to predict')
        pred_params_path = os.path.join(args.train_url, 'model', 'pred_model-0000.params')
        pred_json_path = os.path.join(args.train_url, 'model', 'pred_model-symbol.json')
        mox.file.copy(params_path, pred_params_path)
        mox.file.copy(json_path, pred_json_path)
        for i in range(1, args.num_epochs + 1, 1):
            mox.file.remove('%s-%04d.params' % (save_path, i))
        mox.file.remove(json_path)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num_examples', type=int, default=60000,
                        help='the number of training examples')

    parser.add_argument('--data_url', type=str, default='s3://obs-lpf/data/', help='the training data')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='max num of epochs')
    parser.add_argument('--disp_batches', type=int, default=20,
                        help='show progress for every n batches')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--kv_store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--train_url', type=str, default='s3://obs-lpf/ckpt/mnist',
                        help='the path model saved')
    parser.add_argument('--num_gpus', type=int, default='0',
                        help='number of gpus')
    parser.add_argument('--export_model', type=int, default=1, help='1: export model for predict job \
                                                                     0: not export model')
    args, unkown = parser.parse_known_args()

    fit(args)
