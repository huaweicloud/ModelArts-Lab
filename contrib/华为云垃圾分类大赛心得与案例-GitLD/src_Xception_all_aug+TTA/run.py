# -*- coding: utf-8 -*-
'''
基于resnet50实现的垃圾分类代码
使用方法：
（1）训练
cd {run.py所在目录}
python run.py --data_url='../datasets/garbage_classify/train_data' --train_url='../model_snapshots' 
--num_classes=40 --deploy_script_path='./deploy_scripts' --test_data_url='../datasets/test_data' 
--max_epochs=10

（2）转pb
cd {run.py所在目录}
python run.py --mode=save_pb --deploy_script_path='./deploy_scripts' 
--freeze_weights_file_path='../model_snapshots/weights_000_0.9811.h5' --num_classes=40

（3）评价
cd {run.py所在目录}
python run.py --mode=eval --eval_pb_path='../model_snapshots/model' --num_classes=40 
--test_data_url='../datasets/test_data'
'''
import os
import tensorflow as tf
from moxing.framework import file

tf.app.flags.DEFINE_string('mode', 'train', 'optional: train, save_pb, eval')
tf.app.flags.DEFINE_string('local_data_root', '/cache/',
                           'a directory used for transfer data between local path and OBS path')
# params for pretrained
tf.app.flags.DEFINE_string('pretrained_weight_local', '',
                           'a pretrained weight on local')
tf.app.flags.DEFINE_string('pretrained_weight_url', '',
                           'a pretrained weight on obs')
# params for train
tf.app.flags.DEFINE_string('data_url', '', 'the training data path')
tf.app.flags.DEFINE_string('restore_model_path', '',
                           'a history model you have trained, you can load it and continue trainging')
tf.app.flags.DEFINE_string('train_url', '', 'the path to save training outputs')
tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
                            'the max num of weights files keeps, if set -1, means infinity')
tf.app.flags.DEFINE_integer('num_classes', 0, 'the num of classes which your task should classify')
tf.app.flags.DEFINE_integer('input_size', 224, 'the input image size of the model')
tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, '')
tf.app.flags.DEFINE_integer('max_epochs', 5, '')

# params for save pb
tf.app.flags.DEFINE_string('deploy_script_path', '',
                           'a path which contain config.json and customize_service.py, '
                           'if it is set, these two scripts will be copied to {train_url}/model directory')
tf.app.flags.DEFINE_string('freeze_weights_file_path', '',
                           'if it is set, the specified h5 weights file will be converted as a pb model, '
                           'only valid when {mode}=save_pb')

# params for evaluation
tf.app.flags.DEFINE_string('eval_weights_path', '', 'weights file path need to be evaluate')
tf.app.flags.DEFINE_string('eval_pb_path', '', 'a directory which contain pb file needed to be evaluate')
tf.app.flags.DEFINE_string('test_data_url', '', 'the test data path on obs')

tf.app.flags.DEFINE_string('data_local', '', 'the train data path on local')
tf.app.flags.DEFINE_string('train_local', '', 'the training output results on local')
tf.app.flags.DEFINE_string('test_data_local', '', 'the test data path on local')
tf.app.flags.DEFINE_string('tmp', '', 'a temporary path on local')

FLAGS = tf.app.flags.FLAGS


def check_args(FLAGS):
    if FLAGS.mode not in ['train', 'save_pb', 'eval']:
        raise Exception('FLAGS.mode error, should be train, save_pb or eval')
    if FLAGS.num_classes == 0:
        raise Exception('FLAGS.num_classes error, '
                        'should be a positive number associated with your classification task')

    if FLAGS.mode == 'train':
        if FLAGS.data_url == '':
            raise Exception('you must specify FLAGS.data_url')
        if not file.exists(FLAGS.data_url):
            raise Exception('FLAGS.data_url: %s is not exist' % FLAGS.data_url)
        if FLAGS.restore_model_path != '' and (not file.exists(FLAGS.restore_model_path)):
            raise Exception('FLAGS.restore_model_path: %s is not exist' % FLAGS.restore_model_path)
        if file.is_directory(FLAGS.restore_model_path):
            raise Exception('FLAGS.restore_model_path must be a file path, not a directory, %s' % FLAGS.restore_model_path)
        if FLAGS.train_url == '':
            raise Exception('you must specify FLAGS.train_url')
        elif not file.exists(FLAGS.train_url):
            file.make_dirs(FLAGS.train_url)
        if FLAGS.deploy_script_path != '' and (not file.exists(FLAGS.deploy_script_path)):
            raise Exception('FLAGS.deploy_script_path: %s is not exist' % FLAGS.deploy_script_path)
        if FLAGS.deploy_script_path != '' and file.exists(FLAGS.train_url + '/model'):
            raise Exception(FLAGS.train_url +
                            '/model is already exist, only one model directoty is allowed to exist')
        if FLAGS.test_data_url != '' and (not file.exists(FLAGS.test_data_url)):
            raise Exception('FLAGS.test_data_url: %s is not exist' % FLAGS.test_data_url)

    if FLAGS.mode == 'save_pb':
        if FLAGS.deploy_script_path == '' or FLAGS.freeze_weights_file_path == '':
            raise Exception('you must specify FLAGS.deploy_script_path '
                            'and FLAGS.freeze_weights_file_path when you want to save pb')
        if not file.exists(FLAGS.deploy_script_path):
            raise Exception('FLAGS.deploy_script_path: %s is not exist' % FLAGS.deploy_script_path)
        if not file.is_directory(FLAGS.deploy_script_path):
            raise Exception('FLAGS.deploy_script_path must be a directory, not a file path, %s' % FLAGS.deploy_script_path)
        if not file.exists(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path: %s is not exist' % FLAGS.freeze_weights_file_path)
        if file.is_directory(FLAGS.freeze_weights_file_path):
            raise Exception('FLAGS.freeze_weights_file_path must be a file path, not a directory, %s ' % FLAGS.freeze_weights_file_path)
        if file.exists(FLAGS.freeze_weights_file_path.rsplit('/', 1)[0] + '/model'):
            raise Exception('a model directory is already exist in ' + FLAGS.freeze_weights_file_path.rsplit('/', 1)[0]
                            + ', please rename or remove the model directory ')

    if FLAGS.mode == 'eval':
        if FLAGS.eval_weights_path == '' and FLAGS.eval_pb_path == '':
            raise Exception('you must specify FLAGS.eval_weights_path '
                            'or FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and FLAGS.eval_pb_path != '':
            raise Exception('you must specify only one of FLAGS.eval_weights_path '
                            'and FLAGS.eval_pb_path when you want to evaluate a model')
        if FLAGS.eval_weights_path != '' and (not file.exists(FLAGS.eval_weights_path)):
            raise Exception('FLAGS.eval_weights_path: %s is not exist' % FLAGS.eval_weights_path)
        if FLAGS.eval_pb_path != '' and (not file.exists(FLAGS.eval_pb_path)):
            raise Exception('FLAGS.eval_pb_path: %s is not exist' % FLAGS.eval_pb_path)
        if not file.is_directory(FLAGS.eval_pb_path) or (not FLAGS.eval_pb_path.endswith('model')):
            raise Exception('FLAGS.eval_pb_path must be a directory named model '
                            'which contain saved_model.pb and variables, %s' % FLAGS.eval_pb_path)
        if FLAGS.test_data_url == '':
            raise Exception('you must specify FLAGS.test_data_url when you want to evaluate a model')
        if not file.exists(FLAGS.test_data_url):
            raise Exception('FLAGS.test_data_url: %s is not exist' % FLAGS.test_data_url)


def main(argv=None):
    check_args(FLAGS)
            
    # Create some local cache directories used for transfer data between local path and OBS path
    if not FLAGS.data_url.startswith('s3://'):
        FLAGS.data_local = FLAGS.data_url
    else:
        FLAGS.data_local = os.path.join(FLAGS.local_data_root, 'train_data/')
        if not os.path.exists(FLAGS.data_local):
            file.copy_parallel(FLAGS.data_url, FLAGS.data_local)
        else:
            print('FLAGS.data_local: %s is already exist, skip copy' % FLAGS.data_local)
        
    if not FLAGS.train_url.startswith('s3://'):
        FLAGS.train_local = FLAGS.train_url
    else:
        FLAGS.train_local = os.path.join(FLAGS.local_data_root, 'model_snapshots/')
        if not os.path.exists(FLAGS.train_local):
            os.mkdir(FLAGS.train_local)
    
    if not FLAGS.test_data_url.startswith('s3://'):
        FLAGS.test_data_local = FLAGS.test_data_url
    else:
        FLAGS.test_data_local = os.path.join(FLAGS.local_data_root, 'test_data/')
        if not os.path.exists(FLAGS.test_data_local):
            file.copy_parallel(FLAGS.test_data_url, FLAGS.test_data_local)
        else:
            print('FLAGS.test_data_local: %s is already exist, skip copy' % FLAGS.test_data_local)
    
    FLAGS.tmp = os.path.join(FLAGS.local_data_root, 'tmp/')
    if not os.path.exists(FLAGS.tmp):
        os.mkdir(FLAGS.tmp)

    if FLAGS.mode == 'train':
        from train import train_model
        train_model(FLAGS)
    elif FLAGS.mode == 'save_pb':
        from save_model import load_weights_save_pb
        load_weights_save_pb(FLAGS)
    elif FLAGS.mode == 'eval':
        from eval import eval_model
        eval_model(FLAGS)

if __name__ == '__main__':
    tf.app.run()
