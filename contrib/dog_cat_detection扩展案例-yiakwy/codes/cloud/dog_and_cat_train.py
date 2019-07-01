import os
import sys
import argparse
import functools
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K
try:
    from moxing.framework import file as CloudAPI
except:
    CloudAPI = None
import logging

logging.basicConfig(level=logging.INFO)

def add_path(path):
    path = os.path.abspath(path)
    if path not in sys.path:
        logging.info("load path %s" % path)
        sys.path.insert(0, path)

pwd = os.path.dirname(os.path.realpath(__file__))
logging.info(pwd)

# Add config to python path
add_path(os.path.join(pwd, '..', 'config'))
add_path(os.path.join(pwd, '.'))
add_path(os.path.join(pwd, 'models'))

from config import Settings
from models import VGG16

class CatDogConfig(Settings):
    pass

config = CatDogConfig("settings")
logging.info("DATA DIR: %s" % config.DATA_DIR)

def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

def parse_args(raw_args):
    """Huawei Cloud Case
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg("max_epochs", int, 30, 'Number of trainning iterations')
    add_arg("data_url", str, config.DATA_DIR, 'S3 dataset directory path')
    add_arg("train_url", str, os.path.join(config.OUTPUT_DIR, "services"), 'S3 trainning model output directory path')
    add_arg("batch_size", int, 32, "Number of training iterations" )
    add_arg("num_gpus", int , 1, 'Number of GPUs')
    # directly used by dataset object
    add_arg("dataset_name", str, 'dog_and_cat_200', 'Name of the used dogs and cats dataset')

    FLAGS = parser.parse_args(raw_args)
    return FLAGS

# adopt from https://github.com/huaweicloud/ModelArts-Lab/blob/master/train_inference/image_recognition/codes/dog_and_cat_train.py
# convert Keras model to TF model
def save_keras_model_to_serving(model, export_path):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'images': model.inputs[0]}, outputs={'logits': model.outputs[0]})
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'segmentation': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()

def Program(raw_args):
    FLAGS = parse_args(raw_args)
    
    # update config object
    config.EPOCHS = FLAGS.max_epochs
    config.BATCH_SIZE = FLAGS.batch_size
    config.GPUS = FLAGS.num_gpus

    # sync files from S3 to local storage unit
    if CloudAPI is not None:
        CloudAPI.copy_parallel(FLAGS.data_url, config.DATA_DIR)
    
    # Load Models
    SAVER="{}/catdog".format(config.OUTPUT_DIR)

    if not os.path.isdir(SAVER):
        os.makedirs(SAVER)

    model = VGG16("training", config, SAVER)
    logging.info(model.summary())
    
    # Load pretrained weights, see `notebooks/ModelArts-Explore_ex1`
    check_point = "{}/weights.best.checkpoint.hdf5".format(SAVER)
    if os.path.isfile(check_point):
        model.load_weights()
    else:
        model.load_weights(config.CAT_DOG_PRETRAINED_MODEL)

    # Prepare data
    from dataset import CatDogDataset, Preprocess_img
    cat_dog_dataset = CatDogDataset(name=FLAGS.dataset_name)

    cat_dog_dataset.load_dataset()
    X_train, y_train = cat_dog_dataset.train_reader()
    X_test, _ = cat_dog_dataset.validation_reader()

    # Trainning
    start = timeit.default_timer()
    # For large dataset, we prefer to use SGD to digest dataset quickly
    model.fit(X_train, y_train, optimizer_type="sgd")
    elapsed = timeit.default_timer() - start
    logging.info("Trainnig complete, elapsed: %s(s)" % elapsed)

    predictions = []
    detected = model.infer(X_test)

    for ret in detected:
        predictions.append(np.argmax(ret))
        
    df = pd.DataFrame({
        'data': cat_dog_dataset.test_data,
        'labels': cat_dog_dataset.test_labels,
        'prediction': predictions
    })

    print("evaluation snapshot, top 10: ", df.head(10))

    acc = accuracy_score(cat_dog_dataset.test_labels, predictions)

    print('训练得到的猫狗识别模型的准确度是-pure VGG16：',acc)

    # save accuracy to a local file
    metric_file_name = os.path.join(SAVER, 'metric.json')
    metric_file_content = """
{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}
    """ % acc

    with open(metric_file_name, "w") as f:
        f.write(metric_file_content)
    
    model_proto = "{}/model".format(SAVER)
    if os.path.isdir(model_proto):
        os.system('rm -rf %s' % model_proto)
    save_keras_model_to_serving(model.model, model_proto)

    EXPORTED_PATH="{}/model/preprocessor.json".format(SAVER)
    logging.info("persist preprocessor data to %s" % EXPORTED_PATH)
    cat_dog_dataset.preprocessor.save(EXPORTED_PATH)
    
    # copy local output to remote S3 storage unit
    if CloudAPI is not None:
        CloudAPI.copy_parallel(SAVER, FLAGS.train_url)

    # check
    preprocessor = Preprocess_img()
    preprocessor.load_from(EXPORTED_PATH)
    
