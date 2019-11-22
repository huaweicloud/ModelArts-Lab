import os
import six
from six.moves import cPickle
import time

import numpy as np

import modelarts.aibox.ops as ops
from modelarts.aibox.pipeline import Pipeline
import modelarts.aibox.types as types


_R_MEAN = 125.31
_G_MEAN = 122.95
_B_MEAN = 113.87
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_R_STD = 62.99
_G_STD = 62.09
_B_STD = 66.70
_CHANNEL_STD = [_R_STD, _G_STD, _B_STD]

INPUT_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 1
UFF_FILE = "./resnet_cifar.uff"
DATA_DIR = "/path/your_data_path"
CALIB_FILE = "./cifar_int8_cache"
FILE_ROOT = "./test"

operator_relation_list =  [
                          {"OperatorName": "TensorRT", "OperatorDevice": "gpu"},
                          {"OperatorName": "RawReader", "OperatorDevice": "gpu"}
                          ]
ops.InitOp(operator_relation_list)


class InferencePipeline(Pipeline):
  def __init__(self, batch_size=1, num_threads=1, device_id=0, seed=-1,
               exec_pipelined=True, prefetch_queue_depth=1):
    super(InferencePipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id, seed, exec_pipelined, prefetch_queue_depth)
    self.input = ops.RawReader(device="gpu", file_root=FILE_ROOT,
                               dtype=types.AIBOXDataType.FLOAT,
                               layout_type=types.AIBOXTensorLayout.NCHW,
                               height=INPUT_SIZE, width=INPUT_SIZE, channels=NUM_CHANNELS,
                               prefetch_queue_size=prefetch_queue_depth)
    self.tensorrt = ops.TensorRT(
        device="gpu",
        uffFile=UFF_FILE,
        uffInputs=["input,{},{},{}".format(NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE)],
        outputs=["logits"],
        int8=True,
        calibrationCache=CALIB_FILE,
        workspaceSize=20,
        batchSize=batch_size)
    self.iter = 0

  def define_graph(self):
    self.inputs = self.input()
    tensorrt_out = self.tensorrt(self.inputs)
    return tensorrt_out


def read_data_files(data_dir):
  """Reads from data file and returns images and labels in a numpy array."""
  filenames = [os.path.join(data_dir, 'test_batch')]
  inputs = []
  for filename in filenames:
    with open(filename, 'rb') as f:
      encoding = {} if six.PY2 else {'encoding': 'bytes'}
      inputs.append(cPickle.load(f, **encoding))
  all_images = np.concatenate(
    [each_input[b'data'] for each_input in inputs]).astype(np.float32)
  all_labels = np.concatenate(
    [each_input[b'labels'] for each_input in inputs]).astype(np.int32)
  all_images = all_images.reshape(-1, 3, 32, 32)
  all_images = np.transpose(all_images, [0, 2, 3, 1])
  all_images = normlize(all_images, np.float32)
  all_images = np.transpose(all_images, [0, 3, 1, 2])
  return all_images, all_labels


def normlize(image, dtype=np.float32):
  cifar10_mean = np.array(_CHANNEL_MEANS,
                          dtype=dtype)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
  cifar10_std = np.array(_CHANNEL_STD,
                         dtype=dtype)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
  image -= cifar10_mean
  image /= cifar10_std
  return image


def main():
  if not os.path.exists(FILE_ROOT):
    os.makedirs(FILE_ROOT)
  images_list, labels_list = read_data_files(DATA_DIR)
  test_count = len(images_list)
  for i in range(test_count):
    filename = "./test/" + "%05d" % i + ".raw"
    images_list[i].tofile(filename)

  infer_pipe = InferencePipeline(BATCH_SIZE)
  infer_pipe.build()
  total_time = 0
  top1_predicts = []
  for i in range(test_count):
    start_time = time.time()
    pipe_out = infer_pipe.run()
    total_time += time.time() - start_time
    predict = pipe_out[0].as_cpu().as_array()[0, :, 0, 0].argsort()[-1]
    top1_predicts.append(predict)
  predict_top_1_true = 0
  for i in range(test_count):
    if labels_list[i] == top1_predicts[i]:
      predict_top_1_true += 1
  accuracy = float(predict_top_1_true) / test_count
  print('    accuracy: %.2f' % (accuracy * 100))
  # this time include H2D, the time without H2D is printed in the AIBOX
  print("avg time:{}".format(total_time * 1000 / test_count))


if __name__ == "__main__":
  main()