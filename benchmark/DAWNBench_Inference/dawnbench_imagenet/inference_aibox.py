from glob import glob
import os
import time

import modelarts.aibox.ops as ops
from modelarts.aibox.pipeline import Pipeline
import modelarts.aibox.tfrecord as tfrec
import modelarts.aibox.types as types

from configs import DATA_DIR, IDX_DIR, FILE_ROOT, CALIB_FILE, MODEL_FILE

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256
INPUT_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 1

operator_relation_list = [
                          {"OperatorName": "ImageDecoder", "OperatorDevice": "cpu"},
                          {"OperatorName": "Crop", "OperatorDevice": "cpu"},
                          {"OperatorName": "Resize", "OperatorDevice": "gpu"},
                          {"OperatorName": "CropMirrorNormalize", "OperatorDevice": "gpu"},
                          {"OperatorName": "NormalizePermute", "OperatorDevice": "gpu"},
                          {"OperatorName": "TensorRT", "OperatorDevice": "gpu"},
                          {"OperatorName": "RawReader", "OperatorDevice": "gpu"}
                          ]
ops.InitOp(operator_relation_list)

class DataPipeline(Pipeline):
    def __init__(self, batch_size, tfrec_filenames, tfrec_idx_filenames, num_threads=1, device_id=0, seed=-1,
                 exec_pipelined=True, prefetch_queue_depth=1):
        super(DataPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id, seed, exec_pipelined, prefetch_queue_depth)
        self.input = ops.TFRecordReader(path=tfrec_filenames,
                                        index_path=tfrec_idx_filenames,
                                        initial_fill=10000,
                                        features={
                                            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1)})
        self.decode = ops.ImageDecoder(device="cpu",
                                       output_type=types.AIBOXImageType.RGB)

        self.resize = ops.Resize(device="gpu", resize_shorter=_RESIZE_MIN)
        self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.AIBOXDataType.FLOAT,
                                           crop=(INPUT_SIZE, INPUT_SIZE), image_type=types.AIBOXImageType.RGB,
                                           mean=_CHANNEL_MEANS, std=[1, 1, 1], output_layout=types.AIBOXTensorLayout.NCHW)

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"].cpu()
        images_out = self.decode(images)
        resize_out = self.resize(images_out.gpu())
        cmnp_out = self.cmnp(resize_out)
        return cmnp_out, labels


class InferencePipeline(Pipeline):
  def __init__(self, batch_size, num_threads=1, device_id=0, seed=-1,
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
        uffFile=MODEL_FILE,
        calibrationCache=CALIB_FILE,
        uffInputs=["input,{},{},{}".format(NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE)],
        outputs=["logits"],
        int8=True,
        workspaceSize=20,
        batchSize=1)
    self.iter = 0

  def define_graph(self):
    self.inputs = self.input()
    tensorrt_out = self.tensorrt(self.inputs)
    return tensorrt_out


def get_files(data_dir, filename_pattern):
  if data_dir == None:
    return []
  files = glob(os.path.join(data_dir, filename_pattern))
  if files == []:
    raise ValueError('Can not find any files in {} with pattern "{}"'.format(
        data_dir, filename_pattern))
  return files


def get_tfrecords_count(files):
  import tensorflow as tf
  num_records = 0
  for fn in files:
    for record in tf.python_io.tf_record_iterator(fn):
      num_records += 1
  return num_records


def main():
  if not os.path.exists(FILE_ROOT):
    os.makedirs(FILE_ROOT)
  files = sorted(get_files(DATA_DIR, 'validation*'))
  num_records = get_tfrecords_count(files)
  idx_files = sorted(get_files(IDX_DIR, 'validation*'))
  data_pipe = DataPipeline(BATCH_SIZE, tfrec_filenames=files, tfrec_idx_filenames=idx_files)
  data_pipe.build()
  true_labels = []
  for i in range(num_records):
    pipe_out = data_pipe.run()
    images_gpu, label_cpu = pipe_out
    images_cpu = images_gpu.as_cpu().as_array()
    filename = "./test/" + "%05d" % i + ".raw"
    images_cpu.tofile(filename)
    true_labels.append(label_cpu.as_array())

  infer_pipe = InferencePipeline(BATCH_SIZE)
  infer_pipe.build()
  total_time = 0
  top5_predicts = []
  for i in range(num_records):
    start_time = time.time()
    pipe_out = infer_pipe.run()
    total_time += time.time() - start_time
    predict = pipe_out[0].as_cpu().as_array()[0, :, 0, 0].argsort()[-5:][::-1]
    top5_predicts.append(predict)
  predict_top_5_true = 0
  for i in range(num_records):
    if true_labels[i] in top5_predicts[i]:
      predict_top_5_true += 1
  accuracy = float(predict_top_5_true) / num_records
  print('    accuracy: %.2f' % (accuracy * 100))
  # this time include H2D, the time without H2D is printed in the AIBOX
  print("avg time:{}".format(total_time * 1000 / num_records))


if __name__ == "__main__":
  main()