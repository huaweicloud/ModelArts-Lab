import os
from subprocess import call
import tensorflow as tf

from configs import DATA_DIR, IDX_DIR


def get_files(data_dir, filename_pattern):
  if data_dir == None:
    return []
  files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
  if files == []:
    raise ValueError('Can not find any files in {} with pattern "{}"'.format(
      data_dir, filename_pattern))
  return files


def main():
  files = sorted(get_files(DATA_DIR, 'validation*'))
  tfrecord2idx_script = "tfrecord2idx"
  if not os.path.exists(IDX_DIR):
    os.mkdir(IDX_DIR)
  for i in range(len(files)):
    tfrecord = files[i]
    tfrecord_idx = "{}/{}.idx".format(IDX_DIR, tfrecord.split("/")[-1])
    if not os.path.isfile(tfrecord_idx):
      call([tfrecord2idx_script, tfrecord, tfrecord_idx])
  print("Create index done.")


if __name__ == "__main__":
  main()