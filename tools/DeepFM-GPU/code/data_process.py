# coding:utf-8

import os
import pickle
import glob
import toml
import collections
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time


class DataStatsDict():
  def __init__(self, value_col_num=13, category_col_num=26, multi_category_len=()):
    self.value_col_num = value_col_num
    self.category_col_num = category_col_num
    self.multi_category_col_num = sum(multi_category_len)
    self.multi_category_len = multi_category_len
    self.field_size = value_col_num + category_col_num + self.multi_category_col_num
    self.val_cols = ["val_{}".format(i + 1) for i in range(value_col_num)]
    self.cat_cols = ["cat_{}".format(i + 1) for i in range(category_col_num)]
    self.multi_cat_cols = ["multi_cat_{}".format(i + 1) for i in range(len(multi_category_len))]
    self.val_min_dict = {col: 99999 for col in self.val_cols}
    self.val_max_dict = {col: -99999 for col in self.val_cols}
    self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}
    self.multi_cat_count_dict = {col: collections.defaultdict(int) for col in self.multi_cat_cols}
    self.oov_prefix = "OOV_"
    self.cat2id_dict = {}
    self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
    self.cat2id_dict.update({self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})
    self.cat2id_dict.update({self.oov_prefix + col: i + len(self.val_cols) + len(self.cat_cols) for i, col in enumerate(self.multi_cat_cols)})

  def stats_vals(self, val_list):
    assert len(val_list) == len(self.val_cols)

    def map_max_min(i, val):
      key = self.val_cols[i]
      if val != "":
        if float(val) > self.val_max_dict[key]:
          self.val_max_dict[key] = float(val)
        if float(val) < self.val_min_dict[key]:
          self.val_min_dict[key] = float(val)

    for i, val in enumerate(val_list):
      map_max_min(i, val)

  def stats_cats(self, cat_list):
    assert len(cat_list) == len(self.cat_cols)

    def map_cat_count(i, cat):
      key = self.cat_cols[i]
      self.cat_count_dict[key][cat] += 1

    for i, cat in enumerate(cat_list):
      map_cat_count(i, cat)

  def stats_multi_cats(self, multi_cat_list):
    assert len(multi_cat_list) == sum(self.multi_category_len)

    for multi_cat in multi_cat_list:
      _, multi_cat_id, multi_cat_value = multi_cat.split('_')
      key = "multi_cat_%s" % (int(multi_cat_id) + 1)
      self.multi_cat_count_dict[key][multi_cat_value] += 1

  def save_dict(self, output_path, prefix=""):
    with open(os.path.join(output_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.val_max_dict, file_wrt)
    with open(os.path.join(output_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.val_min_dict, file_wrt)
    with open(os.path.join(output_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.cat_count_dict, file_wrt)
    with open(os.path.join(output_path, "{}multi_cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.multi_cat_count_dict, file_wrt)

  def load_dict(self, dict_path, prefix=""):
    with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.val_max_dict = pickle.load(file_wrt)
    with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.val_min_dict = pickle.load(file_wrt)
    with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.cat_count_dict = pickle.load(file_wrt)
    with open(os.path.join(dict_path, "{}multi_cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.multi_cat_count_dict = pickle.load(file_wrt)
    print("val_max_dict.items()[:50]: {}".format(list(self.val_max_dict.items())))
    print("val_min_dict.items()[:50]: {}".format(list(self.val_min_dict.items())))

  def get_cat2id(self, threshold=100):
    for key, cat_count_d in self.cat_count_dict.items():
      new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
      for cat_str, count in new_cat_count_d.items():
        self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
    for key, multi_cat_count_d in self.multi_cat_count_dict.items():
      new_multi_cat_count_d = dict(filter(lambda x: x[1] > threshold, multi_cat_count_d.items()))
      for multi_cat_str, count in new_multi_cat_count_d.items():
        if multi_cat_str == 'OOV':
          continue
        self.cat2id_dict[key + "_" + multi_cat_str] = len(self.cat2id_dict)

    print("data vocab size: {}".format(len(self.cat2id_dict)))
    print("data vocab size[:50]: {}".format(list(self.cat2id_dict.items())[:50]))

  def map_cat2id(self, values, cats, multi_cats):
    def minmax_scale_value(i, val):
      min_v = float(self.val_min_dict["val_{}".format(i + 1)])
      max_v = float(self.val_max_dict["val_{}".format(i + 1)])
      if val >= max_v:
        return 1.0
      elif val <= min_v or max_v == min_v:
        return 0.0
      else:
        return float(val - min_v) / (max_v - min_v)

    id_list = []
    weight_list = []
    for i, val in enumerate(values):
      if val == "":
        id_list.append(i)
        weight_list.append(0)
      else:
        key = "val_{}".format(i + 1)
        id_list.append(self.cat2id_dict[key])
        weight_list.append(minmax_scale_value(i, float(val)))

    for i, cat_str in enumerate(cats):
      key = "cat_{}".format(i + 1) + "_" + cat_str
      if key in self.cat2id_dict:
        id_list.append(self.cat2id_dict[key])
      else:
        id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
      weight_list.append(1.0)

    for i, multi_cat_str in enumerate(multi_cats):
      _, multi_cat_id, multi_cat_value = multi_cat_str.split('_')
      multi_cat_id = int(multi_cat_id)
      if multi_cat_value == 'OOV':
        key = "OOV_multi_cat_%s" % (multi_cat_id + 1)
      else:
        key = "multi_cat_%s_%s" % (multi_cat_id + 1, multi_cat_value)

      if key in self.cat2id_dict:
        id_list.append(self.cat2id_dict[key])
      else:
        id_list.append(self.cat2id_dict[self.oov_prefix + "multi_cat_{}".format(multi_cat_id + 1)])

      weight_list.append(1.0)

    return id_list, weight_list


def mkdir_path(file_path):
  if not os.path.exists(file_path):
    os.makedirs(file_path)


def statsdata(data_file_path, output_path, data_stats):
  with open(data_file_path, encoding="utf-8") as file_in:
    errorline_list = []
    count = 0
    for line in file_in:
      count += 1
      line = line.strip("\n")
      items = line.split("\t")
      if len(items) != data_stats.field_size + 1:  # feature columns; +  label_col
        errorline_list.append(count)
        print("line: {}".format(line))

        raise ValueError(
          "Expect column count is {}, real column count is {}, please check "
          "your value_col_num and category_col_num. "
          "\nError line number: {}, Error line content: {}".format(
            data_stats.field_size + 1, len(items), count - 1, line))

      if count % 1000000 == 0:
        print("Have handle {}w lines.".format(count // 10000))
      label = items[0]
      features = items[1:]
      values = features[:data_stats.value_col_num]
      cats = features[data_stats.value_col_num:data_stats.value_col_num + data_stats.category_col_num]
      multi_cats = features[data_stats.value_col_num + data_stats.category_col_num:]
      assert len(values) == data_stats.value_col_num, "values.size： {}".format(len(values))
      assert len(cats) == data_stats.category_col_num, "cats.size： {}".format(len(cats))
      assert len(multi_cats) == data_stats.multi_category_col_num, "multi-cats.size： {}".format(len(multi_cats))
      data_stats.stats_vals(values)
      data_stats.stats_cats(cats)
      data_stats.stats_multi_cats(multi_cats)
  data_stats.save_dict(output_path)


def add_write(file_path, wrt_str):
  with open(file_path, 'a', encoding="utf-8") as file_out:
    file_out.write(wrt_str + "\n")


def get_file_line_count(file_path):
  line_count = 0
  with open(file_path, 'r', encoding="utf-8") as file_in:
    for line in file_in:
      line = line.strip("\n")
      if line == "":
        continue
      line_count += 1
  return line_count


def random_split_trans2h5(in_file_path, output_path, data_stats, part_rows=2000000, test_size=0.1,
                          seed=2020, output_format='h5'):
  value_col_num = data_stats.value_col_num
  category_col_num = data_stats.category_col_num
  multi_category_col_num = data_stats.multi_category_col_num
  train_line_count = get_file_line_count(in_file_path)
  test_size = int(train_line_count * test_size)
  train_size = train_line_count - test_size
  all_indices = [i for i in range(train_line_count)]
  np.random.seed(seed)
  np.random.shuffle(all_indices)
  print("all_indices.size: {}".format(len(all_indices)))
  lines_count_dict = collections.defaultdict(int)
  test_indices_set = set(all_indices[: test_size])
  print("test_indices_set.size: {}".format(len(test_indices_set)))
  print("----------" * 10 + "\n" * 2)

  train_feature_file_name = os.path.join(output_path, "train_input_part_{}.h5")
  train_label_file_name = os.path.join(output_path, "train_output_part_{}.h5")
  test_feature_file_name = os.path.join(output_path, "test_input_part_{}.h5")
  test_label_file_name = os.path.join(output_path, "test_output_part_{}.h5")
  train_feature_list = []
  train_label_list = []
  test_feature_list = []
  test_label_list = []
  ids_len = 0
  filtered_train_size = 0
  filtered_test_size = 0
  with open(in_file_path, encoding="utf-8") as file_in:
    count = 0
    train_part_number = 0
    test_part_number = 0
    for i, line in enumerate(file_in):
      count += 1
      if count % 1000000 == 0:
        print("Have handle {}w lines.".format(count // 10000))
      line = line.strip("\n")
      items = line.split("\t")
      if len(items) != 1 + value_col_num + category_col_num + multi_category_col_num:
        continue
      label = float(items[0])
      values = items[1:value_col_num+1]
      cats = items[value_col_num+1:value_col_num+category_col_num+1]
      multi_cats = items[value_col_num+category_col_num+1:]
      assert len(values) == value_col_num, "values.size： {}".format(len(values))
      assert len(cats) == category_col_num, "cats.size： {}".format(len(cats))
      assert len(multi_cats) == multi_category_col_num, "multi-cats.size： {}".format(len(multi_cats))
      ids, wts = data_stats.map_cat2id(values, cats, multi_cats)
      ids_len = len(ids)
      if i not in test_indices_set:
        train_feature_list.append(ids + wts)
        train_label_list.append(label)
      else:
        test_feature_list.append(ids + wts)
        test_label_list.append(label)
      if (len(train_label_list) > 0) and (len(train_label_list) % part_rows == 0):
        if output_format == 'h5':
          pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                              key="fixed")
          pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number), key="fixed")
        else:
          with open(os.path.join(output_path, 'train_part_{}.txt'.format(train_part_number)), 'w') as f:
            for i in range(len(train_feature_list)):
              train_feature = [str(s) for s in train_feature_list[i]]
              train_label = str(int(train_label_list[i]))
              f.write(train_label + ' ' + ' '.join(train_feature) + '\n')
        filtered_train_size += len(train_feature_list)
        train_feature_list = []
        train_label_list = []
        train_part_number += 1
      if (len(test_label_list) > 0) and (len(test_label_list) % part_rows == 0):
        if output_format == 'h5':
          pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number), key="fixed")
          pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number), key="fixed")
        else:
          with open(os.path.join(output_path, 'test_part_{}.txt'.format(test_part_number)), 'w') as f:
            for i in range(len(test_feature_list)):
              test_feature = [str(s) for s in test_feature_list[i]]
              test_label = str(int(test_label_list[i]))
              f.write(test_label + ' ' + ' '.join(test_feature) + '\n')
        filtered_test_size += len(test_feature_list)
        test_feature_list = []
        test_label_list = []
        test_part_number += 1

    if len(train_label_list) > 0:
      filtered_train_size += len(train_feature_list)
      if output_format == 'h5':
        pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                            key="fixed")
        pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number), key="fixed")
      else:
        with open(os.path.join(output_path, 'train_part_{}.txt'.format(train_part_number)),
                  'w') as f:
          for i in range(len(train_feature_list)):
            train_feature = [str(s) for s in train_feature_list[i]]
            train_label = str(int(train_label_list[i]))
            f.write(train_label + ' ' + ' '.join(train_feature) + '\n')
    if len(test_label_list) > 0:
      filtered_test_size += len(test_feature_list)
      if output_format == 'h5':
        pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number), key="fixed")
        pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number), key="fixed")
      else:
        with open(os.path.join(output_path, 'test_part_{}.txt'.format(test_part_number)), 'w') as f:
          for i in range(len(test_feature_list)):
            test_feature = [str(s) for s in test_feature_list[i]]
            test_label = str(int(test_label_list[i]))
            f.write(test_label + ' ' + ' '.join(test_feature) + '\n')

    num_features = len(data_stats.cat2id_dict)
    num_inputs = ids_len

  return num_features, filtered_train_size, filtered_test_size, num_inputs

def fix_multi_cat(data_file_path, multi_cat_col_num, multi_category_len, output_dir, file_pattern, feat_sep, multi_category_sep):

  multi_cat_len = [0 for _ in range(multi_cat_col_num)]

  import glob
  if os.path.isdir(data_file_path):
    data_files_list = glob.glob(os.path.join(data_file_path, file_pattern))
  else:
    data_files_list = [data_file_path]

  for data_file in data_files_list:
    with open(data_file, 'r') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue

        items = line.split(feat_sep)
        multi_cat_items = items[len(items) - multi_cat_col_num:]
        for i, multi_cat in enumerate(multi_cat_items):
          multi_cat_len[i] = max(multi_cat_len[i], len(multi_cat.split(multi_category_sep)))

    for i in range(len(multi_cat_len)):
      if multi_category_len[i] is not None and multi_category_len[i] >= 0:
        multi_cat_len[i] = multi_category_len[i]
  new_data_file_path = os.path.join(output_dir, 'fixed.txt')

  with open(new_data_file_path, 'w') as fw:
    for data_file in data_files_list:
      with open(data_file, 'r') as fr:
        for line in fr:
          line = line.strip()
          if not line:
            continue

          items = line.split(feat_sep)
          ok_items = items[:len(items) - multi_cat_col_num]
          fw.write('\t'.join(ok_items))
          multi_cat_items = items[len(items) - multi_cat_col_num:]

          for i, multi_cat in enumerate(multi_cat_items):
            fw.write('\t')
            c_list = multi_cat.split(multi_category_sep)
            c_list = c_list[:multi_cat_len[i]]
            fw.write('\t'.join(['m_%d_%s' % (i, c) for c in c_list]))
            padding_len = multi_cat_len[i] - len(c_list)
            if padding_len > 0:
              fw.write('\t')
              fw.write('\t'.join(['m_%d_OOV' % i for _ in range(padding_len)]))

          fw.write('\n')

  return new_data_file_path, multi_cat_len


def convert_tfrecords(num_inputs, input_filename, output_filename, samples_per_line):
  # label id1,id2,...,idn   val1,val2,...,valn
  with open(input_filename, "r") as rf:
    line_num = 0

    writer = tf.python_io.TFRecordWriter(output_filename)
    print("Starting to convert {} to {}...".format(input_filename, output_filename))

    ids = []
    values = []
    labels = []
    new_line_num = 1
    while True:
      line = rf.readline()
      if not line:
        break

      data = line.split(" ")
      label = float(data[0])
      id_list = [int(id) for id in data[1: num_inputs + 1]]
      val_list = [float(val) for val in data[num_inputs + 1:]]

      labels.append(label)
      ids.extend(id_list)
      values.extend(val_list)
      line_num += 1
      # Write samples one by one
      if line_num % samples_per_line == 0:
        # print("new line num is %d" % new_line_num)
        assert (len(ids) == num_inputs * samples_per_line)
        new_line_num += 1
        example = tf.train.Example(features=tf.train.Features(feature={
          "label":
            tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
          "feat_ids":
            tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
          "feat_vals":
            tf.train.Feature(float_list=tf.train.FloatList(value=values))
        }))
        writer.write(example.SerializeToString())
        ids = []
        values = []
        labels = []
    writer.close()
    # drop data not satisfy samples_per_line
    print("Starting to convert {} to {} done...".format(input_filename, output_filename))


def main():
  parser = argparse.ArgumentParser(description='Get and Process datasets')

  parser.add_argument('--data_file_path', type=str, default='./raw_data/new_part.txt', help='data file path')
  parser.add_argument('--value_col_num', type=int, default=22,
                      help='continue value column number of data_file_path file.')
  parser.add_argument('--category_col_num', type=int, default=17,
                      help='category column number of data_file_path file. ')
  parser.add_argument('--multi_category_col_num', type=int, default=0,
                      help='multi category column number of data_file_path file. ')
  parser.add_argument('--multi_category_sep', type=str, default=',',
                      help='multi category separator')
  parser.add_argument('--feat_sep', type=str, default='\t',
                      help='features separator')
  parser.add_argument('--multi_category_len', type=list, default=[None, None],
                      help='cut multi category len to this, if None, max len will be used')
  parser.add_argument('--output_path', type=str, default='./raw_data_output', help='The path to save h5 dataset')
  parser.add_argument('--output_format', type=str, default='txt', help='txt or h5')
  parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio')
  parser.add_argument('--part_rows', type=int, default=2000000, help='saved samples of each file')
  parser.add_argument('--threshold', type=int, default=0, help='threshold to filter vocab')
  parser.add_argument('--stats_output_path', type=str, default=None, help='load stats dict')
  parser.add_argument('--line_per_sample', type=int, default=1000,
                      help='The number of samples stored in each record of the tfrecord file')
  parser.add_argument('--file_pattern', type=str, default='*',
                      help='data file pattern')
  parser.add_argument('--train_size', type=int, default=None, help='')
  parser.add_argument('--test_size', type=int, default=None, help='')

  args, _ = parser.parse_known_args()
  s_time = time.time()

  mkdir_path(args.output_path)
  new_data_file_path, new_multi_category_len = fix_multi_cat(args.data_file_path,
                                                             args.multi_category_col_num,
                                                             args.multi_category_len,
                                                             args.output_path,
                                                             args.file_pattern,
                                                             args.feat_sep,
                                                             args.multi_category_sep)
  print(new_data_file_path, new_multi_category_len)

  args.data_file_path = new_data_file_path
  args.multi_category_len = new_multi_category_len

  data_stats = DataStatsDict(value_col_num=args.value_col_num,
                             category_col_num=args.category_col_num,
                             multi_category_len=args.multi_category_len)

  # step 1, stats the vocab and normalize value
  if not args.stats_output_path:
    stats_output_path = os.path.join(args.output_path, "stats_dict")
    mkdir_path(stats_output_path)
    statsdata(args.data_file_path, stats_output_path, data_stats)
  else:
    stats_output_path = args.stats_output_path

  data_stats.load_dict(dict_path=stats_output_path, prefix="")
  data_stats.get_cat2id(threshold=args.threshold)

  # step 2, transform data trans2h5; version 2: np.random.shuffle
  data_save_path = os.path.join(args.output_path, 'preprocessed_data')
  mkdir_path(data_save_path)
  num_features, train_size, test_size, num_inputs = \
    random_split_trans2h5(args.data_file_path, data_save_path, data_stats,
                          part_rows=args.part_rows, test_size=args.test_ratio, seed=2020,
                          output_format=args.output_format)

  # save params for train and inference
  param = {'num_features': num_features,
           'num_inputs': num_inputs,
           'train_size': args.train_size or train_size,
           'test_size': args.test_size or test_size,
           'value_col_num': args.value_col_num,
           'category_col_num': args.category_col_num,
           'multi_category_len': args.multi_category_len,
           'line_per_sample': args.line_per_sample,
           'threshold': args.threshold,
           'feat_sep': args.feat_sep,
           'multi_category_sep': args.multi_category_sep,
           'train_tag': 'train_part',
           'test_tag': 'test_part'}
  with open(os.path.join(args.output_path, 'param.toml'), 'w') as f:
    toml.dump(param, f)

  for k, v in param.items():
    print('%s is %s' % (k, v))

  # convert to tfrecord data
  tfrecord_dir = os.path.join(args.output_path, 'tfrecord')
  mkdir_path(tfrecord_dir)

  for input_file in glob.glob(os.path.join(data_save_path, '*')):
    output_name = '%s.tfrecord' % os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(tfrecord_dir, output_name)
    convert_tfrecords(num_inputs, input_file, output_file, args.line_per_sample)

  e_time = time.time()
  print('cost total time is ', e_time - s_time)


if __name__ == '__main__':
  main()
