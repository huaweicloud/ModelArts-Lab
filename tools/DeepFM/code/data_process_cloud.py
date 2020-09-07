# coding:utf-8

import os
import pickle
import collections
import argparse

import numpy as np
import pandas as pd
import tempfile

import moxing as mox


class DataStatsDict():
  def __init__(self, value_col_num=13, category_col_num=26):
    self.value_col_num = value_col_num
    self.category_col_num = category_col_num
    self.field_size = value_col_num + category_col_num  # value_1-13;  cat_1-26;
    self.val_cols = ["val_{}".format(i + 1) for i in range(value_col_num)]
    self.cat_cols = ["cat_{}".format(i + 1) for i in range(category_col_num)]
    self.val_min_dict = {col: 0 for col in self.val_cols}
    self.val_max_dict = {col: 0 for col in self.val_cols}
    self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}
    self.oov_prefix = "OOV_"
    self.cat2id_dict = {}
    self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
    self.cat2id_dict.update({self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

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

  def save_dict(self, output_path, prefix=""):
    with open(os.path.join(output_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.val_max_dict, file_wrt)
    with open(os.path.join(output_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.val_min_dict, file_wrt)
    with open(os.path.join(output_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
      pickle.dump(self.cat_count_dict, file_wrt)

  def load_dict(self, dict_path, prefix=""):
    with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.val_max_dict = pickle.load(file_wrt)
    with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.val_min_dict = pickle.load(file_wrt)
    with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
      self.cat_count_dict = pickle.load(file_wrt)
    print("val_max_dict.items()[:50]: {}".format(list(self.val_max_dict.items())))
    print("val_min_dict.items()[:50]: {}".format(list(self.val_min_dict.items())))

  def get_cat2id(self, threshold=100):
    for key, cat_count_d in self.cat_count_dict.items():
      new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
      for cat_str, count in new_cat_count_d.items():
        self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
    print("data vocab size: {}".format(len(self.cat2id_dict)))
    print("data vocab size[:50]: {}".format(list(self.cat2id_dict.items())[:50]))

  def map_cat2id(self, values, cats):
    def minmax_scale_value(i, val):
      min_v = float(self.val_min_dict["val_{}".format(i + 1)])
      max_v = float(self.val_max_dict["val_{}".format(i + 1)])
      if max_v == min_v:
        return 0.0
      else:
        return (float(val) - min_v) * 1.0 / (max_v - min_v)

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
        continue
      if count % 1000000 == 0:
        print("Have handle {}w lines.".format(count // 10000))
      label = items[0]
      features = items[1:]
      values = features[: data_stats.value_col_num]
      cats = features[data_stats.value_col_num:]
      assert len(values) == data_stats.value_col_num, "values.size： {}".format(len(values))
      assert len(cats) == data_stats.category_col_num, "cats.size： {}".format(len(cats))
      data_stats.stats_vals(values)
      data_stats.stats_cats(cats)
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
                          seed=2020, value_col_num=13, category_col_num=26):
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
      if len(items) != 1 + value_col_num + category_col_num:
        continue
      label = float(items[0])
      values = items[1:value_col_num+1]
      cats = items[value_col_num+1:]
      assert len(values) == value_col_num, "values.size： {}".format(len(values))
      assert len(cats) == category_col_num, "cats.size： {}".format(len(cats))
      ids, wts = data_stats.map_cat2id(values, cats)
      if i not in test_indices_set:
        train_feature_list.append(ids + wts)
        train_label_list.append(label)
      else:
        test_feature_list.append(ids + wts)
        test_label_list.append(label)
      if (len(train_label_list) > 0) and (len(train_label_list) % part_rows == 0):
        pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                            key="fixed")
        pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number), key="fixed")
        train_feature_list = []
        train_label_list = []
        train_part_number += 1
      if (len(test_label_list) > 0) and (len(test_label_list) % part_rows == 0):
        pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number), key="fixed")
        pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number), key="fixed")
        test_feature_list = []
        test_label_list = []
        test_part_number += 1

    if len(train_label_list) > 0:
      pd.DataFrame(np.asarray(train_feature_list)).to_hdf(train_feature_file_name.format(train_part_number),
                                                          key="fixed")
      pd.DataFrame(np.asarray(train_label_list)).to_hdf(train_label_file_name.format(train_part_number), key="fixed")
    if len(test_label_list) > 0:
      pd.DataFrame(np.asarray(test_feature_list)).to_hdf(test_feature_file_name.format(test_part_number), key="fixed")
      pd.DataFrame(np.asarray(test_label_list)).to_hdf(test_label_file_name.format(test_part_number), key="fixed")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get and Process datasets')

  parser.add_argument('--data_file_path', type=str, default=None, help='data file path')
  parser.add_argument('--value_col_num', type=int, default=None,
                      help='continue value column number of data_file_path file.')
  parser.add_argument('--category_col_num', type=int, default=None,
                      help='category column number of data_file_path file. ')
  parser.add_argument('--output_path', type=str, default=None, help='The path to save h5 dataset')
  parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio')
  parser.add_argument('--part_rows', type=int, default=2000000, help='saved samples of each file')
  parser.add_argument('--threshold', type=int, default=100, help='threshold to filter vocab')

  args, _ = parser.parse_known_args()

  # ============================= copy raw data to local ============
  tmp_data_dir = tempfile.mkdtemp('deepfm')
  mox.file.make_dirs(tmp_data_dir)
  tmp_data_path = os.path.join(os.path.join(tmp_data_dir, 'raw_data'),
                               os.path.split(args.data_file_path)[-1])
  mox.file.copy(args.data_file_path, tmp_data_path)
  args.data_file_path = tmp_data_path

  # ============================= copy raw data to local ============
  ori_output_path = args.output_path
  args.output_path = os.path.join(tmp_data_dir, 'h5_data')
  # ============================= end ===========================

  data_stats = DataStatsDict(value_col_num=args.value_col_num, category_col_num=args.category_col_num)
  stats_output_path = os.path.join(args.output_path, "stats_dict")
  mkdir_path(stats_output_path)
  # step 1, stats the vocab and normalize value
  statsdata(args.data_file_path, stats_output_path, data_stats)
  print("----------" * 10)
  data_stats.load_dict(dict_path=stats_output_path, prefix="")
  data_stats.get_cat2id(threshold=args.threshold)

  # step 2, transform data trans2h5; version 2: np.random.shuffle
  h5_save_path = os.path.join(args.output_path, 'h5_data')
  mkdir_path(h5_save_path)
  random_split_trans2h5(args.data_file_path, h5_save_path, data_stats,
                        part_rows=args.part_rows, test_size=args.test_ratio, seed=2020,
                        value_col_num=args.value_col_num, category_col_num=args.category_col_num)

  # =============================copy save data to obs==============
  mox.file.copy_parallel(args.output_path, ori_output_path)
