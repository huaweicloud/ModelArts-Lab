# DeepFM Data Preprocess

数据预处理脚本请参考： [data_process.py](DeepFM/code/data_process.py)

以[Criteo数据集](https://www.kaggle.com/c/criteo-display-ad-challenge/data "Criteo数据集")为例。

## 1. 输入数据


Criteo原始训练数据集为train.txt, 测试集为test.txt。 样例如下。
<center>

![an image](./pics/demo_origin_data.png) 
criteo原始数据
</center>

criteo 原始数据中，第一列为label，表示是否点击，后续39列为特征列，其中前13列为连续值特征，后26列为离散特征（为经过hash处理后的字符串）。

## 2. 预处理

### 2.1 预处理流程

- 统计各连续值列的min-max value; 统计各类别列的词典频次字典；

- 按threshold=100,对频次词典进行过滤，得到类别映射为id的map字典；

- 最后得到feature特征分为两种，id 和 weights 。
   - id特征存储的值都是map id，包括：
     - 连续特征列的map id（本例连续特征共13列，map id分别对应0-12）
     - 离散特征列的map id值（26列）
   - weights特征包括：
     - 连续值weights（MinMaxScaler处理后的值）
     - 离散值weights（有值的部分为1，Nan(空值)的为0）

处理后的特征数据保存为H5文件，特征数据共78列(0-38为id特征列，39-77为weights特征列)，内容格式如下：

<center>

![an image](./pics/demo_input_features.png) 
criteo feature数据
</center>
<center>

![an image](./pics/demo_output_label.png)  
 criteo label数据
</center>

### 2.2 预处理参数
 
| 名称                 |  默认值 |  类型  |        描述          |
| --------------------| --------| ----- | -------------------- |
|data_file_path       |None     |string |      数据输入路径     |
|output_path          |None     |string |  预处理后数据存储路径  |
|value_col_num        |None     |int    |      连续特征列数     |
|category_col_num     |None     |int    |      离散特征列数     |
|test_ratio           |0.1      |float  |      验证集切分占比   |
|threshold            |100      |int    | 词典中词频低于100的会被过滤掉， 会影响data vocab_size 的大小|
|part_rows            |2000000  |int  | 每个文件保存的样本数，样本量大时，可分成多个输出文件 |

### 2.3 命令参考

	python process_data.py \
	    --data_file_path=/home/xxx/deepfm/data/raw/data.txt \  
	    --output_path=/home/xxx/deepfm/data/h5_data/ \
        --value_col_num=13 \
        --category_col_num=26 \
        --test_ratio=0.2 \
        --threshold=50 \
        --part_rows=10000

注意： 预处理完成后，会在控制台打印 data vocab size, 将作为训练的输入参数。

## 3 输出数据

处理之后的数据分为四部分：

-  train\_input\_part\_xx.h5（训练集，存储feature）
-  train\_output\_part\_xx.h5（训练集，存储label）
-  test\_input\_part\_xx.h5（验证集，存储feature）
-  test\_output\_part\_xx.h5（验证集，存储label）

文件输出格式如下（以3个训练集，2个验证集为例），dataset_path将作为训练的数据输入。
  ```shell
  dataset_path
    |- train\_input\_part\_0.h5
    |- train\_output\_part\_0.h5
    |- train\_input\_part\_1.h5
    |- train\_output\_part\_1.h5
    |- train\_input\_part\_2.h5
    |- train\_output\_part\_2.h5
    |- test\_input\_part\_0.h5
    |- test\_output\_part\_0.h5
    |- test\_input\_part\_1.h5
    |- test\_output\_part\_1.h5
    |...
  ```
