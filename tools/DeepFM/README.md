# DeepFM Data Preprocess

数据预处理脚本请参考： [data_process.py](code/data_process.py)

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

criteo feature数据
<center>	

![an image](./pics/demo_input_features.png) 
</center>
 criteo label数据
<center>

![an image](./pics/demo_output_label.png)  
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
    |- train_input_part_0.h5
    |- train_output_part_0.h5
    |- train_input_part_1.h5
    |- train_output_part_1.h5
    |- train_input_part_2.h5
    |- train_output_part_2.h5
    |- test_input_part_0.h5
    |- test_output_part_0.h5
    |- test_input_part_1.h5
    |- test_output_part_1.h5
    |...
  ```



# 4 数据集和预处理理解

criteo数据集格式（第一列表示label，前n列记录了一些连续特征值，后m列记录了离散特征值，此时n=3,m=2）
```shell
0  240  1  0.01  red    large
0  240  2  0.21  red    smalll
1  120  2  0.21  blue   large
1  30   3  0.31  black  small
```

转换成易于理解的数据格式其实就是针对一些商品的描述，比如商品的颜色（离散特征，枚举值），商品的价格（连续特征）
label则表示预测结果，比如说用户是否购买
```shell
sample     intensity  life    price  color  spec   label
1          240        1       0.01   red    large  0
2          240        2       0.21   red    small  1
3          120        2       0.21   blue   large  0
4          30         3       0.31   black  small  1
```

那么要将这个数据集经过预处理才能训练，预处理首先要针对每个特征建立一个id隐射表
每个连续特征单独占一个id，每个离散特征每种取值各占一个id，OOV表示不在范围内的离散特征
```shell
intensity  shopid  life   color_OOV  color_red  color_blue  color_black  spec_OOV  spec_large  spec_small
0          1       2      3          4          5           6            7         8           9
```

针对连续性特征记录最大值和最小值
```shell
     intensity   life    price
max  240         3       0.31
min  30          1       0.01
```

将每一行样本转换成一下的id列表和value列表，ids表示该样本涉及的特征id
例如sample=2这个样本
涉及连续特征值三个intensity、shopid、life，所以ids中要加入[0,1,2]
那么values则要加入这些连续特征的值，并且要在最大值(max)和最小值(min)上做归一化，也就是(x - min) / (max - min)
sample=2这个样本的intensity是240，在intensity的最大最小值上归一化的结果是 (240 - 30) / (240 - 30) = 1.0
sample=2这个样本的life是2，在life的最大最小值上归一化的结果是 (2 - 1) / (3 - 1) = 0.5
sample=2这个样本的price是0.21，在intensity的最大最小值上归一化的结果是 (0.21 - 0.01) / (0.31 - 0.01) = 0.66667
所以values中要加入[1.0, 0.5, 0.66667]
然后涉及离散特征值，color是red，spec是large，所以取ids中加入[4,9]，离散特征的values为[1,1]

转换完毕得到以下特征，就可以进入DeepFM这个网络进行训练了
```shell
sample   ids                values                          label
1        [0, 1, 2, 4, 8]    [1.0,     0.0, 0.0,     1, 1]   0
2        [0, 1, 2, 4, 9]    [1.0,     0.5, 0.66667, 1, 1]   0
3        [0, 1, 2, 5, 8]    [0.42857, 0.5, 0.66667, 1, 1]   1
4        [0, 1, 2, 6, 9]    [0.0   ,  1.0, 1.0,     1, 1]   1
```

如果有多值特征，例如新增一个pages字段，可以取最大可能转换成定长特征，例如以下特征
```shell
sample  intensity  life    price  color   spec    pages     stat    label
1       240        1       0.01   red     large   A,B       X,Y     0
2       240        2       0.21   red     small   C,D,E,F   X,Y,Z   1
3       120        2       0.21   blue    large   A,G,H     X,Y     0
4       30         3       0.31   black   small   A,B,C     X,Y,Z   1
```

pages是一个多值特征，将其转换成定长
```shell
sample  intensity  life    price  color   spec    pages_1  pages_2  pages_3  pages_4  stat_1  stat_2  stat_3  label
1       240        1       0.01   red     large   A        B        OOV      OOV      X       Y       OOV     0
2       240        2       0.21   red     small   C        D        E        F        X       Y       Z       1
3       120        2       0.21   blue    large   A        G        H        OOV      X       Y       OOV     0
4       30         3       0.31   black   small   A        B        C        OOV      X       Y       Z       1
```


得到的映射表为：
```shell
intensity life price color_OOV spec_OOV pages_OOV stat_OOV color_red color_blue color_black spec_large spec_small pages_A pages_B pages_C pages_D pages_E pages_F pages_G pages_H stat_X stat_Y stat_Z
0         1    2     3         4        5         6        7         8          9           10         11         12      13      14      15      16      17      18      19      20     21     22
```

然后再将每个样本转换成ids + values的形式（为了便于示意，连续值未做归一化）
```shell
sample    ids                                values                           label
1         0,1,2,7,10,12,13,5,5,20,21,6       240,1,0.01,1,1,1,1,1,1,1,1,1     0
2         0,1,2,7,11,14,15,16,17,20,21,22    240,2,0.21,1,1,1,1,1,1,1,1,1     1
3         0,1,2,8,10,12,18,19,5,20,21,6      120,2,0.21,1,1,1,1,1,1,1,1,1     0
4         0,1,2,9,11,12,13,14,5,20,21,22     30,3,0.31,1,1,1,1,1,1,1,1,1      1
```
