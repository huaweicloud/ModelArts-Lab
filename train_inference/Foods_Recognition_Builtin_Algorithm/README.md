# 使用ResNet50预置算法训练美食分类模型

本案例将介绍怎样使用AI Gallery中ModelArts官方发布的`ResNet50`算法和`美食分类数据集`，训练一个美食分类模型。

ModelArts的AI Gallery有丰富的算法，使用这些算法，无需自己开发训练代码和推理代码，只要准备并标注好数据，就可以轻松快速训练并部署模型。

## 准备工作

参考[此文档](https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/ModelArts准备工作/准备工作简易版.md)，完成ModelArts准备工作，包括注册华为云账号、ModelArts全局配置和OBS操作。

## 准备数据

### 下载数据

本案例的数据集已经发布在AI Gallery，我们从华为云AI Gallery订阅数据集至ModelArts，然后就可以在ModelArts中使用了。

点击[此链接](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/aiMarket/datasetDetail/metadata?content_id=c2b35c4a-20d2-4a3f-a4eb-60f4767b3ecd)进入数据集详情页，点击“下载”按钮，进入下载详情页，如下所示：

![创建数据集.PNG](./img/创建数据集.PNG)

其中，目标位置要选择上面 **“准备工作”** 步骤中您自己创建的OBS桶（本文使用的桶名为food--recognition，下文中涉及桶名的地方都需要替换为您自己创建的桶名），点击进入该桶，新建文件夹“food_recognition”，选择该文件夹为目标位置。


## 数据观察

点击[此链接](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dataset)，进入ModelArts数据集列表，在这里可看到刚才下载的数据集。  

![数据集信息.PNG](./img/数据集信息.PNG)

点击数据集名称，进入数据集概览页面，再点击右上角的“开始标注”，然后可以看到如下图所示的数据标注页面：

![数据集信息查看.png](./img/数据集信息查看.png)

可以看到，该数据集共包含4类美食，全部都已经进行了标注，类别名如下所示：

```
美食/凉皮,
美食/柿子饼,
美食/灌汤包,
美食/肉夹馍。
```

## 订阅算法

本实验中，我们从AI Gallery订阅ModelArts官方发布的图像分类算法`ResNet50`来训练模型。

点击进入AI Gallery[ResNet50算法](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/aiMarket/aiMarketModelDetail/overview?modelId=40b66195-5bbe-463d-b8a2-03e57073538d&type=algo)，点击页面右上方的![订阅.png](./img/订阅.png)按钮，然后再点击![继续订阅.png](./img/继续订阅.png)，点击![前往控制台.png](./img/前往控制台.png)，云服务区域选择“华为-北京四”，确定，进入算法管理页面。

![同步训练作业.png](./img/同步训练作业.png)

## 模型训练

我们使用创建的美食数据集和订阅的图像分类算法，提交一个图像分类的训练作业。

### 创建训练作业

接下来回到[ModelArts训练管理页面](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/trainingJobs)，在【训练管理】选择训练作业，点击【创建】，如下图所示：

![创建训练作业](./img/创建训练作业.png)

  在创建训练作业页面中选择算法：

![选择算法](./img/选择算法.PNG)

选择算法，（算法列表是按订阅时间显示的，找到名称为“图像分类-ResNet_v1_50”的算法，选中它）

![算法管理](./img/算法管理.PNG)

按照如下提示，填写创建训练作业的参数

计费模式：按需计费

名称：自定义

算法来源：算法管理

算法名称：`图像分类-ResNet_v1_50`

数据来源：数据集

选择数据集和版本：选择刚刚发布的美食数据集及其版本![作业参数.png](./img/作业参数.png)

然后再填写训练输出目录和调优参数，

![调优参数.png](./img/调优参数.png)

训练输出：选择 OBS路径`/food--recognition/food-recognition/output/`（注意，桶名food--recognition需更换为您自己的桶名，output目录可以通过新建文件夹创建）。训练输出位置用来保存训练生成的模型。

调优参数：用于设置算法中的超参。算法会加载默认参数，但是可以更改和添加参数，设置`learning_rate_strategy=20:0.001`，表示训练20轮，学习率固定为0.001，其他调优参数保持默认。

![作业日志路径.png](./img/作业日志路径.png)

作业日志路径：选择OBS路径`/food--recognition/food-recognition/log/`（此OBS路径如果不存在，可以通过新建文件夹创建）。

![训练规格.png](./img/训练规格.png)

资源池：公共资源池

规格：[限时免费]GPU: 1*NVIDIA-V100-pcie-32gb(32GB) | CPU: 8 核 64GB，如上图所示。（规格有免费规格和收费规格两种，选择免费规格可能会有资源排队情况，如遇排队，可选择继续等待，也可改成使用收费规格）

计算节点个数：选择1，表示我们运行一个单机训练任务

所有字段填写好之后，确认参数无误，点击“下一步”、“提交”按钮，后台则开始训练

点击“查看作业详情”会回到训练作业详情页面，此时训练作业的状态会经历“初始化”和“运行中”两个状态，等待4分钟左右，训练完成，状态变成”运行成功“。

### 查看训练结果

训练作业完成后，可以查看训练作业的运行结果。

在训练作业页面，点击作业名称，进入配置信息页面。可以查看到训练作业的详情。

![查看训练结果.png](./img/查看训练结果.png)

切换到“日志”页签，查看训练作业的训练日志，还可以下载日志到本地查看。

训练日志中会打印一些训练的精度和训练速度等信息。

训练生成的模型会放到训练输出位置OBS路径下，可以参考[此文档](https://support.huaweicloud.com/qs-obs/obs_qs_0009.html)，到OBS中将其下载到本地使用。

## 模型部署

### 导入模型

点击“创建模型”按钮，创建模型。

![创建模型.png](./img/创建模型.png)

按照如下提示，填写导入模型的字段。  
![导入模型信息.PNG](./img/导入模型信息.PNG)

名称：自定义

版本：0.0.1

![模型配置.png](./img/模型配置.png)

元模型来源：从训练中选择

选择训练作业及版本：刚刚的训练作业及版本（会自动加载）

部署类型：默认

推理代码：自动加载

其他保持默认。

点击“立即创建”按钮，开始构建模型，等待5分钟左右，模型的状态变成”正常“，则表示模型已导入成功。

### 部署上线

模型导入成功后，然后点击部署下拉框中的“在线服务”，如下图所示：

![部署在线服务.png](./img/部署在线服务.png)

按照如下指导填写参数：  ![部署参数.png](./img/部署参数.png)

计费模式：按需计费

名称：自定义

是否自动停止：开启，一小时后。会在1小时后自动停止该在线服务。

![资源池配置.png](./img/资源池配置.png)

资源池：公共资源池

模型来源：我的模型

模型：选择刚刚导入美食分类的模型和版本，会自动加载

计算节点规格：选择[限时免费]`CPU：1 核 4 GiB`

计算节点个数：1，如果想要更高的并发数，可以增加计算节点个数，会以多实例的方式部署。

填写好所有参数，点击“下一步”按钮，然后点击“提交”按钮，最后点击”查看服务详情“。状态栏会显示部署进度，等待3分钟左右，部署完成，服务的状态变成”运行中“，接下来就可以上传图片进行测试了。

### 在线服务测试

在线服务的本质是RESTful API，可以通过HTTP请求访问，在本案例中，我们直接在网页上访问在线服务。

先点[此链接]([https://modelarts-labs-bj4.obs.cn-north-4.myhuaweicloud.com:443/end2end/foods_recongition/foods_recongition_test.zip](https://modelarts-labs-bj4.obs.cn-north-4.myhuaweicloud.com/end2end/foods_recongition/foods_recongition_test.zip))下载测试集，解压，再切换到“预测”标签，点击上传按钮，进行测试。

预测结果会出现在右边的输出框：

![在线服务输出.png](./img/在线服务输出.png)

预测结果中的scores字段，表示图片为每种类别的置信度

## 关闭在线服务

为了避免持续扣费，案例完成后，需要关闭在线服务，点击“停止”按钮即可：  
![关闭在线服务.png](./img/关闭在线服务.png)

当需要使用该在线服务的时候，可以重新启动该在线服务。

### 确认关闭所有计费项

点击[此链接](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/manage/dashboard)，进入ModelArts总览页面，如果所有计费中的数字都是0，表示所有计费项都关闭了。

至此，本案例完成。