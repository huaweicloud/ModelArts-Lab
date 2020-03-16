# 冰山识别（使用MoXing框架实现图像分类）<a name="modelarts_10_0004"></a>

本文介绍如何在ModelArts上使用MoXing实现Kaggle竞赛中的冰山图像分类任务。实验所使用的图像为雷达图像，需要参赛者利用算法识别出图像中是冰山（iceberg）还是船（ship）。

开始使用如下样例前，请务必按[准备工作](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0001.html)指导完成必要操作。冰山识别样例的操作流程如下所示。

1.  **[准备数据](#section4865410216)**：获取本示例使用的数据集并上传至OBS，编写代码将数据集格式转换成TFRecord。
2.  **[训练模型](#section19745720175916)**：使用MoXing API编写实现冰山图像分类的网络模型，新建训练作业进行模型训练。
3.  **[预测结果](#section148971738105912)**：再次新建训练作业，对示例数据集进行预测，并将结果保存到“csv“文件。
4.  **[查看结果](#section16235530195913)**：查看“csv“文件中的预测结果。

## 准备数据<a name="section4865410216"></a>

ModelArts在公共OBS桶中提供了MNIST数据集，命名为“Iceberg-Data-Set“，因此，本文的操作示例使用此数据集进行模型构建。您需要执行如下操作，将数据集上传至您的OBS目录下，即准备工作中您创建的OBS目录“test-modelarts/iceberg/iceberg-data“。然后通过Notebook将数据集格式转换成TFRecord格式。

1.  单击[数据集下载链接](https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/dataset-market/Iceberg-Data-Set/archiver/Iceberg-Data-Set.zip)，将“Iceberg-Data-Set“数据集下载至本地。
2.  在本地，将“Iceberg-Data-Set.zip“压缩包解压。例如，解压至本地“Iceberg-Data-Set“文件夹下。
3.  参考[上传文件](https://support.huaweicloud.com/usermanual-obs/obs_03_0307.html)，使用批量上传方式将“Iceberg-Data-Set“文件夹下的所有文件上传至“test-modelarts/iceberg/iceberg-data“OBS路径下。

    其中，训练集“train.json“包含4类数据：“band\_1“、“band\_2“、“inc\_angle“和“is\_iceberg“（测试集）。

    -   “band\_1“和“band\_2“：雷达图像的2个通道，分别是75x75的矩阵。
    -   “inc\_angle“：雷达图拍摄角度，单位是角度。
    -   “is\_iceberg“： 标注。冰山为1，船为0。

4.  进入“开发环境\>Notebook“页面，单击“创建“，在弹出框中填写Notebook名称。单击“下一步“，进入规格确认页面，单击“提交“完成创建操作。

    针对当前样例，推荐使用“公共资源池“的“GPU“。如果选择使用“CPU“，Notebook的运行时间可能较长，且运行过程中容易出现故障。

    **图 1**  创建Notebook<a name="fig121541356135310"></a>  
    ![](figures/创建Notebook.png "创建Notebook")

5.  Notebook创建完成后，在操作列，单击“打开“，进入“Jupyter Notebook“文件目录界面。
6.  单击右上角的“New\>TensorFlow-1.8“，进入代码开发界面。

    **图 2**  创建Notebook开发页面<a name="fig1117464215569"></a>  
    ![](figures/创建Notebook开发页面.png "创建Notebook开发页面")

7.  在Cell中填写数据转换代码。完整代码请参见[data\_format\_conversion.py](https://gitee.com/ModelArts/ModelArts-Lab/blob/master/official_examples/Using_MoXing_to_Create_a_Iceberg_Images_Classification_Application/codes/data_format_conversion.py#)。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >脚本代码中的“BASE\_PATH“参数，请根据数据集实际存储位置修改。本示例中使用的OBS路径为“test-modelarts/iceberg/iceberg-data/“, 即“train.json“和“test.json“的OBS父目录。  

8.  单击Cell上方的“Run“运行代码。运行代码过程可能需要较长时间，如果长时间没有执行结果，请尝试分段执行代码。将脚本示例代码分成多段放在不同的cell中执行。

    代码运行成功后，将在“test-modelarts/iceberg/iceberg-data/“目录下生成如下3个文件：

    -   “iceberg-train-1176.tfrecord“：训练数据集。
    -   “iceberg-eval-295.tfrecord“：验证数据集。
    -   “iceberg-test-8424.tfrecord“：预测数据集。

9.  完成数据准备后，为避免产生不必要的费用，建议进入Notebook管理页面，在操作列中单击“停止“或“删除“，停止或删除此Notebook。

## 训练模型<a name="section19745720175916"></a>

数据准备完成后，您需要使用MoXing接口编写训练脚本代码，ModelArts提供了一个编写好的代码示例“train\_iceberg.py“，您也可以在ModelArts的“开发环境\>Notebook“中编写模型训练脚本，并转成“py“文件。

如下操作使用“train\_iceberg.py“示例训练模型。

1.  从gitee下载[ModelArts-Lab](https://gitee.com/ModelArts/ModelArts-Lab)工程，并在“ModelArts-Lab“工程的“\\ModelArts-Lab-master\\official\_examples\\Using\_MoXing\_to\_Create\_a\_Iceberg\_Images\_Classification\_Application\\codes“目录下获取模型训练脚本文件“train\_iceberg.py“。
2.  将“train\_iceberg.py“文件上传至OBS，假设上传至“test-modelarts/iceberg/iceberg-code/“文件夹下。
3.  在ModelArts管理控制台，进入“训练管理 \> 训练作业“页面，单击左上角的“创建“。
4.  <a name="li1013661073819"></a>填写训练作业相关参数。

    -   “名称“和“描述“：请按照界面提示规则填写。
    -   “算法来源“：选择“常用框架“页签，“AI引擎“选择“TensorFlow“和“TF-1.8.0-python2.7“；“代码目录“选择模型训练脚本文件“train\_iceberg.py“所在的OBS父目录（“test-modelarts/iceberg/iceberg-code/“），“启动文件“请选择“train\_iceberg.py“。
    -   “数据来源“：选择“数据存储位置“，然后选择数据集存储的OBS路径。
    -   “训练输出位置“：选择一个OBS路径用于存放生成模型及预测文件。
    -   “资源池“：选择一个可用的资源池用于训练。GPU资源池性能优于CPU资源池，但是相应的费用也更高。
    -   “计算节点个数“：此示例建议使用1个节点即可。

    **图 3**  创建训练作业<a name="fig0661122024313"></a>  
    ![](figures/创建训练作业.png "创建训练作业")

5.  参数确认无误后，单击“提交“，完成训练作业创建。
6.  在训练作业管理页面，当训练作业变为“运行成功“时，即完成了模型训练过程。如有问题，可单击作业名称，进入作业详情界面查看训练作业日志信息。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >训练作业需要花费一些时间，预计几十分钟。当训练时间超过一定时间（如1个小时），请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。  

7.  （可选）在模型训练的过程中或者完成后，可以通过创建可视化作业查看一些参数的统计信息，如“loss“、“accuracy“等。您也可以选择不创建可视化作业，直接进入下一步：[预测结果](#section148971738105912)。

    进入“训练管理 \> 训练作业 \> 可视化作业“界面，单击“创建“，填写可视化作业名称，“训练输出位置“请选择步骤[4](#li1013661073819)中“训练输出位置“参数中的路径。根据界面提示完成可视化作业创建。当状态变为“运行中“时，表示创建完成。您可以单击可视化作业的名称跳转到其可视化界面，查看此训练作业的相关信息。


## 预测结果<a name="section148971738105912"></a>

待训练作业运行完成后，在“训练输出位置“目录下生成模型文件。由于我们只需要进行一次预测，因此不需要部署在线服务。相关的预测操作已经在“train\_iceberg.py“文件写好，预测结果将输出到“submission.csv“文件。

使用训练作业进行预测的操作步骤如下：

1.  进入ModelArts“训练管理 \> 训练作业“页面，单击左上角的“创建“。
2.  填写相关参数，然后根据界面提示完成训练作业创建。

    “名称“：请根据界面提示要求填写。

    “算法来源“和“数据来源“：填写信息与[训练模型](#section19745720175916)时相同。详情请参见步骤[4](#li1013661073819)。

    “运行参数“：预测时，务必添加参数“is\_training=False“，表示不进行重新训练。

    “训练输出位置“：与[训练模型](#section19745720175916)的步骤[4](#li1013661073819)保持一致。

    “计算节点个数“：预测时“计算节点个数“只能选择1个节点。

    **图 4**  创建一次预测<a name="fig1133251617529"></a>  
    ![](figures/创建一次预测.png "创建一次预测")

3.  进入“训练作业“管理页面，当训练作业状态变为“运行成功“时，表示预测完成。单击训练作业的名称，进入作业详情页面。

    在“日志“页签中，可以查看到在eval数据集上的loss值。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >训练作业需要花费一些时间，预计十几分钟。当训练时间超过一定时间（如1个小时），请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。  


## 查看结果<a name="section16235530195913"></a>

在“训练输出位置“目录下，可通过“submission.csv“文件查看预测结果。

**图 5**  冰山识别预测结果<a name="fig413613171391"></a>  
![](figures/冰山识别预测结果.png "冰山识别预测结果")

