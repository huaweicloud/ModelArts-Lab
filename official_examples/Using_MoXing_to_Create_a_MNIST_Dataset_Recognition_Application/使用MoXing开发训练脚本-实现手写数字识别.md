# 使用MoXing开发训练脚本，实现手写数字识别<a name="modelarts_10_0007"></a>

本示例介绍在ModelArts平台如何使用MoXing实现MNIST数据集的手写数字图像识别应用。

开始使用如下样例前，请务必按[准备工作](https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0003.html)指导完成必要操作。使用MoXing实现手写数字图像识别样例的操作流程如下所示。

1.  **[准备数据](#section6190340465)**：通过ModelArts市场预置数据集创建所需数据集版本。
2.  **[训练模型](#section1710418164461)**：使用MoXing框架编模型训练脚本，并新建训练作业进行模型训练。
3.  **[部署模型](#section9958141119468)**：得到训练好的模型文件后，新建预测作业将模型部署为在线预测服务。
4.  **[验证模型](#section760652810462)**：发起预测请求获取预测结果。

## 准备数据<a name="section6190340465"></a>

在ModelArts的“AI市场“中，默认提供了MNIST数据集，且名称为“Mnist-Data-Set“，您可以将数据集从“AI市场“导入到您的数据集中，以便用于模型训练和构建。

1.  登录ModelArts管理控制台，在左侧菜单栏中选择“AI市场“，进入AI市场主页。
2.  单击“数据集“页签进入数据集管理页面，找到MNIST数据集“Mnist-Data-Set“，单击数据集所在区域进入详情页面。

    **图 1**  找到MNIST数据集<a name="fig179701440141516"></a>  
    ![](figures/找到MNIST数据集.png "找到MNIST数据集")

3.  在详情页面中，单击“导入至我的数据集“。
4.  在“导入至我的数据集“对话框中，填写数据集“名称“及“存储路径“。名称可自行定义，存储路径选择已创建的OBS桶及文件夹，可参考图中示例，填写完成后单击“确定“。

    **图 2**  导入至我的数据集<a name="fig20316170162710"></a>  
    ![](figures/导入至我的数据集.png "导入至我的数据集")

5.  操作完成后，您可以前往“数据管理\>数据集“页面，查看数据导入情况。数据集的导入需要一定时间，大概几分钟，请耐心等待。

    在“数据集目录“中，当数据集版本状态为“正常“时，表示数据集已导入成功，您可以使用此数据集开始模型构建。数据集导入后，此示例数据将被拷贝至已设置好的OBS路径下。

    导入的MNIST数据集中，其中“.gz“文件为相同名称文件的压缩件，本次不会使用，本示例仅使用未压缩前的文件内容，包含的内容如下所示。

    -   “t10k-images-idx3-ubyte“：验证集，共包含10000个样本。
    -   “t10k-labels-idx1-ubyte“：验证集标签，共包含10000个样本的类别标签。
    -   “train-images-idx3-ubyte“：训练集，共包含60000个样本。
    -   “train-labels-idx1-ubyte“：训练集标签，共包含60000个样本的类别标签。

    **图 3**  数据集导入成功<a name="fig62924842420"></a>  
    ![](figures/数据集导入成功.png "数据集导入成功")


## 训练模型<a name="section1710418164461"></a>

数据准备完成后，您需要使用MoXing接口编写训练脚本代码，ModelArts提供了一个编写好的代码示例“train\_mnist.py“，如下操作使用此示例训练模型。

1.  从github下载[ModelArts-Lab](https://github.com/huaweicloud/ModelArts-Lab)工程，并在“ModelArts-Lab“工程的“\\ModelArts-Lab-master\\offical\_examples\\Using\_MoXing\_to\_Create\_a\_MNIST\_Dataset\_Recognition\_Application\\codes“目录下获取模型训练脚本文件“train\_mnist.py“。
2.  将“train\_mnist.py“文件上传至OBS，例如“test-modelarts/mnist-MoXing-code“。
3.  在ModelArts管理控制台，进入“训练作业“页面，单击左上角的“创建“。
4.  在创建训练作业页面，参考[图4](#fig1748310525123)和[图5](#fig348317528128)填写相关信息，然后单击“下一步“。

    **图 4**  创建训练作业-基本信息<a name="fig1748310525123"></a>  
    ![](figures/创建训练作业-基本信息.png "创建训练作业-基本信息")

    **图 5**  创建训练作业-详细参数<a name="fig348317528128"></a>  
    ![](figures/创建训练作业-详细参数.png "创建训练作业-详细参数")

5.  在“规格确认“页面，确认训练作业的参数信息，确认无误后单击“立即创建“。
6.  在训练作业管理页面，当训练作业变为“运行成功“时，即完成了模型训练过程。如有问题，可单击作业名称，进入作业详情界面查看训练作业日志信息。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >训练作业需要花费一些时间，预计十几分钟。当训练时间超过一定时间（如1个小时），请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。  

7.  （可选）在模型训练的过程中或者完成后，可以通过创建TensorBoard作业查看一些参数的统计信息。详细操作指导请参见[创建TensorBoard](https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0035.html)。

    其中，“日志路径“请选择训练作业中“训练输出位置“参数中的路径，如“/test-modelarts/mnist-model/“。根据界面提示完成TensorBoard创建。


## 部署模型<a name="section9958141119468"></a>

模型训练完成后，将模型部署为在线预测服务。其中ModelArt提供了已编写好的推理代码“customize\_service.py“和配置文件“config.json“。

1.  从github下载[ModelArts-Lab](https://github.com/huaweicloud/ModelArts-Lab)工程，并在“ModelArts-Lab“工程的“\\ModelArts-Lab-master\\offical\_examples\\Using\_MoXing\_to\_Create\_a\_MNIST\_Dataset\_Recognition\_Application\\codes“目录下获取推理代码“customize\_service.py“和配置文件“config.json“。
2.  将“customize\_service.py“和“config.json“文件上传至OBS中，需存储至OBS中训练作业生成模型的路径，例如“test-modelarts/mnist-model/model“。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   训练作业将在“训练输出位置“指定路径中新建一个“model“文件夹，用于存储生成的模型。  
    >-   推理代码和配置文件必须上传至“model“文件夹下。  

3.  在ModelArts管理控制台，进入“模型管理“页面，单击左上角“导入“。
4.  在“导入模型“页面，参考[图6](#fig1117910489486)填写相关参数，然后单击“立即创建“。

    在“元模型来源“中，选择“从OBS中选择“页签，然后在“选择元模型“选项中设置为训练作业中的“训练输出位置“指定的路径，不能设置为此路径下的“model“文件夹，否则系统无法自动找到模型及其相关文件。

    **图 6**  导入模型<a name="fig1117910489486"></a>  
    ![](figures/导入模型.png "导入模型")

5.  在“模型管理“页面，当模型状态变更为“正常“时，表示模型已导入成功。您可以在操作列单击“部署\>在线服务“，将模型部署为在线服务。
6.  在“部署“页面，请参考[图7](#fig20614113342113)的示例填写参数，然后单击“下一步“。

    **图 7**  部署在线服务<a name="fig20614113342113"></a>  
    ![](figures/部署在线服务.png "部署在线服务")

7.  在“规格确认“页面，确认信息无误后的，单击“立即创建“。
8.  在线服务创建完成后，系统自动跳转至“部署上线\>在线服务“页面。服务部署需要一定时间，耐心等待即可。当服务状态变为“运行中“时，表示服务部署成功。

## 验证模型<a name="section760652810462"></a>

在线服务部署成功后，您可以进入在线服务，发起预测请求测试服务。

1.  在“在线服务“管理页面，单击在线服务名称，进入在线服务详情页面。
2.  在线服务详情页面中，单击“预测“页签，进入预测页面。
3.  在“选择预测图片文件“右侧，单击“...“按钮，上传一张黑底白字的图片，然后单击“预测“。

    预测完成后，预测结果显示区域将展示预测结果，根据预测结果内容，可识别出此图片的数字是“2“的概率为“1“。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   由于推理代码和配置文件中已指定图片要求，用于预测的图片，大小必须为“28px\*28px“，图片格式必须为“jpg“，且图片必须是黑底白字。  
    >-   建议不要使用数据集中自带的图片，可以使用Windows自带的画图工具绘制一张。  
    >-   如果是其他不符合格式的单通道图片，预测结果可能会存在偏差。  

    **图 8**  预测结果<a name="fig2049295319516"></a>  
    ![](figures/预测结果.png "预测结果")


