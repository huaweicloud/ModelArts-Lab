# 使用TensorFlow实现手写数字识别<a name="modelarts_10_0010"></a>

本文介绍在华为云ModelArts平台如何使用TensorFlow实现MNIST数据集的手写数字图像识别应用。

开始使用如下样例前，请务必按[准备工作](#section51133584536)指导完成必要操作。使用TensorFlow实现手写数字图像识别样例的操作流程如下所示。

1.  **[准备数据](#section16983151416266)**：通过ModelArts市场预置数据集创建所需数据集。
2.  **[训练模型](#section42320342266)**：使用TensorFlow框架编写模型训练脚本，并新建训练作业进行模型训练。
3.  **[部署模型](#section740614592308)**：训练结束后，将模型导入ModelArts，并将模型部署为在线服务。
4.  **[预测结果](#section9961181112311)**：发起预测请求获取预测结果。

## 准备工作<a name="section51133584536"></a>

-   已注册华为云账号，且在使用ModelArts前检查账号状态，账号不能处于欠费或冻结状态。
-   获取此账号的“AK/SK“，并在ModelArts全局配置中填写此信息，完成配置。
-   已在OBS服务中创建桶和文件夹，用于存放样例数据集以及模型。如下示例中，请创建命名为“test-modelarts“的桶，并创建如[表1](#zh-cn_topic_0168474775_table2061005120337)所示的文件夹。

    创建OBS桶和文件夹的操作指导请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/zh-cn_topic_0045829050.html)和[新建文件夹](https://support.huaweicloud.com/usermanual-obs/zh-cn_topic_0045829103.html)。由于ModelArts在“华北-北京一“区域下使用，为保证数据能正常访问，请务必在“华北-北京一“区域下创建OBS桶。

    **表 1**  文件夹列表

    <a name="zh-cn_topic_0168474775_table2061005120337"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0168474775_row1860945116333"><th class="cellrowborder" valign="top" width="31.180000000000003%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0168474775_p10609195123319"><a name="zh-cn_topic_0168474775_p10609195123319"></a><a name="zh-cn_topic_0168474775_p10609195123319"></a>文件夹名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="68.82000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0168474775_p206091751203312"><a name="zh-cn_topic_0168474775_p206091751203312"></a><a name="zh-cn_topic_0168474775_p206091751203312"></a>用途</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0168474775_row06099518334"><td class="cellrowborder" valign="top" width="31.180000000000003%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0168474775_p13609175110335"><a name="zh-cn_topic_0168474775_p13609175110335"></a><a name="zh-cn_topic_0168474775_p13609175110335"></a><span class="parmvalue" id="zh-cn_topic_0168474775_parmvalue86099516336"><a name="zh-cn_topic_0168474775_parmvalue86099516336"></a><a name="zh-cn_topic_0168474775_parmvalue86099516336"></a>“dataset-mnist”</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="68.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0168474775_p1609205193315"><a name="zh-cn_topic_0168474775_p1609205193315"></a><a name="zh-cn_topic_0168474775_p1609205193315"></a>用于存储数据集。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0168474775_row1660955118338"><td class="cellrowborder" valign="top" width="31.180000000000003%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0168474775_p3609175193317"><a name="zh-cn_topic_0168474775_p3609175193317"></a><a name="zh-cn_topic_0168474775_p3609175193317"></a><span class="filepath" id="zh-cn_topic_0168474775_filepath1660918516339"><a name="zh-cn_topic_0168474775_filepath1660918516339"></a><a name="zh-cn_topic_0168474775_filepath1660918516339"></a>“train-mnist”</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="68.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0168474775_p1360910519337"><a name="zh-cn_topic_0168474775_p1360910519337"></a><a name="zh-cn_topic_0168474775_p1360910519337"></a>用于存储训练脚本。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0168474775_row11609185115337"><td class="cellrowborder" valign="top" width="31.180000000000003%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0168474775_p460955193312"><a name="zh-cn_topic_0168474775_p460955193312"></a><a name="zh-cn_topic_0168474775_p460955193312"></a><span class="parmvalue" id="zh-cn_topic_0168474775_parmvalue166091551193319"><a name="zh-cn_topic_0168474775_parmvalue166091551193319"></a><a name="zh-cn_topic_0168474775_parmvalue166091551193319"></a>“mnist-model”</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="68.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0168474775_p260919518338"><a name="zh-cn_topic_0168474775_p260919518338"></a><a name="zh-cn_topic_0168474775_p260919518338"></a>用于存储训练输出的模型和预测文件。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   针对此示例，ModelArts提供了部署上线需使用的训练脚本、推理代码和配置文件。请获取示例文件并上传OBS指定路径，详细请参见[表2](#zh-cn_topic_0168474775_table8311142123817)。

    1.  在Github的[ModelArts-Lab](https://github.com/huaweicloud/ModelArts-Lab)工程中，单击“Clone or download“，然后在如下页面中单击“Download Zip“，下载工程。

        **图 1**  下载ModelArts-Lab<a name="zh-cn_topic_0168474775_fig1230292013811"></a>  
        ![](figures/下载ModelArts-Lab.png "下载ModelArts-Lab")

    2.  下载完成后，解压缩“ModelArts-Lab-master.zip“文件，然后在“\\ModelArts-Lab-master\\offical\_examples\\Using\_TensorFlow\_to\_Create\_a\_MNIST\_Dataset\_Recognition\_Application\\codes“目录中获取到训练脚本“train\_mnist\_tf.py“、推理代码“customize\_service.py“和配置文件“config.json“，并上传至对应OBS路径。上传文件至OBS的操作指导请参见[上传文件](https://support.huaweicloud.com/usermanual-obs/zh-cn_topic_0045829661.html)。

    **表 2**  上传示例文件至OBS

    <a name="zh-cn_topic_0168474775_table8311142123817"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0168474775_row173128427385"><th class="cellrowborder" valign="top" width="28.342834283428342%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0168474775_p2312184210385"><a name="zh-cn_topic_0168474775_p2312184210385"></a><a name="zh-cn_topic_0168474775_p2312184210385"></a>文件名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="20.282028202820282%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0168474775_p931214426388"><a name="zh-cn_topic_0168474775_p931214426388"></a><a name="zh-cn_topic_0168474775_p931214426388"></a>描述</p>
    </th>
    <th class="cellrowborder" valign="top" width="51.375137513751376%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0168474775_p173121842163815"><a name="zh-cn_topic_0168474775_p173121842163815"></a><a name="zh-cn_topic_0168474775_p173121842163815"></a>上传至如下OBS路径中</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0168474775_row193121442183816"><td class="cellrowborder" valign="top" width="28.342834283428342%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0168474775_p11341443143915"><a name="zh-cn_topic_0168474775_p11341443143915"></a><a name="zh-cn_topic_0168474775_p11341443143915"></a><span class="filepath" id="filepath1276211200216"><a name="filepath1276211200216"></a><a name="filepath1276211200216"></a>“train_mnist_tf.py”</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="20.282028202820282%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0168474775_p11312942173810"><a name="zh-cn_topic_0168474775_p11312942173810"></a><a name="zh-cn_topic_0168474775_p11312942173810"></a>训练脚本。</p>
    </td>
    <td class="cellrowborder" valign="top" width="51.375137513751376%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0168474775_p231218421387"><a name="zh-cn_topic_0168474775_p231218421387"></a><a name="zh-cn_topic_0168474775_p231218421387"></a><span class="parmvalue" id="zh-cn_topic_0168474775_parmvalue1527941313402"><a name="zh-cn_topic_0168474775_parmvalue1527941313402"></a><a name="zh-cn_topic_0168474775_parmvalue1527941313402"></a>“test-modelarts/train-mnist”</span></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0168474775_row23121342143817"><td class="cellrowborder" valign="top" width="28.342834283428342%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0168474775_p93121842193815"><a name="zh-cn_topic_0168474775_p93121842193815"></a><a name="zh-cn_topic_0168474775_p93121842193815"></a><span class="filepath" id="zh-cn_topic_0168474775_filepath104971918391"><a name="zh-cn_topic_0168474775_filepath104971918391"></a><a name="zh-cn_topic_0168474775_filepath104971918391"></a>“customize_service.py”</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="20.282028202820282%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0168474775_p531254212388"><a name="zh-cn_topic_0168474775_p531254212388"></a><a name="zh-cn_topic_0168474775_p531254212388"></a>推理代码。</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="51.375137513751376%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0168474775_p10352936124012"><a name="zh-cn_topic_0168474775_p10352936124012"></a><a name="zh-cn_topic_0168474775_p10352936124012"></a><span class="parmvalue" id="zh-cn_topic_0168474775_parmvalue9352143619408"><a name="zh-cn_topic_0168474775_parmvalue9352143619408"></a><a name="zh-cn_topic_0168474775_parmvalue9352143619408"></a>“test-modelarts/mnist-model/model”</span></p>
    <div class="note" id="zh-cn_topic_0168474775_note193410367475"><a name="zh-cn_topic_0168474775_note193410367475"></a><a name="zh-cn_topic_0168474775_note193410367475"></a><span class="notetitle"> 说明： </span><div class="notebody"><a name="ul175761726142314"></a><a name="ul175761726142314"></a><ul id="ul175761726142314"><li>请在训练作业完成后，再执行上传操作。</li><li>如果训练作业运行多次，将生成不同的版本，<span class="filepath" id="filepath1577410271288"><a name="filepath1577410271288"></a><a name="filepath1577410271288"></a>“mnist-model”</span>目录下将生成多种版本的目录，如<span class="filepath" id="filepath199562366284"><a name="filepath199562366284"></a><a name="filepath199562366284"></a>“V0001”</span>、<span class="filepath" id="filepath376883812283"><a name="filepath376883812283"></a><a name="filepath376883812283"></a>“V0002”</span>，请基于训练作业的版本，将文件上传至对应版本下的<span class="filepath" id="filepath49081841172812"><a name="filepath49081841172812"></a><a name="filepath49081841172812"></a>“model”</span>文件夹内。</li></ul>
    </div></div>
    <p id="zh-cn_topic_0168474775_p12312642183811"><a name="zh-cn_topic_0168474775_p12312642183811"></a><a name="zh-cn_topic_0168474775_p12312642183811"></a></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0168474775_row13312114215389"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0168474775_p43122427380"><a name="zh-cn_topic_0168474775_p43122427380"></a><a name="zh-cn_topic_0168474775_p43122427380"></a><span class="filepath" id="zh-cn_topic_0168474775_filepath1541113212394"><a name="zh-cn_topic_0168474775_filepath1541113212394"></a><a name="zh-cn_topic_0168474775_filepath1541113212394"></a>“config.json”</span></p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0168474775_p133124429380"><a name="zh-cn_topic_0168474775_p133124429380"></a><a name="zh-cn_topic_0168474775_p133124429380"></a>配置文件。</p>
    </td>
    </tr>
    </tbody>
    </table>


## 准备数据<a name="section16983151416266"></a>

在ModelArts的“AI市场“中，默认提供了MNIST数据集，且名称为“Mnist-Data-Set“，您可以将数据集从“AI市场“导入到您的数据集中，以便用于模型训练和构建。

1.  登录ModelArts管理控制台，在左侧菜单栏中选择“AI市场“，进入AI市场主页。
2.  单击“数据集“页签进入数据集管理页面，找到MNIST数据集“Mnist-Data-Set“，单击数据集所在区域进入详情页面。

    **图 2**  找到MNIST数据集<a name="zh-cn_topic_0169506907_fig179701440141516"></a>  
    ![](figures/找到MNIST数据集.png "找到MNIST数据集")

3.  在详情页面中，单击“导入至我的数据集“。
4.  在“导入至我的数据集“对话框中，填写数据集“名称“及“存储路径“。名称可自行定义，存储路径选择已创建的OBS桶及文件夹，可参考图中示例，填写完成后单击“确定“。

    **图 3**  导入至我的数据集<a name="zh-cn_topic_0169506907_fig20316170162710"></a>  
    ![](figures/导入至我的数据集.png "导入至我的数据集")

5.  操作完成后，您可以前往“数据管理\>数据集“页面，查看数据导入情况。数据集的导入需要一定时间，大概几分钟，请耐心等待。

    在“数据集目录“中，当数据集版本状态为“正常“时，表示数据集已导入成功，您可以使用此数据集开始模型构建。数据集导入后，此示例数据将被拷贝至已设置好的OBS路径下。

    导入的MNIST数据集中，其中“.gz“文件为相同名称文件的压缩件，本次不会使用，本示例仅使用未压缩前的文件内容，包含的内容如下所示。

    -   “t10k-images-idx3-ubyte“：验证集，共包含10000个样本。
    -   “t10k-labels-idx1-ubyte“：验证集标签，共包含10000个样本的类别标签。
    -   “train-images-idx3-ubyte“：训练集，共包含60000个样本。
    -   “train-labels-idx1-ubyte“：训练集标签，共包含60000个样本的类别标签。

    **图 4**  数据集导入成功<a name="zh-cn_topic_0169506907_fig62924842420"></a>  
    ![](figures/数据集导入成功.png "数据集导入成功")


## 训练模型<a name="section42320342266"></a>

数据准备完成后，您需要使用TensorFlow编写训练脚本代码，ModelArts提供了一个编写好的代码示例“train\_mnist\_tf.py“，如下操作使用此示例训练模型。

1.  参考[准备工作](#section51133584536)的操作指导，将“train\_mnist\_tf.py“文件上传至OBS，例如“test-modelarts/train-mnist“。
2.  在ModelArts管理控制台，进入“训练作业“页面，单击左上角的“创建“。
3.  在“创建训练作业“页面，参考[图5](#fig15868125515112)和[图6](#fig12869655141111)填写相关信息，然后单击“下一步“。

    **图 5**  创建训练作业-基本信息<a name="fig15868125515112"></a>  
    ![](figures/创建训练作业-基本信息.png "创建训练作业-基本信息")

    **图 6**  创建训练作业-详细参数<a name="fig12869655141111"></a>  
    ![](figures/创建训练作业-详细参数.png "创建训练作业-详细参数")

4.  在“规格确认“页面，确认训练作业的参数信息，确认无误后单击“立即创建“。
5.  在训练作业管理页面，当训练作业变为“运行成功“时，即完成了模型训练过程。如有问题，可单击作业名称，进入作业详情界面查看训练作业日志信息。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >训练作业需要花费一些时间，预计十几分钟。当训练时间超过一定时间（如1个小时），请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。  

6.  （可选）在模型训练的过程中或者完成后，可以通过创建TensorBoard作业查看一些参数的统计信息。详细操作指导请参见[创建TensorBoard](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0050.html)。

    其中，“日志路径“请选择步骤训练作业中“训练输出位置“参数中的路径，如“/test-modelarts/mnist-model/“。根据界面提示完成TensorBoard创建。


## 部署模型<a name="section740614592308"></a>

模型训练完成后，可以创建预测作业，将模型部署为在线预测服务。在部署模型之前，获取推理代码“customize\_service.py“和配置文件“config.json“并上传至OBS中。此推理代码和配置文件为ModelArts为操作样例提供的示例。您也可以基于[模型包规范](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)，开发自己所需的推理代码及配置文件。

1.  在部署模型之前，请参见[准备工作](#section51133584536)，将推理代码和配置文件上传至对应OBS路径下。本示例中模型输出位置的OBS路径为“test-modelarts/mnist-model/model“。
2.  在ModelArts管理控制台，进入“模型管理“页面，单击“导入“。
3.  在“导入模型“页面，参考[图7](#fig1391183516364)填写相关参数，然后单击“立即创建“。

    在“元模型来源“中，选择“从OBS中选择“页签。然后在“选择元模型“右侧文本框选择训练作业中“训练输出位置“指定的路径。

    **图 7**  导入模型<a name="fig1391183516364"></a>  
    ![](figures/导入模型.png "导入模型")

    在“模型管理“页面，当模型状态变为“正常“时，表示模型已导入成功。

4.  单击操作列“部署\>在线服务“，将模型部署为在线服务。

    在“部署“页面，参考[图8](#fig3534111012408)填写参数，然后根据界面提示完成在线服务创建。

    **图 8**  部署模型<a name="fig3534111012408"></a>  
    ![](figures/部署模型.png "部署模型")


## 预测结果<a name="section9961181112311"></a>

完成模型部署后，等待服务部署完成，当服务状态显示为“运行中“，表示服务已部署成功。

1.  在“在线服务“页面，单击在线服务名称，进入服务详情页面。
2.  单击“预测“页签，在“选择预测图片文件“右侧，单击“...“按钮选择一张黑底白字的图片，然后单击“预测“。

    预测完成后，预测结果显示区域将展示预测结果，根据预测结果内容，可识别出此图片的数字是“1“。

    >![](public_sys-resources/icon-note.gif) **说明：**   
    >-   由于推理代码和配置文件中已指定图片要求，用于预测的图片，大小必须为“28px\*28px“，且图片必须是黑底白字。  
    >-   建议不要使用数据集中自带的图片，可以使用Windows自带的画图工具绘制一张。  

    **图 9**  预测结果<a name="zh-cn_topic_0168474775_fig2049295319516"></a>  
    ![](figures/预测结果.png "预测结果")

3.  如果不在需要使用此模型及在线服务，建议清除相关资源，避免产生不必要的费用。
    -   在“在线服务“页面，“停止“或“删除“刚创建的在线服务。
    -   在“模型管理“页面，“删除“刚导入的模型。
    -   在“训练作业“页面，“删除“运行结束的训练作业。
    -   在“数据集“管理页面，删除数据集并同步删除对应OBS路径下的数据。


