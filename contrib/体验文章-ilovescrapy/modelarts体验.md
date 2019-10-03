# modelarts新版体验

华为云modelarts自发布以来不断更新，为开发者提供更多的便利，最近modelarts将要迎来新的大变化，modelarts2.0即将发布。对于新用户都可以在该页面上方免费领取25个小时的GPU资源包，我们可以用这个资源包来体验华为云modelarts的服务。

### modelarts总览

![img1](https://raw.githubusercontent.com/ilovescrapy/img-floder/master/img/1570073386(1).png)
在华为云modelarts总览上方有为新手入门提供的自动学习和AI全流程开发，自动学习分为图像分类、物体检测、预测分析和声音分类，AI全流程开发分数据管理、开发环境训练作业、模型管理和部署上线一体化服务。如果创建了相关服务在下方就可以看到，并可以发现是否有付费。

### notebook新功能

在左方开发环境一栏点击notebook，点击创建，如下图，新版的modelarts提供了定时关闭服务，这样这样就不用再担心因为自己忘了关闭而产生的不必要的费用。这个新功能的更新可谓是非常棒的。我平常一般用北京一区，在一区GPU只有1*P100可供选择，但在四区提供了1*V100，供选择的种类更多。存储配置上可以选择EVS和OBS，较常用的是EVS。EVS的费用也是非常低廉的，100G一个小时也仅仅只会产生0.13的费用，在5G以内是不会产生费用的。
![img2](https://raw.githubusercontent.com/ilovescrapy/img-floder/master/img/1570076255(1).jpg)

### 数据集标注

modelarts还提供了数据集的智能标注功能，大大降低了开发者们在数据标注这方面的时间成本。智能标注分为物体、语音和文本三大类，物体包括图像分类和物体检测，音频类分为声音分类、语音内容和语音分割。目前数据集的使用是免费的。
### 模型训练
数据集准备好后，接下来是作业训练，modelarts提供我们一些预制算法，对于作业训练我们可以使用相应预制算法，或者自己设计更好的、更准确的训练算法。模型训练结束后，在模型管理界面进行导入，在8月份modelarts更新中，增加了二次调优功能，目前仅对于用预制算法训练的模型可以二次调优，方便开发者进行更好的调优。
![img3](https://raw.githubusercontent.com/ilovescrapy/img-floder/master/img/1570082060.jpg)

### 模型部署
模型导入后，就可以将其部署为在线服务，等几分钟就成功部署上线了。

### AI市场
华为新版本的AI市场也提供了很多新的功能但有些还在研发中。
如下图，旧版的modelarts不支持通用类型的发布和计费，也不支持数据集的发布和跨region而新版本则将开放这些功能。扩大AI市场，吸引了更多的AI开发者们，让我们可以根据需要订购相应的产品，并发布数据集。
![img4](https://raw.githubusercontent.com/ilovescrapy/img-floder/master/img/1570083204(2).jpg)
