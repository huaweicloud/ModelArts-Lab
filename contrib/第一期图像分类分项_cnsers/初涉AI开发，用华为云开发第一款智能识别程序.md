| Author | Contact          | Date      | Version | Action  |
|-------:|-----------------:|----------:|--------:|--------:|
| cnsers | 7668765@qq.com   | 2019.7.24 | 0.1     | Created |


AI(人工智能)可能是当下最热门的话题，每个人都或多或少的评论几句，每个人都想试试看到底能智能到什么程度。带着这个想法，今天我来给大家分享下超级适合新手朋友的AI识别程序的开发过程。

## 准备工作
首先，我们需要注册一个华为云的账号，注册过程很简单，就略过了。然后呢，我们通过领取“华为新用户大礼包”(根据阅读时间不同，可能该活动会发生变化)，可以获得“20-小时 CPU实例 + 10-小时 GPU实例 (规格P100和P4) + 10-小时 自动学习计算实例”。
这样，我们就可以免费体验AI开发了。

![这是优惠截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724091437.png)

但是大家要记得给账户里充值1元钱，避免我们操作错误，或者超出免费时长后，出现扣费，导致我们的环境突然停止的情况。
之后呢，我们通过点击【账户中心】，进入账户中心，然后选择【管理我的凭证】，进入【管理凭证页面】选择【访问凭证】，再选择【新增访问凭证】，通过手机验证后，会下载一个包含秘钥的CSV文件。

![这是账户中心截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724094432.png)
![这是凭证截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724094618.png)

我们通过点击【服务列表】，找到【EI企业智能】,点击其中的【ModelArts】，就进入了我们的AI开发环境了。

![这是进入ModelArts的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724094844.png)

在这里需要使用我们的访问秘钥了，先点击【全局配置】，再点击【添加访问秘钥】，将我们下载好的CSV文件中的秘钥按照要求填入对应的输入框内。

![这是添加访问秘钥的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724095115.png)

至此，我们的前期准备工作就完成了。虽然因为不熟悉会有一些坎坷，但是熟悉后就好了。我们后续的分享中，基本不会动这里了。


## 开始我们的第一个AI
我们在制作之前，需要先创建OBS桶，我们名字可以设定为AI-tongshi。这里的OBS指的是华为云对象存储服务，桶(bucket)指的就是存放的容器。

![这是访问OBS的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724095812.png)

创建完成后，我们新建一个文件夹，用来区分每一个AI程序，先创建【automl】文件夹，再创建【tongshi】文件夹。
现在我们点击【自动学习】，再点击【图像分类】项目，创建一个图像分类的项目，输入项目名称【zhaotongshi】，路径那里点击文件夹小图标，一级一级的向下选择，直到选择到我们的AI-tongshi文件夹。

![这是创建图像分类的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724100848.png)

点击确定后，我们的【图像分类】AI项目就创建完成了。接下来，我们要开始训练它了。

## 提交数据集
我们需要先准备好一个数据集，用来训练我们的AI。我用的是我自己同事的数据集，每个人10张照片，如果大家没有的话，官方提供了花卉的数据集，下载地址是 https://modelarts-labs.obs.cn-north-1.myhuaweicloud.com/ExeML/ExeML_Flowers_Recognition/flowers_recognition.tar.gz。
ps:如果大家下载官方数据集的话，记得需要先解压出来，数据集是解压后的文件夹flowers_recognition\train。

我们在当前【图像分类】项目中，选择【添加图片】，选中我们要添加进来的数据集图片，点击确定添加进来。

当图片全部添加进来之后，我们选中当前页面中，都是同一个人(或者同一种花)的图片，在右侧输入他们的标签，比如我选中的，标签填写为【李彦宏】，然后回车，就创建了一个新标签。

点击确定之后，这些被我选中的图片就跑到【已标注】列表中了。 这里记得把训练时间设置为0.1H，因为我们的图片少，就是测试用的。省的一会填完忘了，顺手就点了1H训练。

![这是标注的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724102334.png)

我们重复上面的操作，如果标签已存在，大家可以【标签名输入框】中直接点选就可以。
大家标注完毕后，可以通过右侧检查，数据是否一致。因为我每个人都是10张图片，所以右侧显示数据是对的。我就直接点击开始训练了

![这是标注检查的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724102956.png)
![这是提交训练的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724103143.png)

## 部署并测试
这个时候我们需要等待自动学习结束，才能进行下一步。
完成后我们选择【部署】，进行测试

![这是训练完成的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724120251.png)

部署完成后，我们点击【上传】，再点击【预测】就可以看到我们训练后的AI识别结果了

![这是识别完成的截图](https://github.com/qzltianxing/ModelArts-Lab/blob/master/doc_img/QQ%E6%88%AA%E5%9B%BE20190724121046.png)

## 总结
大家刚开始可能会有点慌乱的感觉，时间花费较多。熟悉后，反而就简单的理解了。
我们就是通过华为云提供的AI开发环境，通过自动学习的方式，让我们的AI能够识别出图片的内容类型。
后面的还可以扩展很多，让我们对未来，对AI，越来越有希望。
