# 利用华为云AI平台，训练搭建手写数字识别在线服务，识别本地验证码图片的完整应用案例

## 首先是一些General的提示
1. ModelArts官方文档的打开方式（jupyter notebook格式）
虽然号称可以直接在github中打开，当实测多次基本无法打开（也许是因为网速原因）。因此较靠谱是用以下两种方式：
* 自己安装jupyter notebook环境：由于python2.7的兼容bug https://www.jianshu.com/p/483dd8dcfcfb 以及各种依赖包的版本问题，所以最好不要用pip独立安装ipython或jupyter，推荐直接安装集成环境，如Conda https://www.jianshu.com/p/5eed417e04ca 或Anaconda http://baijiahao.baidu.com/s?id=1596866816908926760&wfr=spider&for=pc
* 使用在线阅读：只是为了读文档去安装环境总是让人有点不爽，于是福利来了，在线环境https://nbviewer.jupyter.org
帮你无痛解决，更幸福的是它扶持URL传参，于是可以直接访问以下链接来读到我们的文档：https://nbviewer.jupyter.org/github/huaweicloud/ModelArts-Lab/blob/master/Notebook%E6%A1%88%E4%BE%8B/01_face_recognition_DL/Face.ipynb

2. 实名认证及密钥
注册华为云后一定要先进行实名认证，然后再点我的凭证，生成密钥。否则生成的密钥无效，在后继的操作中会提示IAM错误。
![](http://wx3.sinaimg.cn/mw690/558fe6e3ly1g36ly6tq95j20hb07kt96.jpg)

3. 收费服务不使用时一定记得关闭，按小时收费，可以在总览中看到所有正在收费的项目

## 接下来我们就可以跟随数字识别项目的主文档进行实践了。
https://github.com/huaweicloud/ModelArts-Lab/tree/master/%E7%BB%BC%E5%90%88AI%E5%BC%80%E5%8F%91%E6%A1%88%E4%BE%8B/MNIST%20Dataset%20Recognition

文档中有些地方讲的不太细，在这里指出使用过程中一些坑：

1. 过程中需要下载三个文件，千万不要直接点击文档中的链接进行另存为，直接下载的是一个html页面。为方便下载我已经做好了这三个文件的web下载链接，点击http://123.206.26.34/html/ai/ 查看文件列表。

2. 创建数据集时首先要新建桶（可以下载OBS客户端，以后上传东西会方便一些，不想下载的可以直接使用网页版 https://storage.huaweicloud.com/obs/
），按文档指引导入Mnist-Data-Set数据集时要选中这个新建的桶，然后在桶中新建文件夹mnist（注意一定要使用这个文件夹名），选中该文件夹后才能确定。
![](http://wx4.sinaimg.cn/mw690/558fe6e3ly1g36lyaf22xj20ic087t94.jpg)

3. 按文档步骤训练完成后，在OBS文件系统的log文件夹中会自动生成一个model的目录（千万不要自己创建，如果没有生成一定是之前的步骤有误），customize_service.py和config.json一定要放进这个model文件夹。如下图。否则系统会提示创建了一个新的config，然后你就不能上传图进行预测了。
![](http://wx1.sinaimg.cn/mw690/558fe6e3ly1g36lye4pp5j217q0d9dh0.jpg)

4. 点创建模型时一定要注意右上角是否有提示信息（一闪而过），有信息就说明你上传的customize_service.py和config.json有错误，如果没有问题会加载出运行时的依赖列表。如下
![](http://wx1.sinaimg.cn/mw690/558fe6e3ly1g36lygo0ozj20yj0mg3zs.jpg)

## 完成在线服务布署这后，进行应用实例开发
利用训练完成后布署的华为云在线服务进行常见的验证码识别。该python脚本在本地运行，本地做的事情包括图片去边、颜色转化、切割成单个数字、大小压缩。然后利用curl+token的方式post到自己布署的华为云在线服务，parse远程的返回结果，连接字符串返回完整识别结果。详细代码及注释如下：
```python
#coding:utf-8
from PIL import Image,ImageOps
import urllib,urllib2,os,json
srcfile = "test.jpeg" #验证码图片路径
outdir = "output" #输出目录，需要事先建好
numbers = 4 #验证码数字的数量
split_l = 30 #左裁边尺寸
split_r = 30 #右裁边尺寸
split_u = 50 #上裁边尺寸
split_d = 50 #下裁边尺寸
username = "xxxxx" #华为云的登录名(必改)
password = "xxxxx" #华为云的登录密码(必改)
endpoint = "cn-north-1" #服务器节点名，默认是北京一
authurl = "https://modelarts.cn-north-1.myhuaweicloud.com/v3/auth/tokens" #获取token服务器节点名，默认是北京一的
serviceurl = "https://xxxx.apigw.cn-north-1.huaweicloud.com/v1/infers/xxx" #你自己的在线服务地址(必改)

raw = Image.open(srcfile).convert('L') #打开并单色化
img = ImageOps.invert(raw) #反转图片颜色，根据实际情况选择。结果要黑底白字
width,height = raw.size #获取原图宽高
w = (width-split_l-split_r)/numbers #w是分割后的单图宽度
img = img.crop((split_l,split_u,width-split_r,height-split_d)) #左上右下的裁边
#img.save("output/img.jpg") #修饰过的验证码图片整体预览
for i in range(numbers):
    split = img.crop((i*w, 0 ,i*w+w, height)).resize((28, 28)) #截出单图并压缩为28,28尺寸，尺寸是训练模型限制的
    split.save(os.path.join(outdir, 'split_' + str(i) + '.jpg'), 'jpeg') #jpeg格式化存储单图

headers = {'Content-Type':'application/json'} #以下为标准urllib2 post json流程
body = {"auth": {"identity": {"methods": ["password" ], "password": {"user": {"name": username,
       "password":password, "domain":{"name": username}}}},"scope":{"project": {"name": endpoint}}}}
req = urllib2.Request(url=authurl,headers=headers,data=json.dumps(body))
res = urllib2.urlopen(req)
token=res.headers.getheaders('X-Subject-Token')[0] #token竟藏在header里，有点玩侦探游戏的意思

result = "AI guess result is:" #初始化最终的输出结果字符串
for i in range(numbers):
    tmpres = os.popen('curl -F "images=@output/split_%d.jpg" -H "X-Auth-Token:%s" -X POST %s' 
    %(i,token,serviceurl)).readlines() #一个一个的通过curl post到你的在线服务上
    result += str(eval(tmpres[0])["predict label"]) #获取并存储识别结果

print "\n" + result + "\nIf result is not pricise, please help to train me." #最终结果输出
```

这是一个比较有实用价值的应用实例，能把常的网站验证码图片进行转换、切割、标准化，再post到你自己搭建的在线识别服务器一一识别，最后整合输出识别结果的一个完整过程。可以用作网站或APP上的数字验证码识别，从而达到自动化或批处理的目的。代码在ubuntu python2.7环境上运行结果如下：

![](https://github.com/wonleing/AI_number_recognize/blob/master/result.jpg?raw=true)

我测试使用的验证码文件可在文件夹中找到，根据不同网站的验证码图片，需要根据实际情况设置裁减尺寸、你的登录信息及服务地址等。详细的说明及代码解析都在写代码注释里。
另外，因为我是随便找了一张验证码图片，与训练过程使用的图差别非常大，所以识别结果并不精准。这可以通过你在训练服务中增加你实际中遇到的样例来提高精准度。这部分内容以后再做补充。
