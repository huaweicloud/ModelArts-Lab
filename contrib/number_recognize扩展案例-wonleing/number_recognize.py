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
body={"auth":{"identity":{"methods":["password" ],"password":{"user":{"name":username,"password": password,
      "domain": {"name": username}}}},"scope": {"project": {"name": endpoint}}}}
req = urllib2.Request(url=authurl,headers=headers,data=json.dumps(body))
res = urllib2.urlopen(req)
token=res.headers.getheaders('X-Subject-Token')[0] #token竟藏在header里，有点玩侦探游戏的意思

result = "AI guess result is:" #初始化最终的输出结果字符串
for i in range(numbers):
    tmpres = os.popen('curl -F "images=@output/split_%d.jpg" -H "X-Auth-Token:%s" -X POST %s' 
    %(i,token,serviceurl)).readlines() #一个一个的通过curl post到你的在线服务上
    result += str(eval(tmpres[0])["predict label"]) #获取并存储识别结果

print "\n" + result + "\nIf result is not pricise, please help to train me." #最终结果输出