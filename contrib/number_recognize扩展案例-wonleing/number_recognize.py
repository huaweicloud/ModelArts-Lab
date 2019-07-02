#coding:utf-8
from PIL import Image,ImageOps
import urllib,urllib2,os,json
srcfile = "test.jpeg" #��֤��ͼƬ·��
outdir = "output" #���Ŀ¼����Ҫ���Ƚ���
numbers = 4 #��֤�����ֵ�����
split_l = 30 #��ñ߳ߴ�
split_r = 30 #�Ҳñ߳ߴ�
split_u = 50 #�ϲñ߳ߴ�
split_d = 50 #�²ñ߳ߴ�
username = "xxxxx" #��Ϊ�Ƶĵ�¼��(�ظ�)
password = "xxxxx" #��Ϊ�Ƶĵ�¼����(�ظ�)
endpoint = "cn-north-1" #�������ڵ�����Ĭ���Ǳ���һ
authurl = "https://modelarts.cn-north-1.myhuaweicloud.com/v3/auth/tokens" #��ȡtoken�������ڵ�����Ĭ���Ǳ���һ��
serviceurl = "https://xxxx.apigw.cn-north-1.huaweicloud.com/v1/infers/xxx" #���Լ������߷����ַ(�ظ�)

raw = Image.open(srcfile).convert('L') #�򿪲���ɫ��
img = ImageOps.invert(raw) #��תͼƬ��ɫ������ʵ�����ѡ�񡣽��Ҫ�ڵװ���
width,height = raw.size #��ȡԭͼ���
w = (width-split_l-split_r)/numbers #w�Ƿָ��ĵ�ͼ���
img = img.crop((split_l,split_u,width-split_r,height-split_d)) #�������µĲñ�
#img.save("output/img.jpg") #���ι�����֤��ͼƬ����Ԥ��
for i in range(numbers):
    split = img.crop((i*w, 0 ,i*w+w, height)).resize((28, 28)) #�س���ͼ��ѹ��Ϊ28,28�ߴ磬�ߴ���ѵ��ģ�����Ƶ�
    split.save(os.path.join(outdir, 'split_' + str(i) + '.jpg'), 'jpeg') #jpeg��ʽ���洢��ͼ

headers = {'Content-Type':'application/json'} #����Ϊ��׼urllib2 post json����
body={"auth":{"identity":{"methods":["password" ],"password":{"user":{"name":username,"password": password,
      "domain": {"name": username}}}},"scope": {"project": {"name": endpoint}}}}
req = urllib2.Request(url=authurl,headers=headers,data=json.dumps(body))
res = urllib2.urlopen(req)
token=res.headers.getheaders('X-Subject-Token')[0] #token������header��е�����̽��Ϸ����˼

result = "AI guess result is:" #��ʼ�����յ��������ַ���
for i in range(numbers):
    tmpres = os.popen('curl -F "images=@output/split_%d.jpg" -H "X-Auth-Token:%s" -X POST %s' 
    %(i,token,serviceurl)).readlines() #һ��һ����ͨ��curl post��������߷�����
    result += str(eval(tmpres[0])["predict label"]) #��ȡ���洢ʶ����

print "\n" + result + "\nIf result is not pricise, please help to train me." #���ս�����