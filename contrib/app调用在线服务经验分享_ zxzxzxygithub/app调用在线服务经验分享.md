## 使用app测试Modelarts在线服务

### 1. **基础准备**

本demo代码已上传github，地址为
[https://github.com/zxzxzxygithub/hwmodelartdemo](https://github.com/zxzxzxygithub/hwmodelartdemo)，
clone下来之后导入android studio，并做以下改动

由于demo需要使用华为云用户名密码，clone 下来之后自行创建config.gradle 文件，此文件为igonore掉的，
不用担心会泄漏，只会存在本地 config.gradle里面内容格式如下,
```
ext.uname = ""
ext.dname = ""
ext.pwd = ""
```
在引号中填写用户名和密码，dname一般和uname保持一致,dname的获取可以在华为云官网进入我的凭证--IAM用户名即是

### 2. **启动已经部署的在线服务**

将MainActivity中的url替换为你的在线服务的url

![image](https://user-images.githubusercontent.com/7334714/61926089-c323a780-afa1-11e9-9e01-7710aa98521e.png)


### 3.  **测试在线服务**

运行代码，在模拟器或者手机上面点拍照或者从相册选择一张图片进行识别

界面截图为：

![模拟器](https://github.com/zxzxzxygithub/hwmodelartdemo/raw/master/testresult.png)
