# 使用Postman测试Modelarts在线服务

本文档介绍使用Postman测试Modelarts部署的在线服务，适用于图像分类、物体检测，其他类别的服务请参照修改。

## 1. 基础准备

Postman的下载、安装、注册、启动、获取用户Token等基础操作，请参照官网[如何通过Postman获取用户Token](https://support.huaweicloud.com/iam_faq/iam_01_034.html)

关于获取用户Token，强调一下Postman中的参数配置：

- post地址：https://iam.cn-north-1.myhuaweicloud.com/v3/auth/tokens

 - Body（raw格式）：
 ```
 { 
   "auth": { 
     "identity": { 
       "methods": [ 
         "password" 
       ], 
       "password": { 
         "user": { 
           "name": "username", 
           "password": "password", 
           "domain": { 
             "name": "username" 
           } 
         } 
       } 
     }, 
     "scope": { 
       "project": { 
         "name": "cn-north-1"
       } 
     } 
   } 
 }
 ```
 *username：华为云账号* ,
 *password：华为云密码*  ,
 *cn-north-1：所选区域*
 

 ![获取用户 toke](img/gettoken.png '获取用户 token')

 点击【Send】后，记录所获取的用户token。

 ## 2. 启动已经部署的在线服务

 依次点击【部署在线】-【在线服务】，启动所要测试的服务，记录【API接口地址】。

 ![启动服务](img/startservice.png '启动服务')

 ## 3. 测试在线服务

 在Postman中新建一个Request，参数配置如下：

 - POST地址：第2步记录的【API接口地址】
 - Headers新建Key：X-Auth-Token，Vlaue：第1步中记录的用户token
 - Body新建Key：images，Value：选择被测试的本地图片

 ![测试参数配置](img/modelarts-test1.png '参数配置 ')

 ![测试参数配置](img/modelarts-test3.png '参数配置 ')

 点击【Send】，返回测试结果。
