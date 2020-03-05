
 # 物体检测——钢筋盘点 实践

  本次实践讲解一个真实应用 —— 钢筋盘点

## 背景

中国的各施工工地每年都要使用大量的钢筋，一车钢筋运到工地现场需要工作人员进行盘点，通常的做法是靠人工一根根数的方式，非常耗时费力。为了提高钢筋盘点效率，业界提出了对进场钢筋图片进行拍照，然后使用深度学习算法检测图片中的钢筋条数，实践证明，该方案不仅准确率高，而且可以极大提高效率。

  ## 数据集
 钢筋进场现场的图片，250张训练，200张测试

  ## 模型
  RFBNet模型，[官方源码](https://github.com/ruinmessi/RFBNet)

  ## 实验环境

  - **环境创建：**
    默认Notebook多引擎，Python3， GPU
  - **Notebook创建：**
    创建Notebook时请选择：**Pytorch-1.0.0**

  ## 实践案例

 - Notebook案例：[rebar_count.ipynb](https://nbviewer.jupyter.org/github/huaweicloud/ModelArts-Lab/blob/master/notebook/DL_rebar_count/rebar_count.ipynb)

