#! /usr/bin/python3.7
import cv2
import hilens
from utils import *


def run():
    # 系统初始化，参数要与创建技能时填写的检验值保持一致
    hilens.init("mask")
    
    # 初始化摄像头
    camera  = hilens.VideoCapture()
    display = hilens.Display(hilens.HDMI)
    
    # 初始化模型
    mask_model_path = hilens.get_model_dir() + "convert-mask-detection.om"
    mask_model      = hilens.Model(mask_model_path)
    
    while True:
        ##### 1. 设备接入 #####
        input_yuv = camera.read() # 读取一帧图片(YUV NV21格式)
        
        ##### 2. 数据预处理 #####
        img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_YUV2RGB_NV21) # 转为RGB格式
        img_preprocess, img_w, img_h = preprocess(img_rgb) # 缩放为模型输入尺寸
    
        ##### 3. 模型推理 #####
        output = mask_model.infer([img_preprocess.flatten()])
        
        ##### 4. 结果输出 #####
        bboxes  = get_result(output, img_w, img_h) # 获取检测结果
        img_rgb = draw_boxes(img_rgb, bboxes)      # 在图像上画框
        output_yuv = hilens.cvt_color(img_rgb, hilens.RGB2YUV_NV21)
        display.show(output_yuv) # 显示到屏幕上
        
    hilens.terminate()


if __name__ == "__main__":
    run()
    