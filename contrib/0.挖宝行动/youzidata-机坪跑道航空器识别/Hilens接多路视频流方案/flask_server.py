# -*- coding: utf-8 -*-
import base64
import warnings

import cv2

warnings.filterwarnings("ignore")

from flask import Flask, request
import numpy as np
import json
import uuid
import time

app = Flask(__name__)

skip_count = {}
skip_frame = 4


@app.route('/', methods=['POST'])
def get_json():
    #    time.sleep(0.01)
    global skip_count
    global skip_frame
    if request.method == 'POST':
        request_data = request.get_data().decode('utf-8')

        info = json.loads(request_data)

        # 识别结果传输出去
        pic_id = info['ori_img_id']  # 图片id
        m_time = info['time']  # 图片对应时间
        cam = info['cam_name']  # 摄像头名称
        # 跳帧
        if cam not in skip_count.keys():
            skip_count[cam] = 0
        else:
            skip_count[cam] += 1
        if skip_count[cam] % skip_frame != 1:
            return ""
        if skip_count[cam] >= 10000:
            skip_count[cam] = 1
        res = []
        if info['boxes']:
            res = [
                [bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h'], bbox['label'], bbox['confidence']]
                for bbox in info['boxes']]
        entity_id = [str(uuid.uuid1()) for i in range(len(res))]
        entities = []
        for i in range(len(res)):
            rl = res[i]
            IEntity = {"entity_id": entity_id[i], "boundary_type": 0,
                       "boundary": [(rl[0], rl[1]), (rl[2], rl[3])],
                       "label": rl[4], "sub_label": '', "score": rl[5]}
            entities.append(IEntity)
        # 原始图像转为byte传输
        img_bin = info['ori_image']  # 原始图BASE64编码结果
        img_base64 = base64.b64decode(img_bin)  # base64图片解码
        img_array = np.fromstring(img_base64, np.uint8)  # 转换np序列
        img_yuv = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)  # 发送的是原始YUV图片，这里注意
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV21)  # 转为BGR格式后，方可正常使用
        cv2.imwrite(uuid.uuid4(), img)  #
        IPic = {"camera_id": cam, "time": m_time, "img": str_encode,
                "entities": json.dumps(entities, ensure_ascii=False, cls=MyEncoder).encode(encoding='utf8')}

        tDict = {'res': res, 'pic_id': pic_id, 'entity_id': entity_id, 'camera_id': cam, 'time': m_time}

        return ""


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
