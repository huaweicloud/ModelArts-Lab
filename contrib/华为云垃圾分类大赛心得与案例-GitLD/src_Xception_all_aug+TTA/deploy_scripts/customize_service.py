import ast
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.saved_model import tag_constants
from model_service.tfserving_model_service import TfServingBaseService


class garbage_classify_service(TfServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 224  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.get_default_graph().as_default():
            self.sess = tf.Session(graph=tf.Graph(), config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_path)
            self.signature = meta_graph_def.signature_def

            # define input and out tensor of your model here
            input_images_tensor_name = self.signature[self.signature_key].inputs[self.input_key_1].name
            output_score_tensor_name = self.signature[self.signature_key].outputs[self.output_key_1].name
            self.input_images = self.sess.graph.get_tensor_by_name(input_images_tensor_name)
            self.output_score = self.sess.graph.get_tensor_by_name(output_score_tensor_name)


        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }

    def addGaussianNoise(self, image, percetage):
        G_Noiseimg = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        return G_Noiseimg

    # dimming
    def darker(self, image, percetage=0.9):
        image_copy = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        # get darker
        for xi in range(0, w):
            for xj in range(0, h):
                image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
                image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
                image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
        return image_copy

    def brighter(self, image, percetage=1.5):
        image_copy = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        # get brighter
        for xi in range(0, w):
            for xj in range(0, h):
                image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
                image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
                image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
        return image_copy
    def center_img(self, img, size=None, fill_value=255):
        """
        center img in a square background
        """
        img = self.addGaussianNoise(img, 0.3)
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = img.resize((int(self.input_size), int(self.input_size)))
        img = img.convert('RGB')
        img = np.array(img)
        img = img[:, :, ::-1]
        return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.preprocess_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img1 = self.center_img(img, self.input_size)
        img2 = self.darker(img)
        img3 = self.brighter(img)

        img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        img1 = img1[np.newaxis, :, :, :]
        img2 = img2[np.newaxis, :, :, :]
        img3 = img3[np.newaxis, :, :, :]

        pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
        pred_score1 = self.sess.run([self.output_score], feed_dict={self.input_images: img1})
        pred_score2 = self.sess.run([self.output_score], feed_dict={self.input_images: img2})
        pred_score3 = self.sess.run([self.output_score], feed_dict={self.input_images: img3})
        pred_score = pred_score+pred_score1+pred_score2+pred_score3
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0], axis=1)[0]
            result = {'result': self.label_id_name_dict[str(pred_label)]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
