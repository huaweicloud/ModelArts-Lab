import logging
import os

import numpy as np

import mxnet as mx
from mms.model_service.mxnet_model_service import MXNetBaseService
from mxnet.io import DataBatch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SHAPE = 416
THRESH = 0.5
CLASS_NAMES = []


class DLSFasterRCNNService(MXNetBaseService):
    '''MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    '''

    def __init__(self, model_name, model_dir, manifest, gpu=None):
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
        self.model_name = model_name
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        self._signature = manifest['Model']['Signature']
        self.data_shape = (1, 3, INPUT_SHAPE, INPUT_SHAPE)
        self._signature['inputs'] = [{
            'data_name': 'images',
            'data_shape': self.data_shape
        }]
        self.scale = []
        self.num_images = 0

        self.provide_data = [
            mx.io.DataDesc(name='data', shape=self.data_shape),
        ]

        # Load MXNet module
        epoch = 0
        try:
            param_filename = manifest['Model']['Parameters']
            epoch = int(param_filename[len(model_name) + 1:-len('.params')])
        except Exception as e:
            logging.info(
                'Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            '%s/%s' % (model_dir, manifest['Model']['Symbol'][:-12]), epoch)

        ## process the arg_params
        self.symbol = sym
        self.mx_model = mx.mod.Module(
            context=self.ctx,
            symbol=sym,
            data_names=['data'],
            label_names=None)
        self.mx_model.bind(
            for_training=False,
            data_shapes=self.provide_data,
            label_shapes=None)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

    def _preprocess(self, data):
        img_list = []
        scale_list = []
        for img in data:
            img_arr = mx.img.imdecode(img, flag=1, to_rgb=True)
            oh, ow, _ = img_arr.shape
            img_arr = mx.img.imresize(
                img_arr,
                INPUT_SHAPE,
                INPUT_SHAPE,
                interp=mx.img.image._get_interp_method(
                    9, (oh, ow, INPUT_SHAPE, INPUT_SHAPE)))
            img_arr = mx.nd.image.to_tensor(img_arr)
            img_arr = mx.nd.image.normalize(img_arr, mean=MEAN, std=STD)
            img_list.append(img_arr)
            scale_list.append((float(oh) / INPUT_SHAPE,
                               float(ow) / INPUT_SHAPE))
        self.scale = scale_list
        self.num_images = len(img_list)
        return img_list

    def _postprocess(self, data):
        for ith in range(self.num_images):
            ids, scores, bboxes = [xx[ith].asnumpy() for xx in data]
            response = {
                'detection_classes': [],
                'detection_boxes': [],
                'detection_scores': []
            }
            for idx in range(ids.shape[0]):
                if ids[idx][0] < 0 or scores[idx][0] < THRESH:
                    continue
                bboxes[idx][0] *= self.scale[ith][1]
                bboxes[idx][1] *= self.scale[ith][0]
                bboxes[idx][2] *= self.scale[ith][1]
                bboxes[idx][3] *= self.scale[ith][0]
                response['detection_classes'].append(
                    str(CLASS_NAMES[int(ids[idx][0])]))
                response['detection_boxes'].append([
                    str(bboxes[idx][0]),
                    str(bboxes[idx][1]),
                    str(bboxes[idx][2]),
                    str(bboxes[idx][3])
                ])
                response['detection_scores'].append(str(scores[idx][0]))
        return response

    def _inference(self, data):
        '''Internal inference methods for MXNet. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        '''
        # Check input shape
        logging.info('%s', str(mx.nd.stack(*data).shape))
        batch_data = mx.io.DataBatch(
            data=[
                mx.nd.stack(*data)
            ],
            provide_data=self.provide_data)

        self.mx_model.forward(batch_data)
        return self.mx_model.get_outputs()

    def ping(self):
        '''Ping to get system's health.

        Returns
        -------
        String
            MXNet version to show system is healthy.
        '''
        return mx.__version__

    @property
    def signature(self):
        '''Signiture for model service.

        Returns
        -------
        Dict
            Model service signiture.
        '''
        return self._signature
