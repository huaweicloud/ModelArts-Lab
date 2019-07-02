import mxnet as mx
import requests
import zipfile
import json
import shutil
import os
import numpy as np

from mxnet.io import DataBatch
from mms.log import get_logger
from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import image, ndarray


logger = get_logger()


def check_input_shape(inputs, signature):
    '''Check input data shape consistency with signature.

    Parameters
    ----------
    inputs : List of NDArray
        Input data in NDArray format.
    signature : dict
        Dictionary containing model signature.
    '''
    assert isinstance(inputs, list), 'Input data must be a list.'
    assert len(inputs) == len(signature['inputs']), 'Input number mismatches with ' \
         'signature. %d expected but got %d.' \
                                           % (len(signature['inputs']), len(inputs))
    for input, sig_input in zip(inputs, signature['inputs']):
        assert isinstance(input, mx.nd.NDArray), 'Each input must be NDArray.'
        assert len(input.shape) == \
               len(sig_input['data_shape']), 'Shape dimension of input %s mismatches with ' \
                                'signature. %d expected but got %d.' \
                                % (sig_input['data_name'], len(sig_input['data_shape']),
                                   len(input.shape))
        for idx in range(len(input.shape)):
            if idx != 0 and sig_input['data_shape'][idx] != 0:
                assert sig_input['data_shape'][idx] == \
                       input.shape[idx], 'Input %s has different shape with ' \
                                         'signature. %s expected but got %s.' \
                                         % (sig_input['data_name'], sig_input['data_shape'],
                                            input.shape)

class DLSMXNetBaseService(MXNetBaseService):
    '''MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    '''
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        print ("-------------------- init classification servive -------------")
        self.model_name = model_name
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        self._signature = manifest['Model']['Signature']
        data_names = []
        data_shapes = []
        for input in self._signature['inputs']:
            data_names.append(input['data_name'])
            # Replace 0 entry in data shape with 1 for binding executor.
            # Set batch size as 1
            data_shape = input['data_shape']
            data_shape[0] = 1
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1
            data_shapes.append(('data', tuple(data_shape)))
        
        # Load MXNet module
        epoch = 0
        try:
            param_filename = manifest['Model']['Parameters']
            epoch = int(param_filename[len(model_name) + 1: -len('.params')])
        except Exception as e:
            logger.warning('Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, manifest['Model']['Symbol'][:-12]), epoch)
        self.mx_model = mx.mod.Module(symbol=sym, context=self.ctx,
                                      data_names=['data'], label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            if input_shape[1] == 1:
                img_arr = image.read(img, 0)
            else:
                img_arr = image.read(img)
            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        dim = len(data[0].shape)
        if dim > 2:
            data = mx.nd.array(np.squeeze(data.asnumpy(), axis=tuple(range(dim)[2:])))
        sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
        top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:5])
        return [{'probability': float(data[0, i].asscalar()), 'class': i}
                for i in top_prob]

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
        check_input_shape(data, self.signature)
        data = [item.as_in_context(self.ctx) for item in data]
        self.mx_model.forward(DataBatch(data))
        return self.mx_model.get_outputs()[0]

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
