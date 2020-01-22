import caffe
from model_service.caffe_model_service import CaffeBaseService
LABELS = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
          '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}

class ResnetService(CaffeBaseService):

    def __init__(self, model_name, model_path):
        super(ResnetService, self).__init__(model_name, model_path)
        # load input and configure preprocessing
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        self.transformer = transformer
        self.num_classes = len(LABELS)

    def _preprocess(self, data):
        for _, v in data.items():
            for _, file_content in v.items():
                im = caffe.io.load_image(file_content, color=False)
                self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
        return

    def _postprocess(self, data):
        data = self.net.blobs['prob'].data[0]
        end_idx = -6 if self.num_classes >= 5 else -self.num_classes - 1
        top_k = data.argsort()[-1:end_idx:-1]
        print(top_k)
        return {
            "predicted_label":
            LABELS[str(top_k[0])],
            "scores":
            [[LABELS[str(idx)], float(data[idx])] for idx in top_k]
        }