from PIL import Image
import numpy as np
from model_service.tfserving_model_service import TfServingBaseService


class mnist_service(TfServingBaseService):
  def _preprocess(self, data):
    preprocessed_data = {}

    for k, v in data.items():
      for file_name, file_content in v.items():
        image1 = Image.open(file_content)
        image1 = np.array(image1, dtype=np.float32)
        image1.resize((1, 784))
        preprocessed_data[k] = image1

    return preprocessed_data

  def _postprocess(self, data):

    outputs = {}
    logits = data['logits'][0]
    label = logits.index(max(logits))
    logits = ['%.3f' % logit for logit in logits]
    outputs['predicted_label'] = str(label)
    label_list = [str(label) for label in list(range(10))]
    scores = dict(zip(label_list, logits))
    scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    outputs['scores'] = scores
  
    return outputs
