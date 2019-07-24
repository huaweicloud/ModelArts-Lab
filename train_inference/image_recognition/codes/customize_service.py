import cv2
from model_service.tfserving_model_service import TfServingBaseService
from PIL import Image
import numpy as np

class dogcat_service(TfServingBaseService):
  def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        image = image[np.newaxis, :, :, :]
        preprocessed_data[k] = image
    return preprocessed_data

  def _postprocess(self, data):
    outputs = {}
    logits = data['logits'][0][1]
    label = "dog" if logits > 0.5 else "cat"
    outputs['predict label'] = label
    confidence = logits if logits > 0.5 else 1 - logits
    outputs['message'] = "I am {:.2%} sure this is a {}".format(float(confidence), label)
    return outputs
