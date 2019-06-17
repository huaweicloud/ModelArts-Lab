# coding:utf-8
import collections
import json
import traceback

import model_service.log as log
from model_service.spark_model_service import SparkServingBaseService
from pyspark.mllib.recommendation import MatrixFactorizationModel

logger = log.getLogger(__name__)


class user_Service(SparkServingBaseService):
    # request data preprocess
    def _preprocess(self, data):
        logger.info("Begin to handle data from user data...")
        json_data = json.loads(data, object_pairs_hook=collections.OrderedDict)
        list_data = []
        for element in json_data["data"]["req_data"]:
            array = []
            for each in element:
                array.append(element[each])
            list_data.append(array)
        return list_data

    # predict
    def _inference(self, data):
        try:
            # load model
            logger.info("Begin to load pyspark model...")
            model = MatrixFactorizationModel.load(self.spark.sparkContext, self.model_path)

            # predict
            logger.info("predict ALS model result...")
            pre_result = []
            for sin_data in data:
                sin_result = model.predict(sin_data[0], sin_data[1])
                pre_result.append(sin_result)

            return pre_result
        except Exception:
            logger.error('Predict failed!')
            logger.error(traceback.format_exc())
            raise Exception("prediction failed , check your model is right ?")

    # predict result process
    def _postprocess(self, data):
        logger.info("Customer service get new data to response")
        # build response data
        resp_data = []
        for element in data:
            resp_data.append({"predictresult": element})
        return resp_data
