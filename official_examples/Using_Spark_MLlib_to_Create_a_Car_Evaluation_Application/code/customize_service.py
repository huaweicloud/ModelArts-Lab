# -*- coding:utf-8 -*-
import copy
import json
from modelarts_pyspark import KNNClassificationModel
from modelarts_pyspark.common import DfUtils
from model_service.spark_model_service import SparkServingBaseService
import model_service.log as log
import traceback

logger = log.getLogger(__name__)


class CustomerService(SparkServingBaseService):

    # request data preprocess, transform the request data into predict data frame
    def _preprocess(self, data):
        logger.info("Customer service begin to handle request data.")
        # transform request data to data frame
        predict_json = json.loads(data)
        req_data = predict_json["data"]["req_data"]
        for data in req_data:
            if not isinstance(data, dict):
                raise Exception(
                    'Request data format is incorrect, parsing request data failed, please check data format')

        json_data = [json.dumps(req_data)]
        json_rdd = self.spark.sparkContext.parallelize(json_data)
        data_frame = self.spark.read.json(json_rdd)

        # read metadata file
        meta_path = self.model_path + '/data/metadata/metadataDesc/part-00000'
        with open(meta_path, 'r') as meta_file:
            schema = json.loads(meta_file.read())

        # build predict data frame
        predict_data_frame = DfUtils.build_data_frame(self.spark, data_frame, schema, check=True)

        return predict_data_frame

    # load model and transform input data frame
    def _inference(self, data):
        logger.info("Customer service start to predict.")
        try:
            # load model
            load_model = KNNClassificationModel.load(self.spark.sparkContext, self.model_path)
        except Exception:
            logger.error(traceback.format_exc())
            raise Exception('Load KNNClassification model failed.', traceback)

        # use the model to make predictions
        predict_result = load_model.transform(data)

        return predict_result

    # predict result postprocess, transform prediction data frame to dict
    def _postprocess(self, pre_data):
        logger.info("Customer service generate response data")
        # build response data
        result = [row.predict for row in pre_data.collect()]
        array_data = []
        dict_data = {}
        for index, element in enumerate(result):
            dict_data['predictresult'] = element
            array_data.append(copy.copy(dict_data))

        return array_data
