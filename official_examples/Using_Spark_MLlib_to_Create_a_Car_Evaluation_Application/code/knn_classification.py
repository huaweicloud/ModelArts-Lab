# -*- coding:utf-8 -*-

import collections
import json
import os
from collections import OrderedDict
from shutil import rmtree

from modelarts_pyspark import KNNClassification
from modelarts_pyspark.common import MeasurementType, Role, DataType, DfUtils, Meta, Attributes
from obs import ObsClient
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import expr
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType

# get the OBS configuration from system environment variables
AK = os.getenv('MINER_USER_ACCESS_KEY')
if AK is None:
    AK = ''

SK = os.getenv('MINER_USER_SECRET_ACCESS_KEY')
if SK is None:
    SK = ''

obs_endpoint = os.getenv('MINER_OBS_URL')
if obs_endpoint is None:
    obs_endpoint = ''
print("obs_endpoint: " + str(obs_endpoint))

obs_path = os.getenv('TRAIN_URL')
if obs_path is None:
    obs_path = ''
print("obs_path: " + str(obs_path))

data_path = os.getenv('DATA_URL')
if data_path is None:
    data_path = ''
print("data_path: " + str(data_path))

# define local temporary file names and paths
LOCAL_MODEL_PATH = '/tmp/spark_model'
MODEL_NAME = 'spark_model'

LOCAL_TRAIN_DATA_PATH = '/tmp/car.csv'
LOCAL_META_DATA_PATH = '/tmp/car_meta.desc'

LOCAL_CUSTOMIZE_SERVICE_PATH = '/tmp/customize_service.py'
CUSTOMIZE_SERVICE = 'customize_service.py'

LOCAL_METRIC_PATH = '/tmp/metric.json'
METRIC_NAME = 'metric.json'

LOCAL_CONFIG_PATH = '/tmp/config.json'
CONFIG_NAME = 'config.json'

# start local spark
spark = SparkSession.builder.config("spark.driver.host", "localhost").master("local[*]").appName(
    "classification").getOrCreate()
sc = spark.sparkContext
metric_dict = {}


def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)


# download file from OBS
def download_dataset(file_name, local_path):
    print("Start to download dataset from OBS")

    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)
    try:
        bucket_name = data_path.split("/", 1)[0]
        train_file = data_path.split("/", 1)[1] + "/" + file_name
        response = obs_client.getObject(bucket_name, train_file, downloadPath=local_path)
        if response.status < 300:
            print('succeeded to download file')
        else:
            print('failed to download file ')
            raise Exception('download file from OBS fail.')
    finally:
        obs_client.close()


# upload file to OBS
def upload_to_obs():
    # get the bucket name and OBS model path
    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)
    bucket_name = obs_path.split("/", 1)[0]
    work_metric = obs_path.split("/", 1)[1] + '/'
    work_model = obs_path.split("/", 1)[1] + '/model/'

    # build bucket path
    model_file = work_model + MODEL_NAME
    config_file = work_model + CONFIG_NAME
    metric_file = work_metric + METRIC_NAME
    customize_service_file = work_model + CUSTOMIZE_SERVICE
    obs_client.putContent(bucket_name, work_model, content=None)

    # upload config file to OBS
    print_title("upload config to obs !")
    obs_client.putFile(bucket_name, config_file, file_path=LOCAL_CONFIG_PATH)

    # upload custom service file to OBS
    print_title("upload customize_service to obs !")
    obs_client.putFile(bucket_name, customize_service_file, file_path=LOCAL_CUSTOMIZE_SERVICE_PATH)

    # upload metric file to OBS
    print_title("upload metric to obs !")
    obs_client.putFile(bucket_name, metric_file, file_path=LOCAL_METRIC_PATH)

    # upload model to OBS
    print_title("upload model to obs !")
    obs_client.putFile(bucket_name, model_file, file_path=LOCAL_MODEL_PATH)

    return 0


# calculate the metric value
def calculate_metric_value(knn_model, knn_df):
    result = knn_model.transform(knn_df)

    # map predicted result string types to numeric type
    indexer = StringIndexer()
    indexer.setInputCol('predict')
    indexer.setOutputCol('predictIndexer')
    index_model = indexer.fit(result)
    index_result = index_model.transform(result)

    # generate label column
    dataset = index_result.withColumn('label', expr("predictIndexer"))
    evaluator = MulticlassClassificationEvaluator(predictionCol="predictIndexer")

    # calculate metric
    metric_dict["f1"] = (evaluator.evaluate(dataset, {evaluator.metricName: "f1"}))
    metric_dict["accuracy"] = (evaluator.evaluate(dataset, {evaluator.metricName: "accuracy"}))
    metric_dict["precision"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedPrecision"}))
    metric_dict["recall"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedRecall"}))


# create config file
def create_config():
    """
    define the model configuration file,
    Can refer to:https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0061.html
    :return:
    """
    schema_model = json.loads(
        '{"model_algorithm":"","model_source":"custom","tunable":"false","model_type":"","metrics":{},"apis":[{"protocol":"http","url":"/","method":"post","request":{"Content-type":"application/json","data":{"type":"object","properties":{"data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"object","properties":{}}]}}}}}},"response":{"Content-type":"application/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"object","properties":{}}]}}}}}]}',
        object_pairs_hook=OrderedDict)

    # define profile properties
    res_properties = collections.OrderedDict()
    col_num = 7
    input_type_list = ["string", "string", "string", "string", "string", "string", "string"]
    output_type = {"type": "string"}
    model_algorithm = "knnclassification"
    model_type = "Spark_MLlib"

    # generate request data attribute
    for i in range(col_num):
        index = "input_%s" % (i + 1)
        res_properties[index] = {"type": input_type_list[i]}
    resp_properties = {"predictresult": output_type}

    # set property values
    schema_model['model_algorithm'] = model_algorithm
    schema_model['model_type'] = model_type
    schema_model["apis"][0]["request"]["data"]["properties"]["data"]["properties"]["req_data"]["items"][0][
        "properties"] = res_properties
    schema_model["apis"][0]["response"]["data"]["properties"]["resp_data"]["items"][0]["properties"] = resp_properties

    # set the metric attribute value
    schema_model["metrics"]["f1"] = metric_dict["f1"]
    schema_model["metrics"]["accuracy"] = metric_dict["accuracy"]
    schema_model["metrics"]["precision"] = metric_dict["precision"]
    schema_model["metrics"]["recall"] = metric_dict["recall"]

    with open(LOCAL_CONFIG_PATH, 'w') as f:
        json.dump(schema_model, f)

    print_title("create config file success!")


# create metric file, these data will be displayed in the evaluation results of the modelarts training assignments.
def create_metric():
    metric_model = json.loads(
        '{"total_metric":{"total_metric_meta":{},"total_reserved_data":{},"total_metric_values":{}}}',
        object_pairs_hook=OrderedDict)
    # The following four parameters require user customization
    metric_model["total_metric"]["total_metric_values"]["f1"] = metric_dict["f1"]
    metric_model["total_metric"]["total_metric_values"]["recall"] = metric_dict["recall"]
    metric_model["total_metric"]["total_metric_values"]["precision"] = metric_dict["precision"]
    metric_model["total_metric"]["total_metric_values"]["accuracy"] = metric_dict["accuracy"]

    with open(LOCAL_METRIC_PATH, 'w') as f:
        json.dump(metric_model, f)

    print_title("create metric file success!")


# custom metadata
def custom_meta():
    """
    this method only needs to provide training data files for training jobs through custom metadata.
    :return: data frame
    """
    df = spark.read.csv(LOCAL_TRAIN_DATA_PATH, header='true', inferSchema='true')

    attr_1 = Meta('buying_price', MeasurementType.NOMINAL, DataType.STRING)
    attr_1.set_values(['vhigh', 'high', 'med', 'low'])
    attr_2 = Meta('maint_price', MeasurementType.NOMINAL, DataType.STRING)
    attr_2.set_values(['vhigh', 'high', 'med', 'low'])
    attr_3 = Meta('doors', MeasurementType.NOMINAL, DataType.STRING)
    attr_3.set_values(['2', '3', '4', '5', 'more'])
    attr_4 = Meta('persons', MeasurementType.NOMINAL, DataType.STRING)
    attr_4.set_values(['2', '4', 'more'])
    attr_5 = Meta('lug_boot', MeasurementType.NOMINAL, DataType.STRING)
    attr_5.set_values(['small', 'med', 'big'])
    attr_6 = Meta('safety', MeasurementType.NOMINAL, DataType.STRING)
    attr_6.set_values(['low', 'med', 'high'])
    attr_7 = Meta('acceptability', MeasurementType.NOMINAL, DataType.STRING, Role.TARGET)
    attr_7.set_values(['acc', 'unacc', 'good', 'vgood'])
    schema = [dict(attr_1), dict(attr_2), dict(attr_3), dict(attr_4), dict(attr_5), dict(attr_6), dict(attr_7)]

    # build train data frame
    return DfUtils.build_data_frame(spark, df, schema)


# read csv with auto-detected schema.
def read_csv():
    """
    the method only needs to provide the training data file,
    and the metadata is automatically detected when the data set is read,
    and the metadata needs to be modified for different algorithms during training.
    :return: data frame
    """
    df = DfUtils.read_csv(sc, LOCAL_TRAIN_DATA_PATH, has_header=True)
    # modify meta
    new_list = DfUtils.modify_metadata(df.schema.fields, target='acceptability', name=Attributes.ROLE,
                                       value=Role.TARGET)
    fields = DfUtils.modify_metadata(new_list, target='acceptability', name=Attributes.MEASUREMENT,
                                     value=MeasurementType.NOMINAL)

    return spark.createDataFrame(df.rdd, StructType(fields))


# read csv with defined metadata file
def read_csv_and_meta():
    """
    this method needs to provide training data files and defined metadata files for training operations.
    :return: data frame
    """
    # read meta and train data file
    data = spark.read.csv(LOCAL_TRAIN_DATA_PATH, header='true', inferSchema='true')
    download_dataset('car_meta.desc', LOCAL_META_DATA_PATH)
    with open(LOCAL_META_DATA_PATH, 'r') as r:
        schema = json.loads(r.read())

    return DfUtils.build_data_frame(spark, data, schema)


def train():
    # download train dataset
    download_dataset('car.csv', LOCAL_TRAIN_DATA_PATH)

    # choose one of the following three methods to build training data frame.
    # 1.custom metadata
    # train_data_frame = custom_meta()

    # 2.read csv with auto-detected schema.
    train_data_frame = read_csv()

    # 3.read csv with defined metadata file
    # train_data_frame = read_csv_and_meta()

    # train
    knn_classification = KNNClassification().set_k(5)
    model = knn_classification.fit(train_data_frame)

    # save model
    model_path = LOCAL_MODEL_PATH
    rmtree(model_path, ignore_errors=True)
    model.save(sc, model_path)

    # calculate metric value
    calculate_metric_value(model, train_data_frame)


if __name__ == '__main__':
    # train
    train()
    # create config file
    create_config()
    # create metric file
    create_metric()
    # upload to OBS
    upload_to_obs()
