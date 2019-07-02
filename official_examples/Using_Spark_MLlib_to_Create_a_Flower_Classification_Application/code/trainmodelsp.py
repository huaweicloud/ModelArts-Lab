# coding:utf-8
import collections
import json
import os
from collections import OrderedDict

import pandas as pd
import pyspark
from obs import ObsClient
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import *
from pyspark.ml.feature import *

# define local temporary file names and paths
TRAIN_DATASET = 'iris.csv'
MODEL_NAME = 'spark_model'
CONFIG_NAME = 'config.json'
METRIC_NAME = 'metric.json'
LOCAL_MODEL_PATH = '/tmp/'
LOCAL_CONFIG_PATH = '/tmp/config.json'
LOCAL_METRIC_PATH = '/tmp/metric.json'
LOCAL_DATA = '/tmp/iris.csv'

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

# start local Spark
spark = pyspark.sql.SparkSession.builder.config("spark.driver.host", "localhost").master("local[*]").appName(
    "flower_classification").getOrCreate()
metric_dict = {}


# download file from OBS
def download_dataset():
    print("Start to download dataset from OBS")

    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    try:
        bucket_name = data_path.split("/", 1)[0]
        train_file = data_path.split("/", 1)[1] + "/iris.csv"
        resp = obs_client.getObject(bucket_name, train_file, downloadPath=LOCAL_DATA)
        if resp.status < 300:
            print('Succeeded to download training dataset')
        else:
            print('Failed to download ')
            raise Exception('Failed to download training dataset from OBS !')

    finally:
        obs_client.close()


# upload file to OBS
def upload_to_obs():
    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    bucket_name = obs_path.split("/", 1)[0]
    work_metric = obs_path.split("/", 1)[1] + '/'
    model_dir = obs_path.split("/", 1)[1] + '/model/'
    model_file = model_dir + MODEL_NAME
    config_file = model_dir + CONFIG_NAME
    metric_file = work_metric + METRIC_NAME

    # upload model to OBS
    print_title("upload model to obs !")
    obs_client.putFile(bucket_name, model_file, file_path=LOCAL_MODEL_PATH + MODEL_NAME)

    # upload config file to OBS
    print_title("upload config to obs !")
    obs_client.putFile(bucket_name, config_file, file_path=LOCAL_CONFIG_PATH)

    # upload metric file to OBS
    print_title("upload metric to obs !")
    obs_client.putFile(bucket_name, metric_file, file_path=LOCAL_METRIC_PATH)

    return 0


def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)


# calculate the metric value
def calculate_metric_value(regression_model, regression_df):
    """
    because modelarts console UI only support displaying value: f1, recall, precision, and accuracy,
    this step maps `mae` to `recall`, `mse` maps to `precision`, and `rmse` maps to `accuracy`.
    :return:
    """
    # split a portion of the data for testing
    dataset = regression_model.transform(regression_df)

    # calculate metric
    evaluator = RegressionEvaluator(predictionCol="prediction")
    # modelarts only supports multi-classification metric currently, borrow recall/precision/accuracy to show regression metric
    metric_dict["f1"] = 0
    metric_dict["recall"] = (evaluator.evaluate(dataset, {evaluator.metricName: "mae"}))
    metric_dict["precision"] = (evaluator.evaluate(dataset, {evaluator.metricName: "mse"}))
    metric_dict["accuracy"] = (evaluator.evaluate(dataset, {evaluator.metricName: "rmse"}))


def create_config():
    """
    define the model configuration file,
    Can refer to:https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0061.html
    :return:
    """
    schema_model = json.loads(
        '{"model_algorithm":"iris_classification","model_source":"custom","tunable":"false","model_type":"Spark_MLlib","metrics":{},"apis":[{"protocol":"http","url":"/","method":"post","request":{"Content-type":"application/json","data":{"type":"object","properties":{"data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"object","properties":{"sepal-length":{"type":"number"},"sepal-width":{"type":"number"},"petal-length":{"type":"number"},"petal-width":{"type":"number"}}}]}}}}}},"response":{"Content-type":"application/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"object","properties":{"predictionClass":{"type":"string"}}}]}}}}}]}',
        object_pairs_hook=OrderedDict)

    res_properties = collections.OrderedDict()
    col_num = 4
    input_name_list = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
    input_type_list = ["number", "number", "number", "number"]

    for i in range(col_num):
        index = input_name_list[i]
        res_properties[index] = {"type": input_type_list[i]}

    output_type = {"type": "string"}
    model_engine = "Spark_MLlib"
    resp_properties = {"predictionClass": output_type}

    schema_model['model_type'] = model_engine
    schema_model["apis"][0]["request"]["data"]["properties"]["data"]["properties"]["req_data"]["items"][0][
        "properties"] = res_properties

    # the following four parameters require user customization
    schema_model["apis"][0]["response"]["data"]["properties"]["resp_data"]["items"][0]["properties"] = resp_properties
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


# train model method
def train_model():
    print_title("download iris data!")
    download_dataset()
    print_title("load data!")
    # load iris.csv into Spark dataframe
    df = spark.createDataFrame(pd.read_csv('/tmp/iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))
    print("First 10 rows of Iris dataset:")
    df.show(10)
    df.printSchema()

    # convert features
    assembler = pyspark.ml.feature.VectorAssembler(inputCols=['sepal-length', 'petal-width', 'petal-length', 'sepal-width'], outputCol='features')

    # convert text labels into indices
    label_indexer = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='label').fit(df)

    # train
    lr = pyspark.ml.classification.LogisticRegression(regParam=0.01)
    label_converter = pyspark.ml.feature.IndexToString(inputCol='prediction', outputCol='predictionClass', labels=label_indexer.labels)
    pipeline = Pipeline(stages=[assembler, label_indexer, lr, label_converter])

    # fit the pipeline to training documents.
    model = pipeline.fit(df)
    model_local_path = os.path.join(LOCAL_MODEL_PATH, MODEL_NAME)

    # save model
    model.save(model_local_path)
    calculate_metric_value(model, df)


if __name__ == '__main__':
    train_model()
    create_metric()
    create_config()
    upload_to_obs()
