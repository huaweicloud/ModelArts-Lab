# coding:utf-8
import collections
import json
import os
from collections import OrderedDict

import pandas as pd
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString

import moxing as mox


obs_path = os.getenv('TRAIN_URL')
if obs_path is None:
    obs_path = ''
print("obs_path: " + str(obs_path))

data_path = os.getenv('DATA_URL')
if data_path is None:
    data_path = ''
print("data_path: " + str(data_path))

# define local temporary file names and paths
TRAIN_DATASET = 'iris.csv'
MODEL_NAME = 'spark_model'
CONFIG_NAME = 'config.json'
METRIC_NAME = 'metric.json'
LOCAL_MODEL_PATH = '/tmp/spark_model'
LOCAL_CONFIG_PATH = '/tmp/config.json'
LOCAL_METRIC_PATH = '/tmp/metric.json'
LOCAL_DATA_PATH = '/tmp/iris.csv'

# start local Spark
spark = pyspark.sql.SparkSession.builder.config("spark.driver.host", "localhost").master("local[*]").appName(
    "flower_classification").getOrCreate()
metric_dict = {}


def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)


# download file from OBS
def download_dataset():
    print("Start to download dataset from OBS")

    try:
        train_file = os.path.join("obs://", data_path, "iris.csv")
        mox.file.copy(train_file, LOCAL_DATA_PATH)
        print('Succeeded to download training dataset')
    except Exception:
        print('Failed to download training dataset from OBS !')
        raise Exception('Failed to download training dataset from OBS !')


# upload file to OBS
def upload_to_obs():
    try:
        # upload model to OBS
        print_title("upload model to obs !")
        mox.file.copy_parallel(LOCAL_MODEL_PATH, os.path.join("obs://", obs_path, "model", MODEL_NAME))

        # upload config file to OBS
        print_title("upload config to obs !")
        mox.file.copy(LOCAL_CONFIG_PATH, os.path.join("obs://", obs_path, "model", CONFIG_NAME))

        # upload metric file to OBS
        print_title("upload metric to obs !")
        mox.file.copy(LOCAL_METRIC_PATH, os.path.join("obs://", obs_path, "model", METRIC_NAME))
    except Exception:
        print('Failed to upload training output to OBS !')
        raise Exception('Failed to upload training output to OBS !')

    return 0


# calculate the metric value
def calculate_metric_value(multiclass_classification_model, df):
    dataset = multiclass_classification_model.transform(df)

    # calculate metric
    # evaluator = RegressionEvaluator(predictionCol="prediction")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    metric_dict["f1"] = (evaluator.evaluate(dataset, {evaluator.metricName: "f1"}))
    metric_dict["recall"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedPrecision"}))
    metric_dict["precision"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedRecall"}))
    metric_dict["accuracy"] = (evaluator.evaluate(dataset, {evaluator.metricName: "accuracy"}))


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
    assembler = VectorAssembler(inputCols=['sepal-length', 'petal-width', 'petal-length', 'sepal-width'], outputCol='features')

    # convert text labels into indices
    label_indexer = StringIndexer(inputCol='class', outputCol='label').fit(df)

    # train
    lr = LogisticRegression(regParam=0.01)
    label_converter = IndexToString(inputCol='prediction', outputCol='predictionClass', labels=label_indexer.labels)
    pipeline = Pipeline(stages=[assembler, label_indexer, lr, label_converter])

    # fit the pipeline to training documents.
    model = pipeline.fit(df)

    # save model
    model.save(LOCAL_MODEL_PATH)
    calculate_metric_value(model, df)


if __name__ == '__main__':
    train_model()
    create_metric()
    create_config()
    upload_to_obs()
