# coding:utf-8
import collections
import json
import os
from collections import OrderedDict

import pyspark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql.functions import expr

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
LOCAL_MODEL_PATH = '/tmp/spark_model'
LOCAL_CONFIG_PATH = '/tmp/config.json'
LOCAL_METRIC_PATH = '/tmp/metric.json'
LOCAL_DATA_PATH = '/tmp/ratings.csv'
MODEL_NAME = 'spark_model'
CONFIG_NAME = 'config.json'
METRIC_NAME = 'metric.json'

LOCAL_CUSTOMIZE_SERVICE_PATH = '/tmp/customize_service.py'
CUSTOMIZE_SERVICE = 'customize_service.py'

# start local Spark
spark = pyspark.sql.SparkSession.builder.config("spark.driver.host", "localhost").master("local[*]").appName(
    "product_recommendation").getOrCreate()
sc = spark.sparkContext
metric_dict = {}


def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)


# download file from OBS
def download_dataset():
    print("Start to download dataset from OBS")
    try:
        train_file = os.path.join("obs://", data_path, "ratings.csv")
        mox.file.copy(train_file, LOCAL_DATA_PATH)
        print('Succeeded to download training dataset')
    except Exception:
        print('Failed to download training dataset from OBS !')
        raise Exception('Failed to download training dataset from OBS !')


# upload file to OBS
def upload_to_obs():
    try:
        # upload metric file to OBS
        print_title("upload model to obs !")
        mox.file.copy_parallel(LOCAL_MODEL_PATH, os.path.join("obs://", obs_path, "model", MODEL_NAME))

        # upload config file to OBS
        print_title("upload config to obs !")
        mox.file.copy(LOCAL_CONFIG_PATH, os.path.join("obs://", obs_path, "model", CONFIG_NAME))

        # upload custom service file to OBS
        if os.path.exists(LOCAL_CUSTOMIZE_SERVICE_PATH):
            print_title("upload customize_service to obs !")
            mox.file.copy(LOCAL_CUSTOMIZE_SERVICE_PATH, os.path.join("obs://", obs_path, "model", CUSTOMIZE_SERVICE))
        else:
            print("user own customize_service.py not exists")
    except Exception:
        print('Failed to upload training output to OBS !')
        raise Exception('Failed to upload training output to OBS !')

    return 0


# calculate the metric value
def calculate_metric_value(predict_result):
    """
    because modelarts console UI only support displaying value: f1, recall, precision, and accuracy,
    this step maps `mae` to `recall`, `mse` maps to `precision`, and `rmse` maps to `accuracy`.
    :return:
    """
    result = predict_result.withColumnRenamed('_2', 'prediction')
    dataset = result.withColumn('label', expr('prediction'))
    evaluator = RegressionEvaluator(predictionCol="prediction")

    # calculate metric
    metric_dict["f1"] = 0
    metric_dict["recall"] = (evaluator.evaluate(dataset, {evaluator.metricName: "mae"}))
    metric_dict["precision"] = (evaluator.evaluate(dataset, {evaluator.metricName: "mse"}))
    metric_dict["accuracy"] = (evaluator.evaluate(dataset, {evaluator.metricName: "rmse"}))


# create config file
def create_config():
    """
    define the model configuration file,
    Can refer to:https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0061.html
    :return:
    """
    schema_model = json.loads(
        '{"model_algorithm":"product_recommendation","model_source":"custom","tunable":"false","model_type":"Spark_MLlib","metrics":{},"apis":[{"protocol":"http","url":"/","method":"post","request":{"Content-type":"application/json","data":{"type":"object","properties":{"data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"object","properties":{"input_1":{"type":"number"},"input_2":{"type":"number"},"input_3":{"type":"number"},"input_4":{"type":"number"}}}]}}}}}},"response":{"Content-type":"application/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"object","properties":{"predictresult":{"type":"number"}}}]}}}}}]}',
        object_pairs_hook=OrderedDict)

    res_properties = collections.OrderedDict()
    col_num = 2
    input_type_list = ["number", "number"]
    output_type = {"type": "number"}
    model_engine = "Spark_MLlib"

    for i in range(col_num):
        index = "input_%s" % (i + 1)
        res_properties[index] = {"type": input_type_list[i]}

    resp_properties = {"predictresult": output_type}

    schema_model['model_type'] = model_engine
    schema_model["apis"][0]["request"]["data"]["properties"]["data"]["properties"]["req_data"]["items"][0][
        "properties"] = res_properties
    schema_model["apis"][0]["response"]["data"]["properties"]["resp_data"]["items"][0]["properties"] = resp_properties

    with open(LOCAL_CONFIG_PATH, 'w') as f:
        json.dump(schema_model, f)

    print_title("create config file success!")


# create metric file,these data will be displayed in the evaluation results of the modelarts training assignments.
def create_metric():
    metric_model = json.loads(
        '{"total_metric":{"total_metric_meta":{},"total_reserved_data":{},"total_metric_values":{}}}',
        object_pairs_hook=OrderedDict)

    metric_model["total_metric"]["total_metric_values"]["f1"] = metric_dict['f1']
    metric_model["total_metric"]["total_metric_values"]["recall"] = metric_dict['recall']
    metric_model["total_metric"]["total_metric_values"]["precision"] = metric_dict['precision']
    metric_model["total_metric"]["total_metric_values"]["accuracy"] = metric_dict['accuracy']

    with open(LOCAL_METRIC_PATH, 'w') as f:
        json.dump(metric_model, f)


def train_model():
    print_title("download data!")
    download_dataset()
    print_title("load data!")
    data = sc.textFile(LOCAL_DATA_PATH)
    ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    rank = 10
    iteration_num = 10
    lambda_ = 0.02
    blocks = 100
    model = ALS.train(ratings, rank, iteration_num, lambda_, blocks)

    test_data = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    # print rates_and_preds
    MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))

    # save model
    print_title("save model!")
    model.save(sc, LOCAL_MODEL_PATH)

    # calculate metric value
    calculate_metric_value(spark.createDataFrame(predictions))


if __name__ == '__main__':
    train_model()
    create_config()
    upload_to_obs()
