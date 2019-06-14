# coding:utf-8
import collections
import json
import os
from collections import OrderedDict

import pyspark
from obs import ObsClient
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql.functions import expr

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
LOCAL_CONFIG_PATH = '/tmp/config.json'
LOCAL_METRIC_PATH = '/tmp/metric.json'
LOCAL_DATA_PATH = '/tmp/ratings.csv'
MODEL_NAME = 'spark_model'
CONFIG_NAME = 'config.json'
METRIC_NAME = 'metric.json'

LOCAL_CUSTOMIZE_SERVICE_PATH = '/tmp/customize_service.py'
CUSTOMIZE_SERVICE = 'customize_service.py'
metric_dic = {}

# start local Spark
spark = pyspark.sql.SparkSession.builder.config("spark.driver.host", "localhost").master("local[*]").appName(
    "product recommendation").getOrCreate()
sc = spark.sparkContext


def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)


# download file from OBS
def download_dataset():
    print("Start to download dataset from OBS")

    obs_client = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    try:
        bucket_name = data_path.split("/", 1)[0]
        train_file = data_path.split("/", 1)[1] + "/ratings.csv"
        resp = obs_client.getObject(bucket_name, train_file, downloadPath=LOCAL_DATA_PATH)
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
    try:
        bucket_name = obs_path.split("/", 1)[0]
        work_metric = obs_path.split("/", 1)[1] + '/'
        work_model = obs_path.split("/", 1)[1] + '/model/'
        model_file = work_model + MODEL_NAME
        config_file = work_model + CONFIG_NAME
        metric_file = work_metric + METRIC_NAME
        customize_service_file = work_model + CUSTOMIZE_SERVICE

        obs_client.putContent(bucket_name, work_model, content=None)

        # upload metric file to OBS
        print_title("upload model to obs !")
        obs_client.putFile(bucket_name, model_file, file_path=LOCAL_MODEL_PATH)

        # upload config file to OBS
        print_title("upload config to obs !")
        obs_client.putFile(bucket_name, config_file, file_path=LOCAL_CONFIG_PATH)

        # upload metric file to OBS
        print_title("upload metric to obs !")
        obs_client.putFile(bucket_name, metric_file, file_path=LOCAL_METRIC_PATH)

        # upload custom service file to OBS
        if os.path.exists(LOCAL_CUSTOMIZE_SERVICE_PATH):
            print_title("upload customize_service to obs !")
            obs_client.putFile(bucket_name, customize_service_file, file_path=LOCAL_CUSTOMIZE_SERVICE_PATH)
        else:
            print("user own customize_service.py not exists")

    finally:
        obs_client.close()

    return 0


# calculate the metric value
def calculate_metric_value(predict_result):
    result = predict_result.withColumnRenamed('_2', 'prediction')
    dataset = result.withColumn('label', expr('prediction'))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    # calculate metric
    metric_dic["f1"] = (evaluator.evaluate(dataset, {evaluator.metricName: "f1"}))
    metric_dic["accuracy"] = (evaluator.evaluate(dataset, {evaluator.metricName: "accuracy"}))
    metric_dic["precision"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedPrecision"}))
    metric_dic["recall"] = (evaluator.evaluate(dataset, {evaluator.metricName: "weightedRecall"}))


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
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    # print ratesAndPreds
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))

    # save model
    print_title("save model!")
    model.save(sc, LOCAL_MODEL_PATH)

    # calculate metric value
    calculate_metric_value(spark.createDataFrame(predictions))


# create config file
def create_config():
    """
    define the model configuration file,
    Can refer to:https://support.huaweicloud.com/usermanual-modelarts/modelarts_02_0061.html
    :return:
    """
    schema_model = json.loads(
        '{"model_algorithm":"Product Recommendation","model_source":"custom","tunable":"false","model_type":"Spark_MLlib","metrics":{},"apis":[{"protocol":"http","url":"/","method":"post","request":{"Content-type":"application/json","data":{"type":"object","properties":{"data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"object","properties":{"input_1":{"type":"number"},"input_2":{"type":"number"},"input_3":{"type":"number"},"input_4":{"type":"number"}}}]}}}}}},"response":{"Content-type":"application/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"object","properties":{"predictresult":{"type":"number"}}}]}}}}}]}',
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
    # The following four parameters require user customization
    schema_model["metrics"]["f1"] = metric_dic['f1']
    schema_model["metrics"]["accuracy"] = metric_dic['accuracy']
    schema_model["metrics"]["precision"] = metric_dic['precision']
    schema_model["metrics"]["recall"] = metric_dic['recall']

    with open(LOCAL_CONFIG_PATH, 'w') as f:
        json.dump(schema_model, f)

    print_title("create config file success!")


# create metric file,these data will be displayed in the evaluation results of the modelarts training assignments.
def create_metric():
    metric_model = json.loads(
        '{"total_metric":{"total_metric_meta":{},"total_reserved_data":{},"total_metric_values":{}}}',
        object_pairs_hook=OrderedDict)

    metric_model["total_metric"]["total_metric_values"]["f1"] = metric_dic['f1']
    metric_model["total_metric"]["total_metric_values"]["recall"] = metric_dic['recall']
    metric_model["total_metric"]["total_metric_values"]["precision"] = metric_dic['precision']
    metric_model["total_metric"]["total_metric_values"]["accuracy"] = metric_dic['accuracy']

    with open(LOCAL_METRIC_PATH, 'w') as f:
        json.dump(metric_model, f)


if __name__ == '__main__':
    train_model()
    create_config()
    create_metric()
    upload_to_obs()
