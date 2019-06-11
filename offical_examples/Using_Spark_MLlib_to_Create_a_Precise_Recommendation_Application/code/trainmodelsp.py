# coding:utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
from obs import ObsClient
from pyspark import SparkConf,SparkContext
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from collections import OrderedDict
import collections
import json

AK=os.getenv('MINER_USER_ACCESS_KEY')
if AK is None:
    AK=''

SK=os.getenv('MINER_USER_SECRET_ACCESS_KEY')
if SK is None:
    SK=''

obs_endpoint=os.getenv('MINER_OBS_URL')
if obs_endpoint is None:
    obs_endpoint=''
print ("obs_endpoint: " + str(obs_endpoint))

obs_path=os.getenv('TRAIN_URL')
if obs_path is None:
    obs_path=''
print ("obs_path: " + str(obs_path))

data_path=os.getenv('DATA_URL')
if data_path is None:
    data_path=''
print ("data_path: " + str(data_path))

LOCAL_MODEL_DIR = '/tmp/recommendmodel'
LOCAL_CONFIG_DIR = '/tmp/config.json'
LOCAL_METRIC_DIR = '/tmp/metric.json'
LOCAL_DATA_DIR = '/tmp/ratings.csv'
MODEL_NAME = 'recommendmodel'
CONFIG_NAME = 'config.json'
METRIC_NAME = 'metric.json'

def print_title(title=""):
    print("=" * 15 + " %s " % title + "=" * 15)

def download_dataset():
    print("Start to download dataset from OBS")

    TestObs = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    try:
        bucketName = data_path.split("/",1)[0]
        resultFileName = data_path.split("/",1)[1] + "/ratings.csv"
        resp = TestObs.getObject(bucketName, resultFileName, downloadPath=LOCAL_DATA_DIR)
        if resp.status < 300:
            print('Succeeded to download training dataset')
        else:
            print('Failed to download ')

    finally:
        TestObs.close()

def upload_to_obs():

    TestObs = ObsClient(AK, SK, is_secure=True, server=obs_endpoint)

    bucketName = obs_path.split("/",1)[0]
    workmetric = obs_path.split("/",1)[1] + '/'
    workmodel = obs_path.split("/",1)[1] + '/model/'
    #workconfig = obs_path.split("/",1)[1] + '/config/'
    filemodel = workmodel + MODEL_NAME
    fileconfig = workmodel + CONFIG_NAME
    filemetric = workmetric + METRIC_NAME
    #resultFileName = obs_path.split("/",1)[1] + '/model/xgboost.m'
    #configName = obs_path.split("/",1)[1] + '/config/config.json'
    TestObs.putContent(bucketName, workmodel, content=None)
    #TestObs.putContent(bucketName, workconfig, content=None)
    print_title("upload model to obs !")
    TestObs.putFile(bucketName, filemodel, file_path=LOCAL_MODEL_DIR)
    print_title("upload config to obs !")
    TestObs.putFile(bucketName, fileconfig, file_path=LOCAL_CONFIG_DIR)
    print_title("upload metric to obs !")
    TestObs.putFile(bucketName, filemetric, file_path=LOCAL_METRIC_DIR)
    return 0

def train_model():

    print_title("download data!")
    download_dataset()
    print_title("load data!")
    conf = SparkConf().setAppName("testapr").setMaster("local")
    sc = SparkContext(conf=conf)
    data = sc.textFile(LOCAL_DATA_DIR)
    ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    rank = 10
    numIterations = 10
    lambda_ = 0.02
    blocks = 100
    model = ALS.train(ratings, rank, numIterations, lambda_, blocks)

    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    #print ratesAndPreds
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))

    print_title("save model!")
    model.save(sc,LOCAL_MODEL_DIR)
   

def create_config():
    
    schema_model=json.loads('{"model_algorithm":"gbtree_classification","model_type":"Spark_MLlib","metrics":{"f1":0.345294,"accuracy":0.462963,"precision":0.338977,"recall":0.351852},"apis":[{"procotol":"http","url":"/","method":"post","request":{"Content-type":"applicaton/json","data":{"type":"object","properties":{"data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"object","properties":{"input_1":{"type":"number"},"input_2":{"type":"number"},"input_3":{"type":"number"},"input_4":{"type":"number"}}}]}}}}}},"response":{"Content-type":"applicaton/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"object","properties":{"predictresult":{"type":"number"}}}]}}}}}]}',object_pairs_hook=OrderedDict)

    res_properties = collections.OrderedDict()

    col_num = 2

    input_type_list = ["number","number"]

    output_type = {"type": "number"}

    model_engine = "Spark_MLlib"

    for i in range(col_num):
        index = "input_%s" % (i+1)
        res_properties[index] = {"type": input_type_list[i]}

    resp_properties={"predictresult":output_type}

    schema_model['model_type']=model_engine
    schema_model["apis"][0]["request"]["data"]["properties"]["data"]["properties"]["req_data"]["items"][0]["properties"]=res_properties
    schema_model["apis"][0]["response"]["data"]["properties"]["resp_data"]["items"][0]["properties"]=resp_properties
    schema_model["metrics"]["f1"]=0.35
    schema_model["metrics"]["accuracy"]=0.49
    schema_model["metrics"]["precision"]=0.35
    schema_model["metrics"]["recall"]=0.36

    with open(LOCAL_CONFIG_DIR, 'w') as f:
        json.dump(schema_model, f)
    
    print_title("create config file success!")

def create_metric():

    metric_model=json.loads('{"total_metric":{"total_metric_meta":{},"total_reserved_data":{},"total_metric_values":{"f1_score":0.814874,"recall":0.8125,"precision":0.817262,"accuracy":1.0}}}',object_pairs_hook=OrderedDict)

    metric_model["total_metric"]["total_metric_values"]["f1_score"]=0.85
    metric_model["total_metric"]["total_metric_values"]["recall"]=0.88
    metric_model["total_metric"]["total_metric_values"]["precision"]=0.90
    metric_model["total_metric"]["total_metric_values"]["accuracy"]=1.5

    with open(LOCAL_METRIC_DIR, 'w') as f:
        json.dump(metric_model, f)

if __name__ == '__main__':
    train_model()
    create_config()
    create_metric()
    upload_to_obs()
