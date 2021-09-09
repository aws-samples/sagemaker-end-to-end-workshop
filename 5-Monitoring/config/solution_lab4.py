import pprint
from time import strftime, gmtime

import pandas as pd
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.model_monitor import DataCaptureConfig, DatasetFormat, DefaultModelMonitor

sess = boto3.Session()
sm = sess.client('sagemaker')
role = sagemaker.get_execution_role()

#Supress default INFO loggingd
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

def get_endpoint_from_lab4():
    print("Getting solution from Lab 4...")
    print("Please wait ~10 minutes for the endpoint to be deployed.")
    
    # Set the paths for the datasets saved locally
    path_to_lab2 = "/root/sagemaker-end-to-end-workshop/2-Modeling/"
    path_to_lab5 = "/root/sagemaker-end-to-end-workshop/5-Monitoring/"
    
    local_train_path = path_to_lab2 + 'config/train.csv'
    train_df = pd.read_csv(local_train_path, header=None)
    local_validation_path = path_to_lab2 + 'config/validation.csv'
    validation_df = pd.read_csv(local_validation_path, header=None)
    model_artifact_path = path_to_lab5 + 'config/model.tar.gz'
    inference_code_path = path_to_lab5 + 'config/inference.py'

    region = sess.region_name
    account_id = sess.client('sts', region_name=region).get_caller_identity()["Account"]
    bucket = 'sagemaker-studio-{}-{}'.format(sess.region_name, account_id)
    prefix = 'xgboost-churn'
    train_dir = f"{prefix}/train"
    val_dir = f"{prefix}/validation"
    model_dir = f"{prefix}/model"

    try:
        if sess.region_name == "us-east-1":
            sess.client('s3').create_bucket(Bucket=bucket)
        else:
            sess.client('s3').create_bucket(Bucket=bucket, 
                                            CreateBucketConfiguration={'LocationConstraint': sess.region_name})
    except Exception as e:
        print("Looks like you already have a bucket of this name. That's good. Uploading the data files...")

    # Return the URLs of the uploaded file, so they can be reviewed or used elsewhere
    print("Uploading data and model files to S3")
    s3url_train = sagemaker.s3.S3Uploader.upload(local_train_path, 's3://{}/{}'.format(bucket, train_dir))
    s3url_validation = sagemaker.s3.S3Uploader.upload(local_validation_path, 's3://{}/{}'.format(bucket, val_dir))
    s3url_model_artifact = sagemaker.s3.S3Uploader.upload(model_artifact_path, 's3://{}/{}'.format(bucket, model_dir))
    
    boto_sess = boto3.Session()
    region = boto_sess.region_name
    role = sagemaker.get_execution_role()
    sm_sess = sagemaker.session.Session()

    region = sess.region_name
    framework_version = '1.2-2'
    docker_image_name = sagemaker.image_uris.retrieve(framework='xgboost', region=region, version=framework_version)

    xgb_inference_model = Model(
        model_data=s3url_model_artifact,
        role=role,
        image_uri=docker_image_name,
    )

    data_capture_prefix = '{}/datacapture'.format(prefix)

    endpoint_name = "model-xgboost-customer-churn-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(f"Deploying Endpoint with name = {endpoint_name}...")
    predictor = xgb_inference_model.deploy( initial_instance_count=1, 
                                            instance_type='ml.m4.xlarge',
                                            endpoint_name=endpoint_name,
                                            data_capture_config=DataCaptureConfig(
                                                enable_capture=True,
                                                sampling_percentage=100,
                                                destination_s3_uri='s3://{}/{}'.format(bucket, data_capture_prefix),
                                                csv_content_types=['text/csv']
                                                
                                            )
                                       )
    return endpoint_name, predictor
    
