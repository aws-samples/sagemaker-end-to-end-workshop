
import boto3
import sagemaker



from sagemaker.sklearn.processing import SKLearnProcessor

# Processing step for feature engineering
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name=f"{base_job_prefix}/sklearn-CustomerChurn-preprocess",  # choose any name
    sagemaker_session=sagemaker_session,
    role=role,
)




from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# Training step for generating model artifacts
model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/CustomerChurnTrain"
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
    region=region,
    version="1.0-1",
    py_version="py3",
    instance_type=training_instance_type,
)
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=1,
    output_path=model_path,
    base_job_name=f"{base_job_prefix}/CustomerChurn-train",
    sagemaker_session=sagemaker_session,
    role=role,
)
xgb_train.set_hyperparameters(
    objective="binary:logistic",
    num_round=50,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    silent=0,
)



from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

# Processing step for evaluation
script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name=f"{base_job_prefix}/script-CustomerChurn-eval",
    sagemaker_session=sagemaker_session,
    role=role,
)



from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

# Register model step that will be conditionally executed
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                "S3Uri"
            ]
        ),
        content_type="application/json",
    )
)