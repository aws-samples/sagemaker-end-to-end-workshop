"""Evaluation script for measuring model accuracy."""
import argparse, os, subprocess, sys
import json
import os
import tarfile
import logging
import pickle

import pandas as pd
import xgboost


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def pip_install(package):
    logger.info(f"Pip installing `{package}`")
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    pip_install("sagemaker-experiments==0.1.31")
    
    # Instantiate SM Experiment Tracker
    from smexperiments.tracker import Tracker
    tracker = Tracker.load()
    
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.info("Loading test input data")
    test_path = "/opt/ml/processing/test/test-dataset.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions_probs = model.predict(X_test)
    predictions = predictions_probs.round()

    logger.info("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions_probs)

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    logger.info("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    logger.info("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    logger.info("Creating and logging plots to Studio")
    tracker.log_precision_recall(y_test, predictions_probs, title="Precision-recall for predicting Churn", output_artifact=True)
    tracker.log_roc_curve(y_test, predictions_probs, title="ROC Curve for predicting Churn", output_artifact=True)
    tracker.log_confusion_matrix(y_test, predictions, title="Confusion matrix for predicting Churn", output_artifact=True)
