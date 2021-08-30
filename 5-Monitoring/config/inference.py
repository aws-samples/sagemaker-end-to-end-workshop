import os
import pickle

import xgboost
import sagemaker_xgboost_container.encoder as xgb_encoders

# Same as in the training script
def model_fn(model_dir):
    """Load a model. For XGBoost Framework, a default function to load a model is not provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns:
        A XGBoost model.
        XGBoost model format type.
    """
    model_files = (file for file in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, file)))
    model_file = next(model_files)
    try:
        booster = pickle.load(open(os.path.join(model_dir, model_file), 'rb'))
        format = 'pkl_format'
    except Exception as exp_pkl:
        try:
            booster = xgboost.Booster()
            booster.load_model(os.path.join(model_dir, model_file))
            format = 'xgb_format'
        except Exception as exp_xgb:
            raise ModelLoadInferenceError("Unable to load model: {} {}".format(str(exp_pkl), str(exp_xgb)))
    booster.set_param('nthread', 1)
    return booster


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    The input_fn that just validates request_content_type and prints
    """
    
    print("Hello from the PRE-processing function!!!")
    
    if request_content_type == "text/csv":
        return xgb_encoders.csv_to_dmatrix(request_body)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )

def predict_fn(input_object, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.
    """
    return model.predict(input_object)[0]


def output_fn(prediction, response_content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    An output_fn that just adds a column to the output and validates response_content_type
    """
    print("Hello from the POST-processing function!!!")
    
    appended_output = "hello from pos-processing function!!!"
    predictions = [prediction, appended_output]

    if response_content_type == "text/csv":
        csv = ','.join(str(x) for x in predictions)
        return 
    else:
        raise ValueError("Content type {} is not supported.".format(response_content_type))
    