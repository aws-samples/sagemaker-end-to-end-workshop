### Hint 2. for exercise b.

1. Take a look at these docs:
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/using_xgboost.html
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/xgboost.html#sagemaker.xgboost.model.XGBoostModel
    
2. Add an custom inference script (look at the `input_fn`, `predict_fn`, `output_fn` and `model_fn` functions)
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/using_xgboost.html#write-an-inference-script
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/using_xgboost.html#sagemaker-xgboost-model-server

Example here:

https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_inferenece_script_mode.ipynb

Ok, want the solution?

[Click here](./b-solution.ipynb)