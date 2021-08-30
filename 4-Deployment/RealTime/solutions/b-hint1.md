### Hint 1. for exercise b.

1. Take a look at these docs:
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/using_xgboost.html
    - https://sagemaker.readthedocs.io/en/v2.42.0/frameworks/xgboost/xgboost.html#sagemaker.xgboost.model.XGBoostModel
    
2. Add an custom inference script

Obs.: If you already deployed a model and wants to update it with new configuration just run ".deploy( ... )"  again but with an "update_endpoint" argument:

```
.deploy( ... , update_endpoint=True)
``` 

Ok, need another hint?

[Click here](./b-hint2.md)