### Tip for exercise a.

1. Remember to create a Trial inside the existing SageMaker Experiment (look at the previous example in the notebook).

2. Each training job can spin up separated infrastructure. So a simple loop calling the CreateTrainingJob API multiple times will run separately (loop calling the `sagemaker.estimator.Estimator.fit( ... )`)

3. Take a look in this doc (see the arguments in `.fit( ... )`):

    - https://sagemaker.readthedocs.io/en/v2.42.0/api/training/estimators.html#sagemaker.estimator.Estimator.fit

Ok, want the solution?

[Click here](./a-solution.ipynb)