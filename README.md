# Amazon SageMaker End to End Workshop

This project was designed to provide an end to end experience on Amazon SageMaker.

It has been adapted from an [AWS blog post](https://aws.amazon.com/blogs/ai/predicting-customer-churn-with-amazon-machine-learning/). 

Losing customers is costly for any business. Identifying unhappy customers early on gives you a chance to offer them incentives to stay.  In this workshop we'll use machine learning (ML) for automated identification of unhappy customers, also known as customer churn prediction.

In this workshop we will use Gradient Boosted Trees (XGBoost) to Predict Mobile Customer Departure.

## The Data

Mobile operators have historical records that tell them which customers ended up churning and which continued using the service. We can use this historical information to train an ML model that can predict customer churn. After training the model, we can pass the profile information of an arbitrary customer (the same profile information that we used to train the model) to the model to have the model predict whether this customer will churn. 

The dataset we use is publicly available and was mentioned in [Discovering Knowledge in Data](https://www.amazon.com/dp/0470908742/) by Daniel T. Larose. It is attributed by the author to the University of California Irvine Repository of Machine Learning Datasets. The `Data sets` folder that came with this notebook contains the churn dataset.

The dataset can be [downloaded here.](https://bcs.wiley.com/he-bcs/Books?action=resource&bcsId=11704&itemId=0470908742&resourceId=46577)

## Resources (Workshop Structure)

To put our model in production we will use some features of SageMaker. Workshop is structured as following:

0. **Introduction**: Initial setup on [Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) environment;
1. **DataPrep**: Load churn dataset, tranform it on [Amazon SageMaker Data Wrangler](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html), and export it to S3;
2. **Modeling**: Create a XGBoost model using [Amazon SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html) and keep track of each training job with [Amazon SageMaker Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html) and also debug our model with [Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html);
3. **Evaluation**: Check model accuracy with [Amazon SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html) and explainability using [Amazon SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html);
4. **Deployment**: Host our model on [Model hosting](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html) and batch inference on [Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html);
5. **Monitoring**: Monitor our model for concept drift with [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-model-monitor.html);
6. **Pipelines**: Create a [Amazon SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html) to run our entire process.

## Getting Started

Although we recommend that you follow and run the Labs in order, _this workshop was built in a way that you can skip labs or just do those that interest you the most_ (e.g. you can just run the last Lab, or just run labs 4 an 5, or lab 1 and 4, etc.). Running the labs in order help us understand the natural flow of an ML project and may make more sense.

> This is only possible because we leverage the design of SageMaker where each component is independent from each other (e.g. training jobs, hosting, processing) and customers have the freedom to use those that fit better to their use-case.

**The `0-Introduction` lab is the only Lab that is strictly required to setup some basic things like creating S3 buckets, installing packages, etc.)**

---
## [You can go here directly to the first lab 0-Introduction](./0-Introduction/introduction.ipynb)

## Run any module independently

Remember that the `0-Introduction` lab **is mandatory**, no matter which module you will run. Following ones, can be executed independently (just follow the instructions for setup in each lab):

## [1-DataPrep](./1-DataPrep/data_preparation.ipynb)

## [2-Modeling](./2-Modeling/modeling.ipynb)

## [3-Evaluation](./3-Evaluation/evaluation.ipynb)

## [4-Deployment-Batch](./4-Deployment/Batch/deployment_batch.ipynb)

## [4-Deployment-RealTime](./4-Deployment/RealTime/deployment_hosting.ipynb)

## [5-Monitoring](./5-Monitoring/monitoring.ipynb)

## [6-Pipelines](./6-Pipelines/pipelines.ipynb)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Questions / Issues?

Please raise an issue on this repo.
