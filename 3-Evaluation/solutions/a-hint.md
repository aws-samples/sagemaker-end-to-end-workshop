# Hint for exercise a.

#### Use the SageMaker Experiments Tracker API inside the Processing Job.

Documentation:

- https://sagemaker-experiments.readthedocs.io/en/latest/tracker.html

The `.load( ... )` classmethod may help:

- https://sagemaker-experiments.readthedocs.io/en/latest/tracker.html#smexperiments.tracker.Tracker.load

Obs.: 
The SageMaker XGBoost Docker image **doesn't come with the SageMaker Experiments SDK installed by default.**

- We could create a custom image from scratch, but to make things simpler as a workaround you can put this in your code:

```python
# ... imports

def pip_install(package):
    logger.info(f"Pip installing `{package}`")
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    pip_install("sagemaker-experiments==0.1.31")
    
    # ... rest of the code
  
```


Ok, want the solution?

[Click here](./a-solution.ipynb)