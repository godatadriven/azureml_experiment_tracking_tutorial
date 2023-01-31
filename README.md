# AzureML experiment tracking tutorial
In this repo, we will show how AzureML can help your training process.

## The problem
The ML problem we are solving in this project is the noisy [two moon problem](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons).
The train and test data sets looks like this:

![train and test data](images/data.png)
You can find the data in the `data` folder and the code to generate the data  in `azureml_scripts/create_dataset.py`.

We will use a Random Forest classifier from scikit-learn to solve this problem.
The training process consists of the following steps:
1. Load the data.
2. Preform a grid search to find the best hyperparameters.
3. Evaluate the model on the test data.
4. Show the resulting decision boundary.
5. Save the model.

## The end goal
This tutorial has two goals. Firstly, we want to move our training process into the cloud such that we can leverage more computing power and train our model faster. 
Secondly, we want to track our experiments in a single place such that we can track experiments over time and can compare different runs.

When we are done we should have the following results:
- All the metadata, hyperparameters and metrics of our experiments are stored a single place.
   ![experiment_overview](images/experiment_overview.jpg)
- We can log and store images, files and models related to our experiments in a single place.
   ![experiment_overview](images/logged_decison_boundary.jpg)
- We can register models and their artifacts in a single place, and we can easily download them for future usage.
  - ![experiment_overview](images/registerd_artifacts.jpg)

## Preparation
In this repo, we assume you have a working AzureML workspace.
In the workspace, you should have a compute cluster and optionally a compute instance.
You can preform all the step in this tutorial either on your local machine or on the compute instance.


### Preparing your local machine
In this tutorial, you need to be able to submit jobs from your machine. Depending on your network setup, your AzureML workspace might not be accessible from your local machine.
In this case, you should either use a compute instance or contact your IT department.
From here on, I assume that your workspace is accessible from your local machine.


Another important thing is that the step "Registering the dataset in AzureML" cannot be performed on M1 Macs.
If you are using an M1 Mac, you should run this step on a compute instance.
All other steps can be performed on M1 Macs.

The first step is ensuring that the Azure CLI is installed on your local machine.
You can find the installation instructions [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest).
After installation, make sure that you are logged in to your Azure account using the following command
```bash
az login
```

Next, we need to obtain the workspace information using a `config.json` file.
You can obtain this file by:

1. Go to your AzureML resource in the [Azure Portal](https://portal.azure.com).
2. Click on the download "Download config.json" button.
3. Place the downloaded config.json file in the same directory as this README.md file.

![config download button](images/config_download.jpg)

The only thing left to do is to install the required python packages.
For this see the "Install the Python environment" section below.

### Preparing your compute instance
An compute instance is a Linux based virtual machine that you can rent via AzureML.
You can do this by:

1. Go to your AzureML workspace at [ml.azure.com](https://ml.azure.com/).
2. Got to the "Compute instance" type and click on the "+ new" button.
![compute instance](images/create_compute_overview.jpeg)
3. Give the compute instance a name.
4. Make sure that the compute instance is located in the same region as your workspace.
5. Pick a VM size. You can a lightweight VM size for this tutorial.
   ![compute instance](images/create_compute.jpg)
6. Click on Advanced settings.
7. If your workspace is in a VNET, you should turn on "Virtual network" and select the VNET of your workspace.
   ![compute instance](images/create_compute_vnet.jpg)
8. Click on "Create" to create the compute instance.
9. Wait 5-10 min until the machine is ready.
10. Click on the "VS Code" button to connect your VS Code to the compute instance. (Sometimes, you need to try this a few times. Especially if you just created the compute instance.)
11. In the VS Code terminal, navigate to your project directory at `~/cloudfiles/code/Users/<YOUR_USER_NAME>`.
12. Run `conda init bash` and restart your terminal.

One important thing to note is that the compute instance has two disks:
- `~/cloudfiles`: This is a network disk that is shared between all compute instances in your workspace. All work on this is automatically saved to the cloud. So, if your restart or kill your compute instance, you will not lose any work. However, this comes at the price of slower disk access. Also be aware that if you save large datasets on this network drive, you get a bill for the storage.
- `~/localfiles`: This is a local disk that is only accessible from your compute instance. This disk is faster than the `~/cloudfiles` disk. However, if you restart or kill your compute instance, you will lose all work on this disk.


### Install the Python environment
In this tutorial, you can eiter use Poetry or Pip to install the required python packages.
We have both options in this repo since we will be showing how you can use both [Poetry](https://python-poetry.org/) and Pip in AzureML.

#### Poetry
If poetry is not installed on your machine, you can install it using:
```bash
pip install poetry
```

You can install the required packages by running:
```bash
poetry install
```
Note: we are using Python 3.8 in this project since this the default Python version in AzureML.
If poetry cannot find the correct Python version, you can tell it where to find it by running:
```bash
poetry env use path/to/python3.8
```

#### Pip
Alternatively, you can install the required packages using pip by running:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Note: make sure run this inside a virtual environment.


## The tutorial
This tutorial will first explore the existing ML solution and run it locally.
Then, we will add experiment tracking to the existing solution such that we can track our experiments in AzureML.
Thirdly, we will move the training process into the cloud to leverage more computing power and train our model faster.
Then, we will explore how we can access the registered models for future usage.
Finally, we explore how we can customize the training environment with a custom Docker image.

### Running the code locally
First, we will run the existing code locally to see how it works.
To do this, we will use the `azureml_tutorial/train_original.py` script.
This scripts performs the following steps:
1. Load the data.
2. Preform a grid search to find the best hyperparameters.
3. Evaluate the model on the test data.
4. Log the resulting hyperparameters and metrics.
5. Visualize the resulting decision boundary.
6. Save the model.

Please take some time to read through the code and comments in the `azureml_tutorial/train_original.py` script.
Make sure that you understand what is happening in each step.
Once you are done, you can run the script using:

```bash
python azureml_tutorial/train_original.py --train_dataset data/train.csv --test_dataset data/test.csv 
```

After running the script, you should see the following output:
```
Hyper-parameters:
gridsearch/n_estimators [10, 25, 100]
gridsearch/criterion ['gini', 'entropy']
gridsearch/max_depth [2, 5, 10, None]
Selected hyper-parameters:
selected/criterion gini
selected/max_depth None
selected/n_estimators 25
Metrics:
accuracy 0.975
```
Furthermore, you should see a plot of the decision boundary stored in `decision_boundary.png` in your `cwd`.
Finally, you should see a file `model.joblib` in your `cwd`.
See, this [link](https://scikit-learn.org/stable/model_persistence.html) for more details on how sklearn uses joblib to store models.

Now, we know how our training process works.
Next, we will add experiment tracking to the existing solution such that we can track our experiments in AzureML.

### Adding experiment tracking
In this section, we will add experiment tracking to the existing solution.
The code will still run locally, but the logs will be collected in the clouds.

The first thing we need to do is tell MLFlow where to store the logs.
We can do this using a MLFlow tracking URI ([docs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)).
AzureML you can get and set the tracking URI by running:
```python
import mlflow
from azureml.core import Workspace

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
```
So, this will be the first thing we need to do in our training script.

Next, we want to send our hyper-parameters and metrics to AzureML.
We can do this by using the `mlflow.log_param` ([docs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_params)) and `mlflow.log_metric` ([docs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metrics)) functions.
Important to note is that mlflow makes a distinction between hyper-parameters and metrics and thus stores them in different places.
By using the `mlflow.log_param` function, we tell mlflow that the value is a hyper-parameter and by using the `mlflow.log_metric` function, we tell mlflow that the value is a metric.
So, in our training script, we need to our print based logging logic with the following:

```python
import mlflow

...
for param, value in param_grid.items():
    mlflow.log_param(f"gridsearch/{param}", str(value))

...
for k, v in grid_search.best_params_.items():
    mlflow.log_param(f"selected/{k}", v)

...

mlflow.log_metric("accuracy", accuracy)
...
```

The last thing we need to do is upload our training artifacts,, like `model.joblib` and `decision_boundary.png`, to AzureML.
We can do this by using the `mlflow.log_artifact` ([docs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact)) function.
This function can upload any file on disk.
MLFlow also has the utility function `mlflow.log_figure` ([docs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure)) that can be used to upload a matplotlib figure directly from memory.
We can use these functions to upload our artifacts to AzureML as follows:

```python
figure = create_decision_boundary_figure(model, df_test)
mlflow.log_figure(figure, "decision_boundary.png")
...
# We use a tmp dir here because we keep the model saved locally.
# However, `log_artifact` needs a path to a file stored on disk.
with tempfile.TemporaryDirectory() as tmp_dir:
    dump(model, f"{tmp_dir}/model.joblib")
    mlflow.log_artifact(f"{tmp_dir}/model.joblib")
```

We have added all these changes to the `azureml_tutorial/train_with_mlflow.py` script.
Have a look at the code and make sure you understand what is happening.
When you are done, you can run the script using:

```bash
python azureml_tutorial/train_with_mlflow.py --train_dataset data/train.csv --test_dataset data/test.csv 
```

In the AzureML portal, you should now see that your experiment has been queued.
Once the experiment has finished, you should see the following:
- All hyper-parameters and metrics:
  ![experiment_overview](images/experiment_overview.jpg)
- The decision boundary should be logged as an artifact:
  ![experiment_overview](images/logged_decison_boundary.jpg)
- Your model be registered:
    - ![experiment_overview](images/registerd_artifacts.jpg)

Now, we have added experiment tracking to the existing solution.
However, we are still running the training process locally.
Now, let's move the training process into the cloud.

### Registering the dataset in AzureML
Before we can run our training process in the cloud, we need to make sure that our data is available in the cloud.
We do this by uploading our dataset to a blob storage and registering it in AzureML.
AzureML already has a default blob storage connected to it.
You can use the Python SDK to upload your data to this blob storage as follows:

```python
from azureml.core import Dataset, Workspace

workspace = Workspace.from_config()
# The default datastore is the blob storage connected to AzureML.
datastore = workspace.get_default_datastore()
Dataset.File.upload_directory("/path/to/local/files", (datastore, 'path/on/the/blob_store'))
```
After this, your data should be available in the blob storage. 
To download the data in a job, we need to remember the path on the blob storage. 
This can be annoying, so we can make it a bit easier by registering the dataset. 
Registering a dataset is just a matter of telling AzureML where the data is stored and giving it a name.
Then in the future, we can just download the data by using `Dataset.get_by_name(workspace, "YOUR_DATASET_NAME").download("path/to/download/location")`.
We can register a dataset as follows:

```python
from azureml.core import Dataset, Workspace
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()

dataset = Dataset.File.from_files(path=[(datastore, 'path/on/the/blob_store')])
dataset.register(workspace, name="moon_dataset")
```

We have implemented this in the `azureml_scripts/create_dataset.py` script.
Have a look at the code and make sure you understand what is happening.

Sadly, this script does not work on `arm` machines like the M1 Macs.
It only works on `x86_64` machines.
During the creation on the virtual environment, poetry/pip automatically checks if you are on a `x86_64` machine and only install the required packages if your machine is compatable.
So if your machine is not compatible, you will get an `No module named 'azureml.dataprep'` error when you run the script.
If you are using a M1 Mac, it might be easier to use a compute instance.
At least, for this step.

When you are ready, you can run the script using:

```bash
python azureml_scripts/create_dataset.py
```

After you run the script, you should see the dataset in the AzureML portal.
You can find it under `Data` in the left menu.


### Running the code in AzureML
Now that we have access to our data in the cloud, we can run our training process in the cloud.
In this project we use multiple Python libraries, so we need to make sure that all the dependencies are available when we run our code.
In AzureML, we can do this by creating an Environment.
This is a Docker image that contains all the dependencies we need to run our code.
AzureML can build this image for us.
All we need to do is tell AzureML which packages we need using a `requirements.txt` file.
The AzureML SDK has a utility function for this called `Environment.from_pip_requirements` ([docs](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment.environment?view=azure-ml-py)), which we can use as follows:

```python
from azureml.core import Environment
env = Environment.from_pip_requirements(
    name="moon_model_env",
    file_path="absolute/path/to/requirements.txt",
)
# Python version must be added in this unclear way
env.python.conda_dependencies.set_python_version("3.8")
```

Another thing we need to do is tell AzureML which code we want to run and on which compute target we want to run it.
We can do this by creating a `ScriptRunConfig` ([docs](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig.scriptrunconfig?view=azure-ml-py)).
In this config, we indicate where our Python files are located.
AzureML will copy and upload these files to the compute target and mount them as a volume in the Docker container.
We also need to tell AzureML which bash command we want to run in the container.
We can do this as follows:

```python
from azureml.core import ScriptRunConfig
...
script_run_config = ScriptRunConfig(
    source_directory="absolute/path/to/src",
    command="python main.py", # Bash command to run
    compute_target="name_of_your_compute_cluster", # The name of the compute target can be found in the AzureML under Compute -> Compute clusters
    environment=env, # The environment we created above
)
```

The last thing we need to do is create an experiment and submit the run.
We can do this as follows:

```python
from azureml.core import Experiment
...
experiment = Experiment(workspace=workspace, name="name_of_experiment")
experiment.submit(script_run_config) # The config we created above
```

We have implemented this in the `azureml_scripts/submit_job.py` script. 
Look at the code to see how we combined the above code snippets. 
Once you have an understanding of what is happening, you can run the script using:
```bash
python azureml_scripts/submit_job.py --compute_target <your_compute_target>
```

This script submits a job to the compute cluster that runs the following 3 scripts in sequence:
1. `azureml_tutorial/download.py` This ensures that the data is downloaded from the blob storage.
2. `azureml_tutorial/train_with_mlflow.py` This runs the training process, just like we did locally.
3. `azureml_tutorial/register.py` This registers the model in the AzureML model registry.

#### Accessing a registered model
After the training process is done, we can access the registered model in the AzureML model registry.
You download the model from the UI or using the AzureML SDK.

If you want to download the model using the SDK, you can do this by using the `Model` class ([docs](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py)).
All you need to do is specify the name of the model and the version.
Have a look at the `azureml_scripts/download.py` script to see how this is done.
Do you want to try it out yourself?
You can run the script using:

```bash
python azureml_scripts/download_model.py --model <your_model_name>:<your_model_version> --output <your_output_dir>
```

### Running the code in AzureML on a custom Docker image
In the previous section, we let AzureML build a Docker image for us. 
This works fine as long as everything is installed using pip or conda. 
Sadly, this is not always the case.
For example, let's say you want to use Poetry to manage your dependencies. 
This is not directly compatible with AzureML.
Luckily, AzureML also allows us to use our own Docker image for scenarios like this. 
So, let's see how we can do this.

The first thing we need to do is create a Docker image.
In this project, we have created a Dockerfile that installs Poetry and all the dependencies.
You can find this Dockerfile in the `Dockerfile` root folder of this repo.

Next, we need to build the image and push it to a Docker registry.
You can do this three ways.
First, you can build it locally and push it to DockerHub.
```bash
docker build --platform=linux/amd64 -t <your_dockerhub_username>/<your_image_name>:<your_image_tag> .
docker push <your_dockerhub_username>/<your_image_name>:<your_image_tag>
```

Secondly, you can build it locally and push it to your private Azure Container Registry.
```bash
az login
az acr login --name <your_acr_name>
docker build --platform=linux/amd64 -t <your_acr_name>.azurecr.io/<your_image_name>:<your_image_tag> .
docker push <your_acr_name>.azurecr.io/<your_image_name>:<your_image_tag>
```

The third option is to let your Azure Devops CI/CD pipeline build and push the image for you.
You can see an example of this in the `azure-pipelines.yml` file in the root of this repo.

At this point, I assume you have build and pushed your Docker image to a registry.
Now, let's change the job submission script such that it use this custom image.
All we need to do is define a different the `Environment` argument in the `ScriptRunConfig`.
We can do this as follows:

```python
# Create the environment
env = Environment(workspace=workspace, name="custom_image")
# We now provide our own docker image
env.docker.base_image = "YOUR_DOCKER_IMAGE"
# Tell AzureML that we will manage the dependencies ourselves.
env.python.user_managed_dependencies = True
```

Another thing we need to do is change the bash command for the job since we want to use Poetry. 
You can see the complete code in the `azureml_scripts/submit_job_custom_docker_image.py` script.
Look at the code and ensure you understand what is happening.
When you are done, you can run the script using the following:

```bash
python azureml_scripts/submit_job_custom_docker_image.py --docker_image <your_docker_image> --compute_target <your_compute_target>
```

## Conclusion
In this tutorial, we have seen how you can use AzureML as an experiment tracking tool.
The big advantage of AzureML is that it is fully compatible with the MLflow API, which makes getting started easier, especially if you already know MLFlow.
An additional advantage is that you can test your experiment tracking code locally using PyTest-based integration tests since you can also start a local MLFlow server.

This tutorial also explored how we can run our code on a compute cluster in the cloud. Here, you have two options. Either use one of the pre-built Docker images or build a custom docker image.
At this point, you should have a good understanding of how you can use AzureML for experiment tracking. Now, it is up to you to start experimenting with AzureML.

## FAQ
### Can not install the required packages due to a cryptography package error
If you get the following error when setting up the environment in a compute instance, make sure that you have run:
```
conda init bash
```
For some reason, this is not done automatically by AzureML during the compute instance creation.
As, you cannot access the base conda environment.

### I get a no module named 'azureml.dataprep' error
This error most like occurs because you are using a none-`x86_64` platform.
The `azureml.dataprep` package is only available for `x86_64` platforms.
The easiest way to fix this is to create a new compute instance.
They always have an `x86_64` platform.