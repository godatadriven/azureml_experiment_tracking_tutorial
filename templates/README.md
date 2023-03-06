### AzureML with mlflow model management and fastapi deployment
In the repository, we will show you how AzureML service can be used to train and manage machine learning models and deploy them safely for consumption.

### The problem
The ML problem we are solving in this project is the noisy [two moon problem](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons).
The train and test data sets looks like this:

![train and test data](../images/data.png)
You can find the data in the `data` folder and the code to generate the data  in `azureml_scripts/create_dataset.py`.

We will use a Random Forest classifier from scikit-learn to solve this problem.
The training process consists of the following steps:
1. Load the data.
2. Preform a grid search to find the best hyperparameters.
3. Evaluate the model on the test data.
4. Show the resulting decision boundary.
5. Save the model.
6. Deploy it for inference.

### The end goal
This tutorial has two parts that are given below: 
1. [Experiment Tracking](azureml_tracking/README.md): Move our training process into the cloud such that we can leverage more computing power and train our model faster. Track our experiments in a single place such that we can track experiments over time and can compare different runs.

2. [Deployment](azureml_deployment/README.md): Use our trained model in production by creating a AzureML endpoint and deploying a custom fastapi container to the cloud. 

When we are done we should have the following results:
- All the metadata, hyperparameters and metrics of our experiments are stored a single place.
   ![experiment_overview](images/experiment_overview.jpg)
- We can log and store images, files and models related to our experiments in a single place.
   ![experiment_overview](images/logged_decison_boundary.jpg)
- We can register models and their artifacts in a single place, and we can easily download them for future usage.
  - ![experiment_overview](images/registerd_artifacts.jpg)

### Preparation
In this repo, we assume you have a working AzureML workspace.
In the workspace, you should have a compute cluster and optionally a compute instance.
You can preform all the step in this tutorial either on your local machine or on the compute instance.


#### Preparing your local machine
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

#### Preparing your compute instance
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



### FAQs
##### Can not install the required packages due to a cryptography package error
If you get the following error when setting up the environment in a compute instance, make sure that you have run:
```
conda init bash
```
For some reason, this is not done automatically by AzureML during the compute instance creation.
As, you cannot access the base conda environment.

##### I get a no module named 'azureml.dataprep' error
This error most like occurs because you are using a none-`x86_64` platform.
The `azureml.dataprep` package is only available for `x86_64` platforms.
The easiest way to fix this is to create a new compute instance.
They always have an `x86_64` platform.