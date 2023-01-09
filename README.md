# AzureML experiment tracking tutorial
In this repo, we will show how AzureML can help your training process.

## The problem
The ML problem we are solving in this project is the noisy [two moon problem](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons).
The train and test data looks like this:

![train and test data](images/data.png)
You can find the data in the `data` folder and the code to generate the data in `azureml_scripts/create_dataset.py`.

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
- All the metadata, hyperparameters and metrics of our experiments in a single place.
   ![experiment_overview](images/experiment_overview.jpg)
- We can log and store images, files and models related to our experiments in a single place.
   ![experiment_overview](images/logged_decison_boundary.jpg)
- We can register models and their artifacts in a single place, and we can easily download them for future usage.
  - ![experiment_overview](images/registerd_artifacts.jpg)

## Preparation
In this repo, we assume you have a working AzureML workspace.
In the workspace, you should have a compute cluster and optionally a compute instance.
You can follow preform all the step in this tutorial either on your local machine or on the compute instance.

### Preparing your local machine
You can only submit jobs from your local machine if the AzureML workspace is accessible from your local machine.
Depending on the network setup, this might not be the case.
If that is the case you should use a compute instance or contact your IT department.
Here, I assume that your workspace is accessible from your local machine.

Another important thing is that the step "Registering the dataset in AzureML" cannot be performed on M1 Macs.
If you are using a M1 Mac, you should run this step on a compute instance.
All other steps can be performed on M1 Macs.

The first step is to make sure that the Azure CLI is installed on your local machine.
You can find the installation instructions [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest).
After installation, make sure that you are logged in to your Azure account using:
```bash
az login
```

Next, we need to obtain the workspace information using a `config.json` file.
You can obtain this file by:

1. Go to your AzureML resource in the [Azure Portal](https://portal.azure.com).
2. Click on the download "Download config.josn" button.
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
10. Click on the "VS Code" button to connect your VS Code to the compute instance. (Sometime you need to try this a few times. Especially, if you just created the compute instance.)
11. In the VS Code terminal, navigate to your project directory at `~/cloudfiles/code/Users/<YOUR_USER_NAME>`.

One important thing to note is that the compute instance has two disks:
- `~/cloudfiles`: This is network disk that is shared between all compute instances in your workspace. All work on this is automatically saved to the cloud. So, if your restart or kill your compute instance, you will not lose any work. However, this comes at the price of slower disk access.
- `~/localfiles`: This is a local disk that is only accessible from your compute instance. This disk is faster than the `~/cloudfiles` disk. However, if you restart or kill your compute instance, you will lose all work on this disk.


### Install the Python environment
In this tutorial, you can eiter use Poetry or Pip to install the required python packages.
We have both options since we will be showing how you can use both Poetry and Pip in AzureML.

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


### Registering the dataset in AzureML
TODO
Important: This script does not work on M1 Macs. 
So, we made the required packages optional. 
You can install on a compatable machine using:
- pip: `pip install -r requirements-azuremlruntime.txt`
- poetry: `poetry install -E azuremlruntime`

If you are using a M1 Mac, it might be easier to use an compute instance.

TODO
```bash
python azureml_scripts/create_dataset.py
```

### Running the code in AzureML
TODO
```bash
python azureml_scripts/submit_job.py --compute_target <your_compute_target>
```

#### Accessing a registered model
TODO
```bash
python azureml_scripts/download_model.py --model <your_model_name>:<your_model_version> --output <your_output_dir>
```

### Running the code in AzureML on a custom Docker image
TODO
```bash
python azureml_scripts/submit_job_custom_docker_image.py --docker_image <your_docker_image> --compute_target <your_compute_target>
```




