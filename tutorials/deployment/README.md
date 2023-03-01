## Tutorial section 2: 
#### Azureml fastapi online endpoint
This section will focus on creating a webapp to serve the model using the fastapi framework. There are four main steps:

1. Download the artifacts of our registered model from AzureML cloud.
2. Create a webapp to serve the model using fastapi.
3. Containerize the webapp and push it to Azure Container Registry.
4. Create an endpoint in AzureML and deploy our custom docker image to the endpoint.

#### 1. Download the model
This can be done using the sdk:
```python
from azureml.core import Model, Workspace
...
# Create a workspace
ws = Workspace.from_config()
# Get the model
model = Model(name="YOUR_MODEL_NAME", version="version", workspace=ws)
# Download the model to a target directory
model.download(target_dir="MODEL_OUTPUT_DIR")
```

It is implemented in `azureml_scripts/download_model.py` script. You can run the script using: 
```bash
poetry run python azureml_scripts/download_model.py --model <YOUR_MODEL_NAME> --output <MODEL_OUTPUT_DIR>
```

#### 2. Create a webapp to serve the model using fastapi.
Fastapi is a framework to build RESTapi in python.
```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Initialize the app
app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(content = {"message": "Hello world!"})
```
The code given above is a basic example of a fastapi webapp that displays a message when it is run. The fastapi app for serving the model is scripted in `azureml_deployment/src/main.py`.

You can run the server locally by:
```bash
uvicorn --host 0.0.0.0 --port 8000 azureml_deployment.src.main:app
```

We can also test the server using a post request with mock data. To test the server:
```bash
python azureml_deployment/sample_request.py
```
You shuld see the following output:
```
[{'x1': 1.0, 'x2': -1.0, 'y': 1}, {'x1': -1.0, 'x2': 1.0, 'y': 0}]
```

#### 3. Build the docker image in Azure container registry (ACR).
Azure Container Registry is used to manage container images and artifacts.
If you do not already have an ACR, you can:
1. Go to portal.azure.com > Azure Container Registry > Create
2. Choose the subscription, resource group and create a new ACR.
3. You should be able to see your newly created ACR in Azure portal.
4. Open your ACR and find your Login server `<YOUR_ACR>.azurecr.io` under the overview tab.
Now that we have a working fastapi app, we will create a docker image from `fastapi.Dockerfile`.
```bash
docker build -t <YOUR_ACR>.azurecr.io/<YOUR_IMAGE_NAME>:latest -f azureml_deployment/fastapi.Dockerfile . 
```

##### Push the image to ACR
```bash
az login
az acr login --name <YOUR_ACR>
docker push <YOUR_ACR>.azurecr.io/azureml/<YOUR_IMAGE_NAME>:latest
```
You should be now be able to see the image in your Azure container registry.


##### Run the image locally
```bash
docker run -p 8000:8000 -it --rm <YOUR_ACR>/azureml/<YOUR_IMAGE_NAME>:latest
```

We can also test the deployment using a sample request. We need three things for this:
1. A sample json containing the data for prediction.
2. The address of the endpoint.
3. requests library for posting the request and retrieving the response.
It is implemeted in `azureml_deployment/sample_request.py`
```bash
poetry run python azureml_deployment/sample_request.py
```

#### 4. Create an endpoint and deploy the model.
Finally, we want to deploy our image to an AzureML endpoint. This can done by using the `azure-ai-ml.MLClient`. This can be done using the MLClient from Azure's python sdk.
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
...
# Create an MLClient for your Workspace
ml_client = MLClient(
            credential = DefaultAzureCredential(), 
            subscription_id = "YOUR_SUBSCRIPTION_ID", 
            resource_group_name = "YOUR_RESOURCE_GROUP", 
            workspace_name = "YOUR_WORKSPACE_NAME"
)
```
You can find the subscription_id, resource_group_name, and workspace_name in your workspace's config.json file that you downloaded in Tutorial section 1. We can use this MLClient to create/manage endpoints and deployments.

We will perform these steps:
- Create an endpoint.
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint

endpoint = ManagedOnlineEndpoint(name="YOUR_ENDPOINT_NAME", auth_mode="key")
operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
```
- Create an environment from the fastapi docker image.
```python
from azure.ai.ml.entities import Environment

inference_config = {
    "liveness_route": {"port": 8000,"path": "/",},
    "readiness_route": {"port": 8000,"path": "/",},
    "scoring_route": {"port": 8000, "path": "/score",},
}
env = Environment(
    name="YOUR_ENVIRONTMENT_NAME",
    image="<YOUR_ACR>.azurecr.io/azureml/<YOUR_IMAGE_NAME>:latest",
    inference_config=inference_config,
)
```
- Deploy the model under the endpoint.
```python
from azure.ai.ml.entities import ManagedOnlineDeployment

deployment = ManagedOnlineDeployment(
    name="YOUR_DEPLOYMENT_NAME",
    environment=env,
    instance_type="Standard_F2s_v2",
    instance_count=1,
    endpoint_name="YOUR_ENDPOINT_NAME",
)
ml_client.online_deployments.begin_create_or_update(deployment)
```
- Route all the endpoint traffic to the new deployment.
```python
endpoint.traffic = {"YOUR_DEPLOYMENT_NAME": 100}
operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
```

The deployment is implemented in `azureml_scripts/make_deployment.py` script. Run the script by:
```bash
poetry run python azureml_scripts/make_deployment.py \
    --subscription_id <YOUR_AZURE_SUBSCRIPTION_ID> \
    --resource_group <YOUR_RESOURCE_GROUP> \
    --workspace <YOUR_WORKSPACE_NAME> \
    --endpoint_name <YOUR_ENDPOINT_NAME> \
    --deployment_name <YOUR_DEPLOYMENT_NAME> \
    --image <YOUR_ACR>.azurecr.io/azureml/<YOUR_IMAGE_NAME>:latest
```


### Conclusion
In this section of the tutorial, we can see how you can deploy your model in AzureML using a custom docker image.
The advantage of using fastapi framework is that it uses pydantic that allows specifying data types for each property and auto generates docs for the api.
