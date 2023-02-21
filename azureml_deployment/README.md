## Tutorial section 2: 
## Azureml fastapi online endpoint
This section will focus on creating a webapp to serve the model using the fastapi framework.
Firstly, we will download the artifacts of our registered model from AzureML cloud. 
Next, we will create a docker container registry for sercving the model in fastapi.
Finally, we will create an endpoint in AzureML and deploy our custom docker image to the endpoint.


### Download the model
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

### Create the fastapi app
Fastapi is a framework to build RESTapis in python.
```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
...
# Initialize the app
app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(content = {"message": "Hello world!"})
```
This code above is a basic example of a fastapi webapp that displays a message when it is run. The fastapi app for serving the model is scripted in `azureml_deployment/src/main.py`.

You can run the server locally by:
```bash
uvicorn --host 0.0.0.0 --port 8000 azureml_deployment.src.main:app
```

We can also test the server using a post request with mock data. To test the server:
```bash
poetry run python azureml_deployment/src/test.py
```


### Build the docker image in Azure container registry (ACR)
Now that we have a working fastapi app, we will create a docker image from `fastapi.Dockerfile`.
```bash
docker build -t <YOUR_ACR>.azurecr.io/<YOUR_IMAGE_NAME>:latest -f fastapi.Dockerfile . 
```

### Push the image to ACR
```bash
az login
az acr login --name <YOUR_ACR>
docker push <YOUR_ACR>.azurecr.io/azureml/<YOUR_IMAGE_NAME>:latest
```
You should be now be able to see the image in your Azure container registry.


### Run the image locally
```bash
docker run -p 8000:8000 -it --rm <YOUR_ACR>/azureml/<YOUR_IMAGE_NAME>:latest
```
To test the docker server, you can again run:
```bash
poetry run python azureml_deployment/src/test.py
```

### Deployment
Finally, we want to deploy our image to an AzureML endpoint. This can done by using the `azure-ai-ml.MLClient`.
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint
)
...
# Create an MLClient for your Workspace
ml_client = MLClient("YOUR_CREDENTIAL", "YOUR_SUBSCRIPTION_ID", "YOUR_RESOURCE_GROUP", "YOUR_WORKSPACE_NAME")

# Create the endpoint
endpoint = ManagedOnlineEndpoint(name="YOUR_ENDPOINT_NAME", auth_mode="key")
operation = ml_client.online_endpoints.begin_create_or_update(endpoint)

# Create the deployment environment
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

# Create the deployment
deployment = ManagedOnlineDeployment(
    name="YOUR_DEPLOYMENT_NAME",
    environment=env,
    instance_type="Standard_F2s_v2",
    instance_count=1,
    endpoint_name="YOUR_ENDPOINT_NAME",
)

# Route the traffic to your deployment
endpoint.traffic = {"YOUR_DEPLOYMENT_NAME": 100}
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint)
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
In this section of the tutorial, we can see how you can deploy your model in AzureML using custom docker image.
The advantage of using fastapi framework is that it uses pydantic that allows specifying data types for each property and auto generates docs for the api.
