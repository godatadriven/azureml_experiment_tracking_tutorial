import tempfile
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from logging import Logger

import requests
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential


def main() -> None:
    module = "azureml-model-deployment"
    logger = Logger(module)
    args = parse_args()

    logger.info(f"{module}: Creating Azure MLClient...")

    # Create a AzureML client
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace,
    )

    # Create an endpoint
    logger.info(f"{module}: Getting or creating endpoint...")
    endpoint = _get_or_create_endpoint(ml_client, args.endpoint_name)

    # Create a deployment environment
    logger.info(f"{module}: Creating deployment environment...")
    env = _create_deployment_env(args.image, args.env_name)

    # Create the deployment
    logger.info(f"{module}: Deploying to endpoint...")
    _create_or_update_deployment(
        ml_client,
        args.endpoint_name,
        args.deployment_name,
        env,
    )

    # Route all traffic to the deployment
    logger.info(f"{module}: Routing all traffic to current deployment...")
    endpoint = _route_all_traffic_to_deployment(
        ml_client, endpoint, args.deployment_name
    )

    # Get the API key
    logger.info(f"{module}: Getting api-key and testing endpoint...")
    api_key = ml_client.online_endpoints.get_keys(args.endpoint_name).primary_key
    # Make a request to the endpoint to test it
    url = f"https://{args.endpoint_name}.westeurope.inference.ml.azure.com"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # 'azureml-model-deployment': deployment_name # Can be used to route traffic to a specific deployment if it doesn't have 100% traffic
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    print(response.json())
    logger.info(f"{module} Success!")


def _does_endpoint_exist(ml_client: MLClient, endpoint_name: str):
    return endpoint_name in [e.name for e in ml_client.online_endpoints.list()]


def _get_or_create_endpoint(ml_client, endpoint_name):
    if not _does_endpoint_exist(ml_client, endpoint_name):
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
        operation.wait()
        assert operation.done()

    return ml_client.online_endpoints.get(endpoint_name)


def _create_or_update_deployment(
    ml_client: MLClient, endpoint_name: str, deployment_name: str, env: Environment
) -> None:
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        environment=env,
        instance_type="Standard_F2s_v2",
        instance_count=1,
        endpoint_name=endpoint_name,
    )
    operation = ml_client.online_deployments.begin_create_or_update(deployment)
    operation.wait()
    assert operation.done()


def _create_deployment_env(image: str, env_name: str) -> Environment:
    # We need to use a custom inference config else it does not work with a custom image
    # this config tells the endpoint where to send specific types of requests
    inference_config = {
        "liveness_route": {
            "port": 8000,
            "path": "/",
        },
        "readiness_route": {
            "port": 8000,
            "path": "/",
        },
        "scoring_route": {
            "port": 8000,
            "path": "/score",
        },
    }
    env = Environment(
        name=env_name,
        image=image,
        inference_config=inference_config,
    )
    return env


def _route_all_traffic_to_deployment(
    ml_client: MLClient, endpoint: ManagedOnlineEndpoint, deployment_name: str
):
    endpoint.traffic = {deployment_name: 100}
    operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
    operation.wait()

    return ml_client.online_endpoints.get(endpoint.name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="custom-deployment-env")
    return parser.parse_args()


if __name__ == "__main__":
    main()
