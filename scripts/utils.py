
import pandas as pd
from azureml.core import Run, Workspace
from azureml.exceptions import UserErrorException
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_workspace() -> Workspace:
    """
    Obtain the workspace from either the current run or from the config file.
    If you are running it locally, make sure you have downloaded the config file.
    If you submitted the job to AzureML, this function will obtain the workspace from the current run.

    An exception will be raised if the workspace cannot be obtained by either method.

    :return: The AzureML workspace.
    """

    try:
        return Workspace.from_config()
    except UserErrorException:
        run = Run.get_context(allow_offline=False)
        return run.experiment.workspace


def create_azureml_client(
    subscription_id, resource_group, workspace, credential=DefaultAzureCredential()
):
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
    return ml_client
    