from argparse import ArgumentParser
import mlflow
from azureml.core import Workspace, Run

parser = ArgumentParser()
parser.add_argument(
    "--model_name", type=str, required=True, help="Name of the model to register."
)
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    required=False,
    help=(
        "ID AzureML has given the run."
        "You can find in the UI under raw json properties. "
        "If not provided, it will try to get it from the mlflow context."
    ),
)

args = parser.parse_args()

model_name = args.model_name
if args.run_id is not None:
    run_id = args.run_id
run_id = Run.get_context(allow_offline=False)
run_id = run_id.id

ws = Workspace.from_config()
# Set the tracking URI to the AzureML workspace
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# the artifact path is the path where the model artifact is stored within the run.
artifact_path = "model.joblib"
model_uri = f"runs:/{run_id}/{artifact_path}"
mlflow.register_model(model_uri=model_uri, name=model_name)
