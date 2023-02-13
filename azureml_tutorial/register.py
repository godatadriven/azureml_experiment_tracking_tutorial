from argparse import ArgumentParser, Namespace

import mlflow
from azureml.core import Run
from utils import get_workspace


def main() -> None:
    """
    This script is used to register run in AzureML as a model in the AzureML model store.
    After a model is registered, you easily download it from the AzureML model store using:
        model = Model(workspace, model_name)
        model.download(target_dir=".")
    :return:
    """

    args = process_args()
    model_name = args.model_name
    run_id = get_run_id(args)

    ws = get_workspace()
    # Set the tracking URI to the AzureML workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # the artifact path is the path where the model artifact is stored within the run.
    artifact_path = "model"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.register_model(model_uri=model_uri, name=model_name)


def get_run_id(args: Namespace) -> str:
    if args.run_id is not None:
        return args.run_id
    run_id = Run.get_context(allow_offline=False)
    return run_id.id


def process_args() -> Namespace:
    parser = ArgumentParser(
        description="Register a model run as a model in the AzureML model store"
    )
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
            " You can find in the UI under raw json properties. "
            "If not provided, it will try to get it from the mlflow context."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
