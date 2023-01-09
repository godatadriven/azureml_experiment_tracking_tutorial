from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


def main() -> None:
    """
    An example script to submit a job to Azure ML that run on a custom image.
    To demonstrate that you can manage your own dependencies, we will run the same job now using poetry.
    We only need to make the following changes:
    - Tell AzureML to use the custom image
    - Change the command to use poetry
    """

    args = parse_args()
    workspace = Workspace.from_config()

    # Create the environment
    env = Environment(workspace=workspace, name="custom_image")
    # We now provide our own docker image
    env.docker.base_image = args.docker_image
    # Tell AzureML that we will manage the dependencies ourselves.
    env.python.user_managed_dependencies = True

    source_directory = Path(__file__).parent.parent
    # Some small changes to make the commands poetry compatible
    commands = [
        "poetry install -E azuremlruntime",  # Tell poetry to install the optional azuremlruntime.
        "poetry run python azureml_example/download.py --dataset_name moon_dataset --output_folder data",
        "poetry run python azureml_example/train.py --train_dataset data/train.csv --test_dataset data/test.csv",
        "poetry run python azureml_example/register.py --model_name moon_model",
    ]
    # Here we combine all the configuration
    script_run_config = ScriptRunConfig(
        source_directory=str(source_directory),
        command=" && ".join(commands),
        compute_target=args.compute_target,
        environment=env,
    )

    # Here we submit the configuration as an experiment job to AzureML.
    experiment = Experiment(workspace=workspace, name="Default")
    experiment.submit(script_run_config)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--docker_image",
        type=str,
        required=True,
        help="The docker image to use",
    )
    parser.add_argument(
        "--compute_target",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
