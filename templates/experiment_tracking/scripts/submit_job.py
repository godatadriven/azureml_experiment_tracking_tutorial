from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


def main() -> None:
    """
    An example script to submit a job to Azure ML.
    In this script, we build the environment based on a requirements.txt file.
    This way AzureML takes care of building the environment (docker image) for us.
    """

    args = parse_args()
    workspace = Workspace.from_config()

    # We need to know where the root directory of the repo is.
    # We know where it is relative to this file.
    # So, we can use the following trick to get the path to the repo root.
    repo_root = Path(__file__).parent.parent

    # here we obtain the path to the requirements.txt file
    # we know where it is relative to the repo root directory
    requirements_path = repo_root / "requirements.txt"
    # resolve makes a relative path absolute
    abs_requirements_path = requirements_path.resolve()
    # Create the environment
    env = Environment.from_pip_requirements(
        name="moon_model_env",
        file_path=str(abs_requirements_path),
    )
    # Python version must be added in this unclear way
    env.python.conda_dependencies.set_python_version("3.8")

    # Here give the path to source directory.
    # This directory will be copied to AzoureML and be mounted in environment.
    source_directory = repo_root / "azureml_tracking"
    # After the source directory is mounted, AzureML will execute the following bash command:
    commands = [
        "python download_dataset.py --dataset_name moon_dataset --output_folder data",
        "python train_with_mlflow.py --train_dataset data/train.csv --test_dataset data/test.csv",
        "python register_model.py --model_name moon_model",
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
    print("You follow the experiment here:")
    print(experiment.get_portal_url())


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--compute_target",
        type=str,
        required=True,
        help=(
            "The name of the compute target to use."
            " You can find under AzureML > compute > compute clusters"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
