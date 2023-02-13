from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.core import Model, Workspace


def main() -> None:
    """
    A example script to download a model from Azure ML.
    """

    args = process_args()

    ws = Workspace.from_config()

    name, version = args.model.split(":")
    model = Model(name=name, version=version, workspace=ws)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.download(target_dir=output_dir / f"{version}")


def process_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to download. Expected format: <model_name>:<version>",
    )

    default_output = Path(__file__).parent / ".." / "model" / "versions"
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="Output directory for the model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
