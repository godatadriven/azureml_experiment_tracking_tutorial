from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.core import Model, Workspace


def main() -> None:
    args = process_args()

    ws = Workspace.from_config()

    name, version = args.model.split(":")
    model = Model(name=name, version=version, workspace=ws)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.download(target_dir=output_dir, exist_ok=True)


def process_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to download. Expected format: <model_name>:<version>",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
