from argparse import ArgumentParser, Namespace

from azureml.core import Dataset
from utils import get_workspace


def main() -> None:
    """
    This script donwloads a dataset from AzureML and saves it to a folder.
    Note: this scripts only works if you have the `azureml-dataset-runtime` package installed.
        This package is cannot be installed M1 macs.
    """

    args = parse_args()

    workspace = get_workspace()
    dataset = Dataset.get_by_name(workspace, args.dataset_name)
    dataset.download(args.output_folder)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="moon_dataset",
        help="The name of the dataset to download. The dataset must be registered in AzureML.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data",
        help="The folder to save the dataset to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
