from argparse import ArgumentParser
from azureml.core import Workspace, Dataset

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
args = parser.parse_args()

workspace = Workspace.from_config()

dataset = Dataset.get_by_name(workspace, args.dataset_name)
dataset.download(args.output_folder, overwrite=True)
