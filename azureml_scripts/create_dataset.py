import tempfile

import pandas as pd
from azureml.core import Dataset, Workspace
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def main() -> None:
    """
    This scripts generates a dataset and uploads it to Azure ML.
    """

    # Generate a train and test dataset.
    X, y = make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=42)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    workspace = Workspace.from_config()
    datastore = workspace.get_default_datastore()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # We can only upload from disk, so we save it to a temporary directory.
        df_train.to_csv(f"{tmp_dir}/train.csv", index=False)
        df_test.to_csv(f"{tmp_dir}/test.csv", index=False)

        # Upload the datasets to the default datastore (blobstore container)
        # In this container it puts it inside the "datasets/moons" folder
        blob_store_path = "datasets/moons"
        Dataset.File.upload_directory(tmp_dir, (datastore, blob_store_path))

        dataset = Dataset.File.from_files(path=[(datastore, blob_store_path)])
        dataset.register(workspace, "moon_dataset")


if __name__ == "__main__":
    main()
