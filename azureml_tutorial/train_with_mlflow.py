import tempfile
from argparse import ArgumentParser, Namespace
from typing import Any

import mlflow
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from utils import create_decision_boundary_figure, get_workspace


def main() -> None:
    """
    This script trains a random forest classifier and preforms some hyperparameter tuning.
    The results (metrics and model) are logged to AzureML using mlflow.
    The expected data format is a csv file with the following columns:
        x1: float: The first feature.
        x2: float: The second feature.
        y: int: The target class.
    """

    args = parse_args()

    ws = get_workspace()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    df_train = pd.read_csv(args.train_dataset)
    df_test = pd.read_csv(args.test_dataset)

    # We define the hyperparameters we want to tune
    param_grid = {
        "n_estimators": args.n_estimators,
        "criterion": args.criterion,
        "max_depth": args.max_depth,
    }
    # We log the selected hyper-parameters to azureml using mlflow
    # You can find the best hyper-parameters in the azureml UI under parameters.
    for param, value in param_grid.items():
        mlflow.log_param(f"gridsearch/{param}", str(value))

    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=args.n_cross_vals, n_jobs=-1)

    # We train the model
    grid_search.fit(df_train[["x1", "x2"]], df_train["y"])
    model = grid_search.best_estimator_

    # Here we evaluate the model
    predictions = model.predict(df_test[["x1", "x2"]])
    accuracy = accuracy_score(df_test["y"], predictions)

    # We log the accuracy to azureml using mlflow
    # You can see the logged metrics in the azureml UI under the "Metrics" tab
    mlflow.log_metric("accuracy", accuracy)

    # We log the selected hyper-parameters to azureml using mlflow
    # You can find the best hyper-parameters in the azureml UI under parameters.
    for k, v in grid_search.best_params_.items():
        mlflow.log_param(f"selected/{k}", v)

    # We log the decision boundary to azureml using mlflow as an image artifact.
    # You can find the image in the azureml UI under images.
    figure = create_decision_boundary_figure(model, df_test)
    mlflow.log_figure(figure, "decision_boundary.png")

    # Export the model and log it to azureml using mlflow
    _export_model(model)


def _export_model(model: Any) -> None:

    # We use a tmp dir here because we keep the model saved locally.
    # However, `log_artifact` needs a path to a file stored on disk.
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump(model, f"{tmp_dir}/model.joblib")
        mlflow.log_artifact(f"{tmp_dir}/model.joblib")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/train.csv")
    parser.add_argument("--test_dataset", type=str, default="data/test.csv")
    parser.add_argument("--n_cross_vals", type=int, default=5)
    parser.add_argument("--max_depth", default=[2, 5, 10, None])
    parser.add_argument("--n_estimators", default=[10, 25, 100])
    parser.add_argument("--criterion", default=["gini", "entropy"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
