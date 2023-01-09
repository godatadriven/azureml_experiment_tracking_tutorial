from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from azureml.core import Run, Workspace
from azureml.exceptions import UserErrorException
from sklearn.inspection import DecisionBoundaryDisplay


def get_workspace() -> Workspace:
    """
    Obtain the workspace from either the current run or from the config file.
    If you are running it locally, make sure you have downloaded the config file.
    If you submitted the job to AzureML, this function will obtain the workspace from the current run.

    An exception will be raised if the workspace cannot be obtained by either method.

    :return: The AzureML workspace.
    """

    try:
        return Workspace.from_config()
    except UserErrorException:
        run = Run.get_context(allow_offline=False)
        return run.experiment.workspace


def create_decision_boundary_figure(
    model: Any,
    dataset: pd.DataFrame,
) -> plt.Figure:
    """
    Here we use the DecisionBoundaryDisplay to plot the decision boundary of the model.
    See https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html
    :param model: The trained model to plot the decision boundary for.
    :param dataset: The test data to plot the decision boundary for.
    :return: The figure containing the decision boundary plot.
    """
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    display = DecisionBoundaryDisplay.from_estimator(
        model,
        X=dataset[["x1", "x2"]],
        response_method="predict",
        ax=ax,
    )
    display.plot()
    ax.scatter(dataset["x1"], dataset["x2"], c=dataset["y"])
    ax.title.set_text("Decision boundary on test data")
    return figure
