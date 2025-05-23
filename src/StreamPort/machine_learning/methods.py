"""
This module contains processing methods for machine learning data analysis.
"""

from typing import Literal
from numpy.random import Generator as NpRandomState
from sklearn.ensemble import IsolationForest
from src.StreamPort.core import ProcessingMethod
from src.StreamPort.machine_learning.analyses import MachineLearningAnalyses
from src.StreamPort.machine_learning.analyses import IsolationForestAnalyses


class MachineLearningMethodIsolationForestSklearn(ProcessingMethod):
    """
    This class implements the Isolation Forest algorithm for anomaly detection using the sklearn library.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: float | Literal["auto"] = "auto",
        contamination: float | str = "auto",
        max_features: float = 1,
        bootstrap: bool = False,
        n_jobs: int | None = None,
        random_state: int | NpRandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        super().__init__()
        self.data_type = "MachineLearning"
        self.method = "IsolationForest"
        self.algorithm = "Sklearn"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
        }

    def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
        """
        Runs the Isolation Forest algorithm on the provided data from a MachineLearning instance.
        Args:
            analyses (MachineLearning): The MachineLearning instance containing the data to be processed.
        Returns:
            analyses (IsolationForestAnalyses): Child class of Analyses containing the processed data.
        """
        data = analyses.data
        variables = data["variables"]
        model = IsolationForest(**self.parameters)
        model.fit(variables)
        data["model"] = model
        data["model_scores"] = model.decision_function(variables)
        data["parameters"] = self.parameters
        analyses = IsolationForestAnalyses()
        analyses.data = data
        return analyses
