"""
This module contains processing methods for machine learning data analysis.
"""

import pandas as pd
from typing import Literal
from numpy.random import Generator as NpRandomState
import shap
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing as scaler
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
        variables = data.get("variables")

        scaler_model = data.get("scaler_model")
        if scaler_model is not None:
            scaled_variables = scaler_model.transform(variables)
            variables = pd.DataFrame(
                scaled_variables, columns=variables.columns, index=variables.index
            )

        model = IsolationForest(**self.parameters)
        model.fit(variables)
        data["model"] = model
        # data["model_scores"] = model.decision_function(variables)
        data["parameters"] = self.parameters
        analyses = IsolationForestAnalyses()
        analyses.data = data
        return analyses


class MachineLearningScaleFeaturesScalerSklearn(ProcessingMethod):
    """
    Adds a scalling

    Args:
        scaler_type (str): The type of scaler to use. Options are:
            - "MinMaxScaler"
            - "StandardScaler"
            - "RobustScaler"
            - "MaxAbsScaler"
            - "MaxNormalizer"
    """

    def __init__(self, scaler_type: str = "MinMaxScaler"):
        super().__init__()
        self.data_type = "MachineLearning"
        self.method = "ScaleFeatures"
        self.algorithm = "Sklearn"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"type": scaler_type}

    def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
        """
        Scales features of pressure curves using a scaler from sklearn.
        Args:
            analyses (PressureCurvesAnalyses): The PressureCurvesAnalyses instance to process.
        Returns:
            PressureCurvesAnalyses: The processed PressureCurvesAnalyses instance with scaled features.
        """
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        scaler_type = self.parameters["type"]
        if scaler_type == "MinMaxScaler":
            scaler_model = scaler.MinMaxScaler()
        elif scaler_type == "StandardScaler":
            scaler_model = scaler.StandardScaler()
        elif scaler_type == "RobustScaler":
            scaler_model = scaler.RobustScaler()
        elif scaler_type == "MaxAbsScaler":
            scaler_model = scaler.MaxAbsScaler()
        elif scaler_type == "MaxNormalizer":
            scaler_model = scaler.Normalizer(norm="max")
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        variables = data["variables"]
        scaled_variables = scaler_model.fit_transform(variables)
        # if hasattr(variables, "columns") and hasattr(variables, "index"):
        #     scaled_variables = pd.DataFrame(
        #         scaled_variables, columns=variables.columns, index=variables.index
        #     )
        data["scaler_model"] = scaler_model
        # data["variables"] = scaled_variables
        analyses.data = data
        return analyses


# class MachineLearningExplainModelShap(ProcessingMethod):
#     """
#     Explains the predictions of a machine learning model using SHAP (SHapley Additive exPlanations).
#     """

#     def __init__(self, model: Literal["classification", "regression"] = "regression", model_type: Literal["tree", "linear", "deep"] = "tree"):
#         super().__init__()
#         self.data_type = "MachineLearning"
#         self.method = "ExplainModel"
#         self.algorithm = "SHAP"
#         self.input_instance = dict
#         self.output_instance = dict
#         self.number_permitted = 1
#         self.parameters = {"type": model_type, "model": model}

#     def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
#         """
#         Runs the SHAP explanation on the provided data from a MachineLearning instance.
#         Args:
#             analyses (MachineLearning): The MachineLearning instance containing the data to be processed.
#         Returns:
#             MachineLearning: The processed MachineLearning instance with SHAP explanations.
#         """
#         data = analyses.data
#         if len(data) == 0:
#             print("No data to process.")
#             return analyses

#         model = data.get("model")
#         if model is None:
#             raise ValueError("No model found for explanation.")

#         #tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)
#         if self.parameters["type"] == "tree":
#             explainer = shap.TreeExplainer(model)

#         #linear models
#         #elif self.parameters["type"] == "linear":
#         #    explainer = shap.LinearExplainer(model, X_train)
        
#         # for deep learning models (Keras, PyTorch)
#         #elif self.parameters["type"] == "deep":
#         #    explainer = shap.DeepExplainer(model, X_train)

#         test_variables = data.get("prediction variables")
#         if test_variables is None:
#             raise ValueError("Pass test data to algorithm for explanation.")
        
#         if self.parameters["model"] == "classification":
#             shap_values = explainer.shap_values(test_variables)
#         elif self.parameters["model"] == "regression":
#             shap_values = explainer.expected_value(test_variables)

#         print("SHAP values calculated. Run shap.summary_plot(shap_values, test_data) to visualize them.")

#         data["shap_values"] = shap_values
#         analyses.data = data
#         return analyses