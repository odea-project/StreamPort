"""
This module contains processing methods for machine learning data analysis.
"""

import pandas as pd
from typing import Literal
from numpy.random import Generator as NpRandomState

import shap

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing as scaler

from src.StreamPort.core import ProcessingMethod
from src.StreamPort.machine_learning.analyses import MachineLearningAnalyses
from src.StreamPort.machine_learning.analyses import IsolationForestAnalyses
from src.StreamPort.machine_learning.analyses import NearestNeighboursAnalyses


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


# class MachineLearningMethodNearestNeighboursClassifierSklearn(ProcessingMethod):###----INCOMPLETE!!---###
#     """
#     This class implements a K-Nearest Neighbors (KNN)-based classification algorithm using the sklearn library.
#     It estimates classes based on the average distance to the k-nearest neighbors.
#     """

#     def __init__(
#         self,
#         n_neighbors: int = 5,
#         contamination: float = 0.1,
#         scale_data: bool = True,
#     ):
#         super().__init__()
#         self.data_type = "MachineLearning"
#         self.method = "KNearestNeighbours"
#         self.algorithm = "Sklearn"
#         self.input_instance = dict
#         self.output_instance = dict
#         self.number_permitted = 1
#         self.parameters = {
#             "n_neighbors": n_neighbors,
#             "contamination": contamination,
#             "scale_data": scale_data,
#         }

#     def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
#         """
#         Runs KNN-based anomaly detection on the provided data.
        
#         Args:
#             analyses (MachineLearningAnalyses): The instance containing the data to be processed.
        
#         Returns:
#             NearestNeighboursAnalyses: Child class of Analyses containing processed model and outlier results.
#         """
#         data = analyses.data
#         variables = data.get("variables")

#         scaler_model = data.get("scaler_model")
#         if self.parameters["scale_data"] and scaler_model is not None:
#             variables = pd.DataFrame(
#                 scaler_model.transform(variables),
#                 columns=variables.columns,
#                 index=variables.index,
#             )

#         # Fit Nearest Neighbors
#         model = NearestNeighbors(n_neighbors=self.parameters["n_neighbors"])
#         model.fit(variables)

#         # Compute average distance to k neighbors
#         distances, _ = model.kneighbors(variables)
#         mean_distances = distances.mean(axis=1)
#         data["model_scores"] = mean_distances

#         # Threshold based on contamination
#         contamination = self.parameters["contamination"]
#         threshold = np.percentile(mean_distances, 100 * (1 - contamination))
#         labels = np.where(mean_distances > threshold, "outlier", "normal")

#         # Store results
#         data["model"] = model
#         data["outlier_labels"] = labels
#         data["threshold"] = threshold
#         data["parameters"] = self.parameters

#         # Return new analysis object with data
#         analyses = NearestNeighboursAnalyses()
#         analyses.data = data
#         return analyses


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


class MachineLearningEvaluateModelStability(ProcessingMethod):###----INCOMPLETE!!---###
    def __init__(self, 
                 test_record_path: str = None, 
                 log_folder="dev", 
                 confidence_buffer=0.1, 
                 times_classified=2, 
                 rest_indices=None,
                 test_indices=None):
        self.test_record_path = test_record_path
        self.log_folder = log_folder
        self.confidence_buffer = confidence_buffer
        self.times_classified = times_classified
        self.rest_indices = rest_indices
        self.test_indices = test_indices

    def _load_test_record(self):
        df = pd.read_csv(self.test_record_path)
        return df.sort_values("date")

    def _load_test_logs(self, test_record):
        if self.rest_indices is not None and self.test_indices is not None:
            min_required = len(self.rest_indices) // len(self.test_indices)
        else:
            min_required = 0

        if len(test_record) <= min_required:
            print("Not enough evidence of true inliers! Please run more tests for more data.")
            return None

        logs = []
        for date in test_record["date"]:
            log_path = f"{self.log_folder}/error_lc_test_{date}_classified_samples.csv"
            if os.path.exists(log_path):
                logs.append(pd.read_csv(log_path))
            else:
                print(f"No records for {date}")
        return logs

    def _get_true_class(self, group):
        outlier_count = ((group['class'] == 'outlier') & (group['confidence'] > 1 + self.confidence_buffer)).sum()
        inlier_count  = ((group['class'] == 'normal')  & (group['confidence'] < 1 - self.confidence_buffer)).sum()

        if outlier_count > self.times_classified and outlier_count > inlier_count:
            return "outlier"
        elif inlier_count > self.times_classified and inlier_count > outlier_count:
            return "normal"
        else:
            return "not set"

    def _get_confidence_variation(self, group, majority_class_series):
        majority = majority_class_series.loc[group.name]
        confidence_values = group.loc[group['class'] == majority, 'confidence']
        if len(confidence_values) == 0:
            return np.nan
        mean_conf = confidence_values.mean()
        std_conf = confidence_values.std()
        return np.nan if mean_conf == 0 or pd.isna(mean_conf) else std_conf / mean_conf

    def run(self):
        test_record = self._load_test_record()
        if test_record is None:
            return None, None

        logs = self._load_test_logs(test_record)
        if logs is None or len(logs) == 0:
            return None, None

        summary = pd.concat(logs, ignore_index=True).sort_values("index")

        # True class assignment
        class_results = summary.groupby('index').apply(self._get_true_class, include_groups=False).reset_index(name='class_true')
        summary = summary.merge(class_results, on='index', how='left')

        # Keep class_true only on first occurrence per index
        first_occurrence = ~summary.duplicated(subset='index')
        summary.loc[~first_occurrence, 'class_true'] = ""

        # Agreement analysis
        label_counts = summary.groupby(['index', 'class']).size().unstack(fill_value=0)
        max_counts = label_counts.max(axis=1)
        total_counts = label_counts.sum(axis=1)
        agreement_ratio = max_counts / total_counts

        majority_class = label_counts.idxmax(axis=1)

        # Confidence variation per index
        confidence_variation = summary.groupby('index').apply(
            lambda g: self._get_confidence_variation(g, majority_class), 
            include_groups=False
        ).fillna(10)

        confidence_consistency = 1 / (1 + confidence_variation)
        combined_stability = (agreement_ratio + confidence_consistency) / 2

        # Optional: Print diagnostic table
        true_classes = pd.concat([
            class_results.set_index("index"),
            majority_class.rename("majority_class"),
            confidence_consistency.rename("confidence_consistency")
        ], axis=1)

        print("True classes:\n", true_classes)

        # Merge stability scores into summary
        stability_df = combined_stability.reset_index(name='stability_score')
        summary = summary.merge(stability_df, on='index', how='left')

        return summary, combined_stability.mean()


# class MachineLearningMethodShapExplainer(ProcessingMethod):###----INCOMPLETE!!---###
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