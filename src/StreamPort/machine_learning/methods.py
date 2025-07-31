"""
This module contains processing methods for machine learning data analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal
from numpy.random import Generator as NpRandomState

import shap

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing as scaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

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
        #model.fit(variables)
        data["model"] = model
        data["parameters"] = self.parameters
        analyses = IsolationForestAnalyses()
        analyses.data = data
        return analyses


class MachineLearningMethodNearestNeighboursClassifierSklearn(ProcessingMethod):
    """
    This class implements a K-Nearest Neighbors (KNN)-based classification algorithm using the sklearn library.
    It estimates classes based on the average distance to the k-nearest neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
    ):
        super().__init__()
        self.data_type = "MachineLearning"
        self.method = "KNearestNeighbours"
        self.algorithm = "Sklearn"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {
            "n_neighbors": n_neighbors,
            "weights": weights,
        }

    def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
        """
        Runs KNN classification on the provided data.

        Args:
            analyses (MachineLearningAnalyses): The instance containing the data to be processed.

        Returns:
            NearestNeighboursAnalyses: Child class of Analyses containing the trained model.
        """
        data = analyses.data

        variables = data.get("variables")        
        labels = data.get("metadata")["label"]           

        if variables is None or labels is None:
            raise ValueError("Both 'variables' (features) and 'labels' must be provided in data.")

        scaler_model = data.get("scaler_model")
        if scaler_model is not None:
            scaled_variables = scaler_model.transform(variables)
            variables = pd.DataFrame(
                scaled_variables,
                columns=variables.columns,
                index=variables.index,
            )

        #create the KNN classifier
        model = KNeighborsClassifier(n_neighbors=self.parameters["n_neighbors"], weights=self.parameters["weights"])
        #model.fit(variables, labels)

        #store trained model
        data["model"] = model
        data["parameters"] = self.parameters

        # Return updated analysis object
        analyses = NearestNeighboursAnalyses()
        analyses.data = data
        return analyses


class MachineLearningScaleFeaturesScalerSklearn(ProcessingMethod):
    """
    Adds a scaler model fit to the training data

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

        data["scaler_model"] = scaler_model

        analyses.data = data
        return analyses


class MachineLearningEvaluateModelStabilityNative(ProcessingMethod):
    """
    Creates a method to evaluate the internal stability of a learning model, i.e, how well it performs classification/regression over many test iterations with no parameter changes

    Args:
        test_records (pd.DataFrame): A dataframe containing the classification results of all tests conducted by an ML object. 
                                     Must-have information is class labels, classification confidence and date of test.
        confidence_buffer (float): Indicates an interval above or below the detection threshold. 
                                   Classifications with a confidence within this interval are considered a false positive or -negative.
        times_classified (int): A measure to judge the accuracy of a classification based on how many times the sample has received the same class label.
        test_size (int): The size of the test sample. This decides how many times each sample may have been tested over many iterations.  
    """
    def __init__(self,
                 test_records: pd.DataFrame = None,  
                 confidence_buffer: float = 0.1, 
                 times_classified: int = 2,
                 ):
        super().__init__()
        self.data = {}

        # if a sample has been classified atleast n times, there is enough information to make a judgement on model stability. n = 2
        if test_records is not None and isinstance(test_records, pd.DataFrame):
            self.test_records = test_records
        else:    
            raise ValueError("Insufficient test records given. Please pass a test record Dataframe.")
    
        self.confidence_buffer = confidence_buffer if isinstance(confidence_buffer, float) and 0.0 <= confidence_buffer <= 0.2 else 0.1 
        self.times_classified = times_classified if isinstance(times_classified, int) and times_classified >= 2 else 2

    def _get_true_classes(self, group):
        outlier_count = ((group['class'] == 'outlier') & (group['confidence'] > 1 + self.confidence_buffer)).sum()
        inlier_count  = ((group['class'] == 'normal')  & (group['confidence'] < 1 - self.confidence_buffer)).sum()

        if outlier_count > self.times_classified and outlier_count > inlier_count:
            return "outlier"
        elif inlier_count > self.times_classified and inlier_count > outlier_count:
            return "normal"
        else:
            return "not set"
        
    # the confidence with which a sample is classified/flagged fluctuates based on ML kernel, train size, test group, scaling.
    def _get_confidence_variation(self, group, majority_class_series): 
        majority = majority_class_series.loc[group.name]
        confidence_values = group.loc[group['class'] == majority, 'confidence']
        if len(confidence_values) == 0:
            return np.nan
        mean_conf = confidence_values.mean()
        std_conf = confidence_values.std()
        return np.nan if mean_conf == 0 or pd.isna(mean_conf) else std_conf / mean_conf

    def run(self) -> dict:# should the two private methods be standalone public methods in place of run() for better usability?
        """
        Evaluates the internal stability of a model and assigns true classes to data based on previous classification test results

        Args:
            instance attributes.

        Returns:
            dict(
                true_classes (pd.DataFrame): A dataframe containing the calculated true classes per test index.
                                            Samples that were not tested sufficiently to form a decision are assigned "not_set" or the value of the majority class until further tests are run.
                stability_score (float): The mean value of the model's combined stability.
                    combined_stability: agreement score (how many times the same sample received the same class label) 
                                    +   confidence_consistency (consistency of classification confidence values per sample = 1/(std/mean)variation in confidences)
            )  
        """
        summary = self.test_records.sort_values("index")

        # true class assignment
        class_results = summary.groupby('index').apply(self._get_true_classes, include_groups=False).reset_index(name='class_true')

        # agreement analysis: how often a sample received the same label 
        label_counts = summary.groupby(['index', 'class']).size().unstack(fill_value=0)
        max_counts = label_counts.max(axis=1)
        total_counts = label_counts.sum(axis=1)
        agreement_ratio = max_counts / total_counts

        majority_class = label_counts.idxmax(axis=1)

        # confidence variation in class_assignment per sample
        confidence_variation = summary.groupby('index').apply(
            lambda g: self._get_confidence_variation(g, majority_class), 
            include_groups=False
        ).fillna(10)

        confidence_consistency = 1 / (1 + confidence_variation)
        combined_stability = (agreement_ratio + confidence_consistency) / 2

        true_classes = pd.concat([
            class_results.set_index("index"),
            majority_class.rename("majority_class"),
            confidence_consistency.rename("confidence_consistency")
        ], axis=1)
        true_classes.reset_index(inplace=True)

        if (true_classes["class_true"] == "not_set").any():
            print("Classification Complete")
        else:
            print("Some samples are unverified. Setting true_classes to majority class")
            mask = true_classes["class_true"] == "not set"
            true_classes.loc[mask, "class_true"] = true_classes.loc[mask, "majority_class"]

        self.data["true_classes"] = true_classes
        self.data["summary"] = summary

        return {"true_classes" : true_classes, 
                "stability_score" : combined_stability.mean()}
    
    def plot_confidences(self) -> go.Figure:

        summary = self.data.get("summary")
        true_classes = self.data.get("true_classes")

        if summary is None or true_classes is None:
            self.data = self.run()
            summary = summary = self.data.get("summary")
            true_classes = self.data.get("true_classes")

        merged_df = summary.merge(true_classes, on='index', how='left')
        
        # number of normal and outlier classifications for each index
        normal_counts = merged_df[merged_df['confidence'] <= 1].groupby('index').size()
        outlier_counts = merged_df[merged_df['confidence'] > 1].groupby('index').size()

        # total number of tests per index
        total_tests = merged_df.groupby('index').size()
    
        # average confidence per index
        avg_confidence = merged_df.groupby('index')['confidence'].mean()

        # unique indices sorted in ascending order
        indices_sorted = sorted(merged_df['index'].unique())

        hovertext = [
            f"Index: {idx}<br>Times Tested: {total_tests[idx]}<br>Normal: {normal_counts.get(idx, 0)}<br>Outlier: {outlier_counts.get(idx, 0)}<br>Average Confidence: {avg_confidence[idx]:.2f}"
            for idx in indices_sorted
        ]

        colors = [
            'red' if merged_df.loc[merged_df['index'] == idx, 'class_true'].values[0] == 'outlier' else 'blue'
            for idx in indices_sorted
        ]   

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=indices_sorted,
            y=[total_tests[idx] for idx in indices_sorted],
            text=hovertext,
            width=0.6,
            hoverinfo="text",  
            marker=dict(color=colors),
            name="Total Tests per Index"
        ))

        fig.update_layout(
            title="Classification confidence over tests",
            xaxis=dict(
                title="Analysis Index",
                tickmode="array",
            ),
            yaxis=dict(
                title="Number of Tests",
                showgrid=True,
            ),
            bargap=0.9,
            template="simple_white",
            height=500,
            showlegend=False
        )

        return fig


class MachineLearningExplainModelPredictionShap(ProcessingMethod):###----INCOMPLETE!!---###
    """
    Explains the predictions based on feature importances of a machine learning model using SHAP (SHapley Additive exPlanations).
    """

    def __init__(self, model: None = None, task: Literal["classification", "anomaly_detection"] = "classification", model_type: Literal["distance", "tree", "linear", "deep"] = "distance"):
        super().__init__()
        self.data_type = "MachineLearning"
        self.method = "ExplainModel"
        self.algorithm = "SHAP"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"model" : model, "task": task, "model_type": model_type}

    def run(self, analyses: MachineLearningAnalyses) -> MachineLearningAnalyses:
        """
        Runs the SHAP explanation on the provided data from a MachineLearning instance.
        Args:
            analyses (MachineLearning): The MachineLearning instance containing the data to be processed.
        Returns:
            analyses (MachineLearning): The processed MachineLearning instance with SHAP explanations.
        """
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        model = data.get("model")
        if model is None:
            raise ValueError("No model provided for explanation.")

        # distance-based models (KNN)
        if self.parameters["model_type"] == "distance":
            explainer = shap.KernelExplainer(model)

        # tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)
        if self.parameters["model_type"] == "tree":
            explainer = shap.TreeExplainer(model)

        # linear models
        #elif self.parameters["model_type"] == "linear":
        #    explainer = shap.LinearExplainer(model, X_train)
        
        # deep learning models (Keras - Autoencoder, PyTorch)
        #elif self.parameters["model_type"] == "deep":
        #    explainer = shap.DeepExplainer(model, X_train)

        test_variables = data.get("prediction variables")
        if test_variables is None:
            raise ValueError("Pass test data to algorithm for explanation.")
        
        if self.parameters["task"] == "classification":
            shap_values = explainer.shap_values(test_variables)
        elif self.parameters["task"] == "anomaly_detection":
            shap_values = explainer.expected_value(test_variables)

        print("SHAP values calculated. Run shap.summary_plot(shap_values, test_data) to visualize them.")

        data["shap_values"] = shap_values
        analyses.data = data
        return analyses


class MachineLearningAutomateTestParameterTuningGridSearchSklearn(ProcessingMethod):
    """
    Performs automated cross-validation of multiple parameter combinations and estimates the best setup for predictions using GridSearchCV.

    Args:
        - model (sklearn): An ML model object (e.g. sklearn.neighbors.KNeighborsClassifier). 
        - scaler (sklearn): A scaler object (sklearn.preprocessing). If no scaler is provided, StandardScaler will be used by default.
        - train_data (pd.DataFrame): A DataFrame of the training data.
        - train_metadata (pd.DataFrame): A DataFrame of training metadata/labels.                         
        - parameter_grid (dict): A dict of test parameters that should be Cross-validated by GridSearchCV. If none is given, it will be estimated based on the model provided.
        - cv (int): Cross-validation folds to take for parameter tuning. Defaults is 3
        - scoring (str): Scoring metric to be applied on the model's classification results. Defaults to 'accuracy'
        - verbose (int):
        - n_jobs (int):
    """
    def __init__(self, 
                 model: None = None,
                 scaler: scaler = scaler.StandardScaler(), 
                 train_data: pd.DataFrame = None, 
                 train_metadata: pd.DataFrame = None,
                 parameter_grid: dict = None,  
                 cv: int = 3, 
                 scoring: str = 'accuracy', 
                 verbose: int = 2, 
                 n_jobs: int = -1
                 ):
        super().__init__()
        self.data_type = "MachineLearning"
        self.method = "GridSearchCV"
        self.algorithm = "Sklearn"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {
            "model" : model,
            "scaler" : scaler,
            "data" : train_data,
            "metadata" : train_metadata,
            "parameter_grid" : parameter_grid if parameter_grid is not None else {
                        'n_estimators': [50, 100, 200],
                        'max_samples': ['auto', 0.5, 0.7],
                        'contamination': [0.05, 0.1, 0.2],
                        'max_features': [0.5, 0.7, 1.0],
                        'random_state': [42]  
                        } if isinstance(model,IsolationForest) else {'n_neighbors' : list(range(1, 11))},
            "cv" : cv,
            "scoring" : scoring,
            "verbose" : verbose,
            "n_jobs" : n_jobs,
        }
    
    def run(self):
        """
        Executes GridSearchCV on the given parameters. 

        Args:
            Instance attributes.

        Returns:
            grid (dict): A dict containing the scaler used on the training data, and the GridSearchCV object fit to the parameters. 
        """
        search = GridSearchCV(
                            estimator = self.parameters["model"],
                            param_grid = self.parameters["parameter_grid"],
                            cv = self.parameters["cv"],
                            scoring = self.parameters["scoring"],
                            verbose = self.parameters["verbose"],
                            n_jobs = self.parameters["n_jobs"],
                            )
        
        scaler = self.parameters["scaler"]

        data = self.parameters["data"]
        metadata = self.parameters["metadata"]

        scaled_data = scaler.fit_transform(data)
        scaled_train = pd.DataFrame(
            scaled_data,
            columns=data.columns,
            index=data.index
        )
        ####Expand for XGBoost and further
        if isinstance(self.parameters["model"], KNeighborsClassifier):
            y = metadata["label"]

        else:
            y = None
            raise TypeError("Wrong model passed to GridSearch. Please provide an XGBoost or KNeighborsClassifier object.")

        search.fit(scaled_train,
                    y
                )

        print("Best params: ", search.best_params_)
        print("Get accuracy of best estimator on test set using: grid['grid_object'].best_estimator_.score(X_test_scaled, y_test)")
        grid = {
            "scaler" : scaler,
            "grid_object" : search
        }

        return grid