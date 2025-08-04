"""
This module contains analyses child classes for machine learning data processing.
"""

import umap
import pandas as pd
import numpy as np
#from datetime import datetime
import plotly.graph_objects as go
import plotly.colors
from src.StreamPort.core import Analyses


class MachineLearningAnalyses(Analyses):
    """
    This class is a child of the Analyses class and is used to perform machine learning analysis.
    """

    def __init__(self, variables: pd.DataFrame = None, metadata: pd.DataFrame = None):
        """
        Initializes the MachineLearningAnalyses class with the given data.

        Args:
            variables (pd.DataFrame): DataFrame containing the features for machine learning.
            metadata (pd.DataFrame): DataFrame containing the metadata of machine learning analyses.
        """
        super().__init__(data_type="MachineLearning", formats=[])
        self.data = {
            "variables": variables,
            "metadata": metadata,
        }

    def __str__(self):
        """
        Returns a string representation of the MachineLearningAnalyses class.
        """
        if self.data is not None:
            str_data = f"  variables: {self.data["variables"].shape[0]} rows, {self.data["variables"].shape[1]} columns"
        else:
            str_data = "  variables: None"

        if self.data["metadata"] is not None:
            str_metadata = f"\n  metadata: {self.data["metadata"].shape[0]} rows, {self.data["metadata"].shape[1]} columns"
        else:
            str_metadata = "  metadata: None"

        return f"\n{type(self).__name__} \n" f"{str_data}{str_metadata}\n"

    def plot_data(self, indices: int = None) -> go.Figure:
        """
        Plots the data using Plotly.
        Args:
            indices (int, optional): The indices of the data to plot. If None, all data is plotted.
        """
        data = self.data["variables"]
        if data is None:
            raise ValueError("No data to plot.")
        if indices is None:
            indices = data.index

        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        metadata = self.data["metadata"]
        if metadata is not None:
            text = [
                "<br>".join(f"{k}: {v}" for k, v in row.items())
                for row in metadata.loc[indices].to_dict(orient="records")
            ]
        else:
            text = None

        color_sequence = plotly.colors.qualitative.Plotly
        color_map = {
            col: color_sequence[i % len(color_sequence)]
            for i, col in enumerate(data.columns)
        }

        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=data[col][indices],
                    mode="lines+markers",
                    name=col,
                    legendgroup=col,
                    marker=dict(color=color_map[col]),
                    line=dict(color=color_map[col]),
                    text=text,
                    hovertemplate=(
                        f"<b>{col}</b><br>" + "%{{x}}<br>" + "%{{y}}<extra></extra>"
                        if text is None
                        else f"<b>%{{text}}</b><br><b>x: </b>%{{x}}<br><b>{col}: </b>%{{y}}<extra></extra>"
                    ),
                )
            )

        prediction_variables = self.data.get("prediction_variables")
        if prediction_variables is not None:

            scaler_model = self.data.get("scaler_model")
            if scaler_model is not None:
                scaled_prediction_variables = scaler_model.transform(
                    prediction_variables
                )
                prediction_variables = pd.DataFrame(
                    scaled_prediction_variables,
                    columns=prediction_variables.columns,
                    index=prediction_variables.index,
                )

            fig.add_vline(
                x=indices.max(),
                line={"color": "red", "dash": "dash"},
                annotation_text="Prediction",
                annotation_position="top left",
            )

            metadata_prediction = self.data.get("prediction_metadata")
            if metadata_prediction is not None:
                text = [
                    "<br>".join(f"{k}: {v}" for k, v in row.items())
                    for row in metadata_prediction.to_dict(orient="records")
                ]
            else:
                text = None

            for col in prediction_variables.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metadata_prediction.index + len(data),
                        y=prediction_variables[col],
                        mode="lines+markers",
                        name=col,
                        legendgroup=col,
                        showlegend=False,
                        marker=dict(color=color_map[col]),
                        line=dict(color=color_map[col]),
                        text=text,
                        hovertemplate=(
                            f"<b>{col}</b><br>" + "%{{x}}<br>" + "%{{y}}<extra></extra>"
                            if text is None
                            else f"<b>%{{text}}</b><br><b>x: </b>%{{x}}<br><b>{col}: </b>%{{y}}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            xaxis_title="Analysis index",
            yaxis_title="Value / U.A.",
            legend_title="Features",
            template="simple_white",
        )

        return fig


class IsolationForestAnalyses(MachineLearningAnalyses):
    """
    This class extends the MachineLearningAnalyses class and is used to perform Isolation Forest analysis.
    """

    def __init__(self):
        super().__init__()

    def train(self, data: pd.DataFrame = None):
        """
        Trains the Isolation Forest model on the provided data.
        The original data is replaced with the new data when given.
        When new data is not given, the original data is used.

        Args:
            data (pd.DataFrame): The input data for training.
        """
        if data is None:
            data = self.data["variables"]
        if data is None:
            raise ValueError("No data to train.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[0] == 0:
            raise ValueError("Data must have at least one row.")

        parameters = self.data["parameters"]
        if parameters is None:
            raise ValueError("No parameters to train the model.")
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")

        self.data["variables"] = data

        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.fit_transform(data)
            self.data["scaler_model"] = scaler_model
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        self.data["model"].fit(data)

    def get_training_scores(self):
        """
        Returns the training scores of the Isolation Forest model.

        Returns:
            np.ndarray: The training scores of the model.
        """

        if self.data.get("model") is None:
            return None
        if self.data.get("variables") is None:
            return None
        data = self.data.get("variables")
        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        scores = self.data["model"].decision_function(data)
        return scores

    def predict(self, data: pd.DataFrame = None, metadata: pd.DataFrame = None):
        """
        Predicts the output using the Isolation Forest model and adds the prediction to the data.

        Args:
            data (pd.DataFrame): The input data for prediction.
        """
        train_metadata = self.data.get("metadata")

        if self.data["model"] is None:
            raise ValueError("Model not trained yet.")
        if data is None:
            raise ValueError("No data to predict.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[1] != self.data["variables"].shape[1]:
            raise ValueError(
                "Data must have the same number of columns as the training data."
            )
        if data.shape[0] == 0:
            raise ValueError("Data must have at least one row.")

        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise ValueError("Metadata must be a pandas DataFrame.")
            if metadata.shape[0] != data.shape[0]:
                raise ValueError(
                    "Metadata must have the same number of rows as the data."
                )
            # safety to remove samples from test set if they already exist in train set
            if train_metadata is not None:
                in_train_set = metadata["index"].isin(train_metadata["index"])
                matching_indices = metadata.index[in_train_set].tolist()
                metadata = metadata[~in_train_set]
                
            self.data["prediction_metadata"] = metadata

        data = data.drop(index=matching_indices)

        self.data["prediction_variables"] = data

    def get_prediction_scores(self):
        """
        Returns the prediction scores of the Isolation Forest model.

        Returns:
            np.ndarray: The prediction scores of the model.
        """

        if self.data.get("prediction_variables") is None:
            return None
        if self.data.get("model") is None:
            return None
        data = self.data.get("prediction_variables")
        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        scores = self.data["model"].decision_function(data)
        return scores

    def test_prediction_outliers(self, threshold: float | str = "auto") -> pd.DataFrame:
        """
        Tests the prediction outliers using the Isolation Forest model.

        Args:
            threshold (float | str): The threshold for outlier detection. If "auto", the threshold is set to the mean
            of the training scores minus 3 times the standard deviation of the training scores.

        Returns:
            pd.DataFrame: A DataFrame containing the details of the predictions.
        """
        training_scores = self.get_training_scores()
        prediction_scores = self.get_prediction_scores()

        if prediction_scores is None:
            raise ValueError("No prediction scores to test.")

        if threshold == "auto":
            threshold = np.mean(training_scores) - 3 * np.std(
                training_scores
            )
        elif not isinstance(threshold, (int, float)):
            raise ValueError("Threshold must be a number.")
        
        prediction_metadata = self.data.get("prediction_metadata")
        outliers = pd.DataFrame(
            {
                "index": prediction_metadata["index"],
                "outlier": prediction_scores < threshold,
                "threshold" : threshold,
                "score": prediction_scores,
                "confidence" : ((prediction_scores / threshold)).round(2),
            }
        )
        outliers = self._assign_class_labels(outliers)

        return outliers
    
    def _assign_class_labels(self, 
                             outliers: pd.DataFrame = None, 
                             ):         
        """
        Assigns class labels to the current variables of the isolation forest.
        
        Args:
            outliers (pd.DateFrame): The result of test_prediction_outliers()

        Returns:
            outliers (pd.DataFrame): A dataframe of classified samples, including the confidence level of the classification
        """
        ###set classes
        if outliers is not None and isinstance(outliers, pd.DataFrame):
            prediction_metadata = self.data.get("prediction_metadata")  
            if not outliers["index"].equals(prediction_metadata["index"]):
                raise ValueError("False set of prediction results were provided. Outliers should belong to the latest prediction performed.")
            
        else:    
            outliers = self.test_prediction_outliers()
            prediction_metadata = self.data.get("prediction_metadata")  

        outliers["outlier"] = outliers["outlier"].map({True: "outlier", False: "normal"})
        outliers["class"] = outliers["outlier"]
        outliers.drop(columns=["outlier"], inplace = True)

        return outliers

    # def evaluate_classes(self, confidence_buffer : float = 0.1, times_classified : int = 2):#move to modelstability class
    #     """
    #     Evaluates and reassigns class labels in the test history of the isolation forest.
    #     In a non-deterministic model, if a sample has been declared an inlier/outlier a minimum of x times with a certain confidence >< y, it can be certified to be so. 

    #     Args:
    #         confidence_buffer (float): The buffer value from the threshold where classification result may be faulty/ambiguous.
    #         times_classified (int): The number of times a particular sample has been classified.

    #     Returns:
    #         true_classes (pd.DataFrame): A dataframe of samples that have been tested enough times and have received final class labels
    #     """
    #     test_history = self.data.get("test_history")
    #     true_classes = self.data.get("true_classes")
    #     if test_history is not None:
    #         sorted_history = dict(sorted(test_history.items()))
            
    #         summary = pd.DataFrame()
    #         for test in list(sorted_history):
    #             summary = pd.concat([summary, sorted_history[test][1]], ignore_index=True)
    #         summary = summary.sort_values("index")

    #         def get_true_classes(group):
    #             outlier_count = ((group['class'] == 'outlier') & (group['confidence'] > 1 + confidence_buffer)).sum() #1 confidence = threshold value. anomaly > 1, normal < 1
    #             inlier_count = ((group['class'] == 'normal') & (group['confidence'] < 1 - confidence_buffer)).sum() #values between 0.9 and 1.1 taken as a buffer zone of ambiguous detection accuracy

    #             if outlier_count > times_classified and outlier_count > inlier_count:
    #                 return "outlier"
    #             elif inlier_count > times_classified and inlier_count > outlier_count:
    #                 return "normal"
    #             else:
    #                 return "not set"

    #         class_results = summary.groupby('index').apply(get_true_classes, include_groups=False).reset_index(name='class_true')
    #     else:
    #         print("No tests have been recorded yet.")
        
    #     if true_classes is not None and isinstance(true_classes, pd.DataFrame):
    #         pass
    #     else:
    #         true_classes = pd.DataFrame()
        
    #     true_classes = pd.concat([true_classes, class_results])
    #     true_classes.drop_duplicates(inplace=True)
        
    #     self.data["true_classes"] = true_classes

    #     return true_classes
    
    def add_data(self, variables: pd.DataFrame = None, metadata: pd.DataFrame = None):
        """
        Adds or increments data to the Isolation Forest.

        Args:
            variables (pd.DataFrame): The input data for training.
            metadata (pd.DataFrame): The metadata for the input data.
        """
        if variables is not None:
            if not isinstance(variables, pd.DataFrame):
                raise ValueError("Variables must be a pandas DataFrame.")
            if variables.shape[1] != self.data["variables"].shape[1]:
                raise ValueError(
                    "Variables must have the same number of columns as the training data."
                )
            if variables.shape[0] == 0:
                raise ValueError("Variables must have at least one row.")

            if self.data["variables"] is None:
                self.data["variables"] = variables
            else:
                self.data["variables"] = pd.concat(
                    [self.data["variables"], variables], ignore_index=True
                )
                self.data["variables"].drop_duplicates(inplace=True)
        else:
            raise ValueError("No variables to add.")
        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise ValueError("Metadata must be a pandas DataFrame.")
            if metadata.shape[0] != variables.shape[0]:
                raise ValueError(
                    "Metadata must have the same number of rows as the variables."
                )
            if metadata.shape[1] != self.data["metadata"].shape[1]:
                raise ValueError(
                    "Metadata must have the same number of columns as the training data."
                )
            if self.data["metadata"] is None:
                self.data["metadata"] = metadata
            else:
                self.data["metadata"] = pd.concat(
                    [self.data["metadata"], metadata], ignore_index=True
                )
                self.data["metadata"].drop_duplicates(inplace=True)
        elif self.data["metadata"] is not None:
            raise ValueError("No metadata to add but required.")

        if self.data["model"] is not None:
            self.train()

    def add_prediction(
        self, add_outliers: bool = False, threshold: float | str = "auto"
    ):
        """
        Adds the prediction to the data and returns the outliers.

        Args:
            add_outliers (bool): Whether to add the outliers to the data.
            threshold (float | str): The threshold for outlier detection. If "auto", the threshold is set to the mean
            of the training scores minus 3 times the standard deviation of the training scores.
        """
        outliers = self.test_prediction_outliers(threshold)

        if outliers is None:
            raise ValueError("No outliers to add.")

        prediction_variables = self.data.get("prediction_variables")
        prediction_metadata = self.data.get("prediction_metadata")

        if prediction_variables is None:
            raise ValueError("No prediction variables to add.")

        if prediction_metadata is None:
            raise ValueError("No prediction metadata to add.")

        for i in range(len(outliers)):
            if outliers.iloc[i, outliers.columns.get_loc("class")] == "normal" or add_outliers:
                self.add_data(
                    prediction_variables.iloc[[i], :], prediction_metadata.iloc[[i], :]
                )

        for i in reversed(range(len(outliers))):
            if outliers.iloc[i, outliers.columns.get_loc("class")] == "normal" or add_outliers:
                idx = self.data["prediction_variables"].index[i]
                self.data["prediction_variables"] = self.data[
                    "prediction_variables"
                ].drop(idx)
                #self.data["prediction"] = np.delete(self.data["prediction"], i)
                if prediction_metadata is not None:
                    idx_meta = prediction_metadata.index[i]
                    self.data["prediction_metadata"] = self.data[
                        "prediction_metadata"
                    ].drop(idx_meta)

    def plot_scores(self, threshold: float | str = "auto") -> go.Figure:
        """
        Plots the scores of the Isolation Forest model using Plotly.
        """

        training_scores = self.get_training_scores()
        prediction_scores = self.get_prediction_scores()

        mt = self.data.get("metadata")
        text = None
        if mt is not None:
            text = [
                "<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items())
                for row in mt.to_dict(orient="records")
            ]

        if training_scores is None:
            raise ValueError("No training scores to plot.")

        if threshold == "auto":
            threshold = np.mean(training_scores) - 3 * np.std(training_scores)
        elif not isinstance(threshold, (int, float)):
            raise ValueError("Threshold must be a number.")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(training_scores)),
                y=training_scores,
                mode="lines+markers",
                line=dict(color="black"),
                marker=dict(color="black", size=10),
                name="Train Scores",
                legendgroup="Train",
                text=text,
                hovertemplate=text,
            )
        )

        if prediction_scores is not None:
            mt_prediction = self.data.get("prediction_metadata")
            str_text_prediction = None
            if mt_prediction is not None:
                str_text_prediction = [
                    "<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items())
                    for row in mt_prediction.to_dict(orient="records")
                ]

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(prediction_scores)) + len(training_scores),
                    y=prediction_scores,
                    mode="lines+markers",
                    line=dict(color="blue"),
                    marker=dict(color="blue", size=10),
                    name="Prediction Scores",
                    legendgroup="Prediction",
                    text=str_text_prediction,
                    hovertemplate=str_text_prediction,
                )
            )
        else:
            prediction_scores = np.array([])

        if threshold is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, len(training_scores) + len(prediction_scores) - 1],
                    y=[threshold, threshold],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Threshold",
                    legendgroup="Threshold",
                )
            )

        fig.update_layout(
            xaxis_title="Analysis index",
            yaxis_title="Score / U.A.",
            legend_title="Scores",
            template="simple_white",
        )

        return fig


class NearestNeighboursAnalyses(MachineLearningAnalyses):
    """
    This class extends the MachineLearningAnalyses class and is used to perform K - Nearest Neighbours analysis.
    """

    def __init__(self):
        super().__init__()

    def add_labels(self, data: pd.DataFrame = None, labels: list = None) -> None:
        if data is None:
            data = self.data.get("variables")
        
        metadata = self.data.get("metadata")
        
        if labels is None:
            if metadata is not None and "label" in metadata.columns:
                labels = metadata["label"]
            else:
                labels = self.data.get("labels")

        elif isinstance(labels, (np.ndarray, pd.Series, list)):
            for i in labels:
                if isinstance(i, int) and i in (0,1):
                    continue
                elif isinstance(i, str) and i in ("normal", "outlier"):
                    continue
                else:
                    raise ValueError("Class labels must be either 0 or 1, or string values 'normal' or 'outlier'")
        
        else:
            raise TypeError("Labels must be a list, NumPy array, or Pandas Series.")
        
        self.data["variables"] = data
        self.data["label"] = labels
        print("Labels have been successfully added!")
    
        return None

    def train(self, data: pd.DataFrame = None):
        """
        Trains the KNN classifier model on the provided data.
        The original data is replaced with new data when given.
        When new data is not given, the original data is used.

        Args:
            data (pd.DataFrame): The input data for training.
        """
        if data is None:
            data = self.data["variables"]
        if data is None:
            raise ValueError("No data to train.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[0] == 0:
            raise ValueError("Data must have at least one row.")

        parameters = self.data["parameters"]
        if parameters is None:
            raise ValueError("No parameters to train the model.")
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")

        self.data["variables"] = data
        self.data["labels"] = self.data.get("metadata")["label"]
        labels = self.data["labels"]

        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.fit_transform(data)
            self.data["scaler_model"] = scaler_model
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        self.data["model"].fit(data, labels)

    def get_training_labels(self):
        """
        Returns the training scores of the Isolation Forest model.

        Returns:
            np.ndarray: The training scores of the model.
        """

        if self.data.get("model") is None:
            return None
        if self.data.get("variables") is None:
            return None
        data = self.data.get("variables")
        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        scores = self.data["model"].predict(data)

        return pd.DataFrame({"index" : self.data.get("metadata")["index"],
                "label" : scores})

    def predict(self, data: pd.DataFrame = None, metadata: pd.DataFrame = None):
        """
        Predicts the output using the Isolation Forest model and adds the prediction to the data.

        Args:
            data (pd.DataFrame): The input data for prediction.
        """
        train_metadata = self.data.get("metadata")

        if self.data["model"] is None:
            raise ValueError("Model not trained yet.")
        if data is None:
            raise ValueError("No data to predict.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[1] != self.data["variables"].shape[1]:
            raise ValueError(
                "Data must have the same number of columns as the training data."
            )
        if data.shape[0] == 0:
            raise ValueError("Data must have at least one row.")

        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise ValueError("Metadata must be a pandas DataFrame.")
            if metadata.shape[0] != data.shape[0]:
                raise ValueError(
                    "Metadata must have the same number of rows as the data."
                )
            # safety to remove samples from test set if they already exist in train set
            if train_metadata is not None:
                in_train_set = metadata["index"].isin(train_metadata["index"])
                matching_indices = metadata.index[in_train_set].tolist()
                metadata = metadata[~in_train_set]
                
            self.data["prediction_metadata"] = metadata

        data = data.drop(index=matching_indices)

        self.data["prediction_variables"] = data

    def get_prediction_labels(self):
        """
        Returns the prediction scores of the KNN Classifier.

        Returns:
            pd.DataFrame: The prediction scores of the model.
        """

        if self.data.get("prediction_variables") is None:
            return None
        if self.data.get("model") is None:
            return None
        data = self.data.get("prediction_variables")
        metadata = self.data.get("prediction_metadata")
        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        scores = self.data["model"].predict(data)

        scores = pd.DataFrame(
            {
                "index" : metadata["index"],
                "label" : scores,
            }
        )

        return scores

    def get_true_labels(self, data: str = None) -> pd.DataFrame:
        """
        Returns the true labels of the KNN Classifier's test data.

        Returns:
            pd.DataFrame: The true labels of the test data.
        """
        if data == "test" or data is None:
            test_metadata = self.data.get("prediction_metadata")
            if test_metadata is None:
                raise ValueError("No test data available.")
            else:
                return test_metadata[["index", "label"]]
            
        elif data == "train":
            train_metadata = self.data.get("metadata")
            if train_metadata is None:
                raise ValueError("No training data available.")
            else:
                return train_metadata[["index", "label"]]
            
        else:
            raise ValueError("Invalid input to function. Usage: get_true_labels(['test'|'train'])")
            
    def get_prediction_probabilities(self, threshold: float | str = "auto") -> pd.DataFrame:
        """
        Tests the prediction outliers using the Isolation Forest model.

        Args:
            threshold (float | str): The threshold for outlier detection. If "auto", the threshold is set to the mean
            of the training scores minus 3 times the standard deviation of the training scores.

        Returns:
            pd.DataFrame: A two row DataFrame containing the outlier and score columns for each prediction.
        """
        if self.data.get("prediction_variables") is None:
                    return None
        if self.data.get("model") is None:
                    return None
        data = self.data.get("prediction_variables")
        metadata = self.data.get("prediction_metadata")
        scaler_model = self.data.get("scaler_model")
        if scaler_model is not None:
            scaled_data = scaler_model.transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        probs = self.data["model"].predict_proba(data)

        probs_df = pd.DataFrame(probs, columns=["normal", "outlier"], index=metadata["index"])
        
        return probs_df
    
