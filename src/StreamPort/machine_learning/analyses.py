"""
This module contains analyses child classes for machine learning data processing.
"""

import umap
import pandas as pd
import numpy as np
from timeit import timeit
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

    def __init__(self, evaluator):
        super().__init__()
        self.results = {}
        self.evaluator = evaluator
        self.evaluation_object = None

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

        self.data["train_time"] = timeit(lambda: self.data["model"].fit(data), number=1)

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
    
    def _get_num_tests(self, scaling_const: int = 100, offset: int = 5, n_min: int = 3, n_max: int = 50):
        """
        Dynamically calculates the number of tests n_tests required based on the size of the training set.

        Args:
            - scaling_const: Constant that adjusts the rate of decrease of n_tests with respect to train set size. Higher scaling_const means a slower decrease
            - offset: Offset to ensure n_tests does not fall below a certain minimum
            - n_min: Minimum number of tests to run
            - n_max: Maximum number of tests
        
        Returns:
            - n_tests: The computed appropriate number of tests to perform
        """
        try:
            train_size = len(self.data.get("metadata"))
        except TypeError:
            train_size = 0

        if train_size == 0:
            raise ValueError("No train data available.")
        
        # calculate n using a log scale. train_size + 1 avoids the log of 0 or a very small number
        n_tests = scaling_const / (np.log(train_size + 1)) + offset

        # ensure n is within the defined bounds
        n_tests = max(n_min, min(n_max, int(n_tests)))
        
        return n_tests

    def test_prediction_outliers(self, threshold: float | str = "auto", n_tests: int = None, show_scores : bool = False) -> pd.DataFrame:
        """
        Tests the prediction outliers using the Isolation Forest model.

        Args:
            threshold (float | str): The threshold for outlier detection. If "auto", the threshold is set to the mean
                                    of the training scores minus 3 times the standard deviation of the training scores.
            n_tests (int): The number of times a sample should be scored using decision_function(). This mitigates poor model generalization with small datasets. 
                                    If None, defaults to a dynamically assigned value based on the train size (see _get_num_tests()).
            show_scores (bool): If True, the method plots the anomaly scores for each of the [n_tests] test runs.

        Returns:
            outliers (pd.DataFrame): A DataFrame containing the details of the predictions, including the confidence level of the classification. 
        """  
        if n_tests is None or n_tests == 0:
            n_tests = self._get_num_tests()

        prediction_metadata = self.data.get("prediction_metadata")

        index = prediction_metadata["index"].iloc[0]
        batch_position = prediction_metadata["batch_position"].iloc[0]

        for i in range(n_tests):
            
            training_scores = self.get_training_scores()
            if training_scores is None:
                raise ValueError("No training scores available.")

            if threshold == "auto":
                threshold = np.mean(training_scores) - 3 * np.std(
                    training_scores
                )
            elif not isinstance(threshold, (int, float)):
                raise ValueError("Threshold must be a number.")

            prediction_scores = self.get_prediction_scores()
            if prediction_scores is None:
                raise ValueError("No prediction scores to test.")

            outliers = pd.DataFrame(
                {
                    "index": index,
                    "batch_position": batch_position,
                    "outlier": prediction_scores < threshold,
                    "train_size" : len(training_scores),
                    "train_time" : self.data["train_time"],
                    "threshold" : threshold,
                    "score": prediction_scores,
                    "confidence" : abs((prediction_scores / threshold)).round(2),
                }
            )
            outliers = self._assign_class_labels(outliers)

            if show_scores == True:
                if n_tests == 1:
                    print("Only one test was run. plot scores using analyses.plot_scores()")
                else:
                    score_plot = self.plot_scores()
                    score_plot.update_layout(title = f"Test run {i + 1}")
                    score_plot.show()

            self.results[f"{prediction_metadata["index"].iloc[0]}_{i+1}"] = outliers
            if n_tests > 1 and i != n_tests - 1: # no need to train after last test since add_data already calls it
                self.train()

        if n_tests > 1:
            result_list = []
            for key, data in self.results.items():
                temp = data.copy()
                temp["test_number"] = key 
                result_list.append(temp)

            test_records = pd.concat(result_list, axis=0, ignore_index=True) 
            
            # run ModelEvaluator
            evaluator = self.evaluator(test_records=test_records)
            evaluator.run()
            self.evaluation_object = evaluator

            true_classes = evaluator.get_true_classes()
            
            outliers["class"] = true_classes.set_index("index").loc[outliers["index"].iloc[0], "class_true"]

        return outliers
    
    def _assign_class_labels(self, 
                             outliers: pd.DataFrame = None, 
                             ):         
        """
        Assigns class labels to the current variables of the isolation forest.
        
        Args:
            outliers (pd.DateFrame): The result of test_prediction_outliers()

        Returns:
            outliers (pd.DataFrame): A dataframe of classified samples
        """
        if not isinstance(outliers, pd.DataFrame):
            outliers = self.test_prediction_outliers()  
            
        else:    
            prediction_metadata = self.data.get("prediction_metadata")  
            if not outliers["index"].equals(prediction_metadata["index"]):
                raise ValueError("False set of prediction results were provided. Outliers should belong to the latest prediction performed.")

        outliers["class"] = outliers["outlier"].map({True: "outlier", False: "normal"})
        outliers.drop(columns="outlier", inplace = True)

        return outliers
    
    def get_results(self, indices: int | list = None, summarize: bool = False) -> pd.DataFrame | None:
        """
        Returns the IsolationForestAnalyses results.

        Args:
            indices (int | list(int), optional): The sample(s) identified by index/indices for which the results should be retrieved. If None, returns all results.
            summarize (bool, optional): If True, prints a summary of the records for each unique index instead.
        Returns:
            results (pd.DataFrame | None): A DataFrame containing records of the predictions of this model or prints a stringified summary of the records. Grouped by unique index.
        """
        results = self.results

        if indices is None:
            indices = []

        elif isinstance(indices, list):
            for idx in indices:
                if not isinstance(idx, int):
                    raise ValueError("Indices must be an int or a list of ints.")
                                            
        elif isinstance(indices, int):
            indices = [indices]

        else:
            raise ValueError("Indices must be an int or a list of ints.")

        filtered_results = {}
        for key in list(results.keys()):
            if int(key.split("_")[0]) in indices:
                filtered_results[key] = results[key]

        if filtered_results:
            results = filtered_results

        results = {key : results[key] for key in sorted(results.keys(), key = lambda k: ( int(k.split("_")[0]), int(k.split("_")[-1]) ))}

        if not results:
            return None
        else:
            results_df = pd.concat(results.values(), ignore_index=True)
            results_df["test_number"] = [key.split("_")[-1] for key in results.keys() for i in range(len(results[key]))]

        if summarize:    
            group_stats = results_df.groupby("index").agg(
                train_size = ("train_size", "mean"),
                normal_count = ("class", lambda x: (x == "normal").sum()),
                outlier_count = ("class", lambda x: (x == "outlier").sum()),
                num_tests = ("test_number", "count")
            ).reset_index()

            summary = ""

            for row in group_stats.itertuples(index=False):
                summary += "Index: " + str(row.index) + "\n"
                summary += "Train size: " + str(int(row.train_size)) + "\n"
                summary += "Num. Tests: " + str(row.num_tests) + "\n" 
                summary += "Normal: " + str(row.normal_count) + "\n" 
                summary += "Outlier: " + str(row.outlier_count) + "\n" 
                summary += "\n"

            print(summary)
            return None

        return results_df
    
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
        result = self.evaluation_object
        if result is None:
            outliers = self.test_prediction_outliers()
        else:
            class_data = result.get_true_classes()
            outliers = class_data.copy()
            outliers.rename(columns={"class_true" : "class"}, inplace = True)
            
        if outliers is None:
            raise ValueError("No outliers to add.")

        prediction_variables = self.data.get("prediction_variables")
        prediction_metadata = self.data.get("prediction_metadata")

        idx = prediction_metadata["index"].iloc[0]
        outliers = outliers[outliers["index"] == idx]
        # only add indices from outliers that are present in prediction_variables and metadata

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
    
    def plot_confidence_variation(self):
        """
        Plot the variation in detection confidence over n runs of a single test sample
        """
        evaluator = self.evaluation_object
        if evaluator is None:
            self.test_prediction_outliers()
            evaluator = self.evaluation_object

        confidence_plot = evaluator.plot_confidence_variation()
        return confidence_plot
    
    def plot_threshold_variation(self):
        """
        Plot the variation in detection threshold w.r.t train_size
        """
        evaluator = self.evaluation_object
        if evaluator is None:
            self.test_prediction_outliers()
            evaluator = self.evaluation_object
        
        threshold_plot = evaluator.plot_threshold_variation()
        return threshold_plot
    
    def plot_train_time(self):
        """
        Plot the variation in model train time w.r.t train size
        """
        evaluator = self.evaluation_object
        if evaluator is None:
            self.test_prediction_outliers()
            evaluator = self.evaluation_object

        train_time_plot = evaluator.plot_train_time()
        return train_time_plot
    
    def plot_model_stability(self):
        """
        Plot the variation in model stability w.r.t the change in average classification/detection confidence and n_tests requirement over change in train size
        """
        evaluator = self.evaluation_object
        if evaluator is None:
            self.test_prediction_outliers()
            evaluator = self.evaluation_object

        stability_plot = evaluator.plot_model_stability()
        return stability_plot
    

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
    
