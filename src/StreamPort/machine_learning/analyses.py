"""
This module contains analyses child classes for machine learning data processing.
"""

import pandas as pd
import plotly.graph_objects as go
from src.StreamPort.core import Analyses


class MachineLearning(Analyses):
    """
    This class is a child of the Analyses class and is used to perform machine learning analysis.
    """

    def __init__(self, variables: pd.DataFrame = None, metadata: pd.DataFrame = None):
        """
        Initializes the MachineLearningAnalyses class with the given data.

        Args:
            features (pd.DataFrame): DataFrame containing the features for machine learning.
            metadata (pd.DataFrame): DataFrame containing the metadata of .
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

        metadata = self.data["metadata"]
        if metadata is not None:
            text = [
                "<br>".join(f"{k}: {v}" for k, v in row.items())
                for row in metadata.loc[indices].to_dict(orient="records")
            ]
        else:
            text = None

        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=data[col][indices],
                    mode="lines",
                    name=col,
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
