"""
This module contains analyses child classes for machine learning data processing.
"""

import pandas as pd
import plotly.graph_objects as go
from src.StreamPort.core import Analyses


class MachineLearning(Analyses):
    """
    This class is a child of the Analyses class and is used
    to perform machine learning analyses on data given as a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame = None):
        """
        Initializes the MachineLearningAnalyses class with the given data.
        If no data is provided, it initializes with an empty DataFrame.

        Args:
            data: The data to be analyzed, as a pandas DataFrame.
            If None, an empty DataFrame is initialized.
        """
        super().__init__(data_type="MachineLearning", formats=[])
        self.data = data

    def __str__(self):
        """
        Returns a string representation of the MachineLearningAnalyses class.
        """
        if self.data is not None:
            str_data = (
                f"  data: {self.data.shape[0]} rows, {self.data.shape[1]} columns"
            )
        else:
            str_data = "  data: None"

        return f"\n{type(self).__name__} \n" f"{str_data} \n"

    def plot_data(self, indices: int = None) -> go.Figure:
        """
        Line plot of the data using Plotly with column names as color, index of the data DataFrame as x-axis, values as y-axis.
        """
        data = self.data
        if data is None:
            raise ValueError("No data to plot.")
        if indices is None:
            indices = data.index

        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=data[col][indices],
                    mode="lines",
                    name=col,
                )
            )

        fig.update_layout(
            xaxis_title="Analysis index",
            yaxis_title="Value / U.A.",
            legend_title="Features",
        )

        return fig
