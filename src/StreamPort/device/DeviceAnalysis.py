from ..core.CoreEngine import Analysis

import plotly.graph_objects as go
import pandas as pd

class DeviceAnalysis(Analysis):

    """
    Represents a DeviceAnalysis object, inherited from Analysis Class.

    Class Attributes:
        name (str): The name of the analysis. Uniquely identified using the method name and the date of creation.
        replicate (str): The name of the replicate.
        blank (str): The name of the blank.
        data (dict): The data of the analysis, which is a dict of one dimension numpy arrays.

    Instance Attributes:
        _anatype (str/list(str), optional): Marker(s) to specify the type of data the current Analysis is related to (pressure, temperature, ..)

        ***Note*** : _anatype must have same size as data.

    Methods: (specified are methods only belonging to child class. For superclass methods, see Analysis)

        validate(self): Validates the analysis object while allowing for flexibility in handling varying datatypes for each DeviceAnalysis instance.

        plot(self, analyses(DataFrame/list(DataFrame))) : Plots the selected (pressure) curves
            
    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, anatype=None):
        
        super().__init__(name, replicate, blank, data)

        self._anatype = str(anatype) if not isinstance(anatype, type(None)) else "Unknown"


    def validate(self):

        if not isinstance(self.replicate, str):
            pass

        if not isinstance(self.blank, str):
            pass           


    def plot(self, analyses):

        def make_plot(data):
            
            # Initialize traces and buttons
            traces = []

            x_axis = data.columns[0]
            # Iterate over columns (excluding the first one)
            for col in data.columns[1:]:
                # Create a scatter trace for each column
                trace = go.Scatter(x=data.iloc[:, 0], y=data[col], visible=True, name=col)
                traces.append(trace)

            # Create the layout
            layout = go.Layout(
                title="Pressure/" + x_axis,
                xaxis=dict(title="Time(min)"),
                yaxis=dict(title="Pressure(bar)"),
                showlegend=True
            )
            # Now you can use 'traces' and 'layout' to create your plot
            # (e.g., using plotly.offline.plot or plotly.io.show)

            fig = go.Figure(data=traces, layout=layout)
            fig.show() 


        if isinstance(analyses, list):
            for i in analyses:
                make_plot(i)

        else:
            make_plot(analyses)
        