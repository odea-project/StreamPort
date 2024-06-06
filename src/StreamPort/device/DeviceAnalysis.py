from ..core.CoreEngine import Analysis

import plotly.graph_objects as go
#plotly needs to be added to requirements.txt

import pandas as pd

class DeviceAnalysis(Analysis):

    """
    Represents a DeviceAnalysis object, inherited from Analysis Class.

    Class Attributes:
        name (str): The name of the analysis. Uniquely identified using the method name and the date of creation.
        replicate (str): The name of the replicate.
        blank (str): The name of the blank.
        data (dict/list): The data of the analysis, which is a dict or list of one dimension numpy arrays or dicts or lists.

    Instance Attributes:
        _analysis_type (str/list(str), optional): Marker(s) to specify the type of data the current Analysis is related to (pressure, temperature, ..)

        ***Note*** : _anatype must have same size as data.

    Methods: (specified are methods only belonging to child class. For superclass methods, see Analysis)

        validate (self, _analysis_type) Validates the analysis object while allowing for flexibility in handling varying datatypes for each DeviceAnalysis instance.

        plot (self, analyses(DataFrame/list(DataFrame))) : Plots the selected (pressure) curves.

        get_features (self, features_df(DataFrame), features_list(list(str)) : add features extracted by DeviceEngine to DeviceAnalysis object.
            
    """


    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type=None):
        
        super().__init__(name, replicate, blank, data)
        self._analysis_type = str(analysis_type) if not isinstance(analysis_type, type(None)) else "Unknown"



    def validate(self):

        if not isinstance(self.replicate, str):
            pass
        if not isinstance(self.blank, str):
            pass

        for i in self.data:
            dict_list = self.data[i]     
            if isinstance(dict_list, dict) or isinstance(dict_list, list):
                pass



    def plot(self):

        # Initialize traces and buttons
        traces = []

        identifier = self.name

        key = 'Pressure Dataframe'

        # Iterate over columns (excluding the first one)
        data = self.data[key]
        curves = data.columns[1:]
        time_axis = data.columns[0]

        for sample in curves:
            # Create a scatter trace for each column
            trace = go.Scatter(x=data[time_axis], y=data[sample], visible=True, mode = 'markers', name=sample)
            traces.append(trace)

        # Create the layout
        layout = go.Layout(
        title="Pressure/" + identifier, 
        xaxis=dict(title="Time(min)"),
        yaxis=dict(title="Pressure(bar)"),
        showlegend=True
        )
        #the created traces and layouts are used for the final plots

        fig = go.Figure(data=traces, layout=layout)
        fig.show() 
                    
        

    def get_features(self, features_df, features_list):

        if not isinstance(features_df, type(None)):

            extracted_features = features_df.iloc[ : , 1 : ].agg(features_list)

        else:

            print("No data was provided!")

        sample_names = []
        runtime = pd.DataFrame()
        runtype = pd.DataFrame()

        for d in self.data:

            sample_names.append(self.data[d]['Sample'])

            if self.data[d]['Method'] in d:
                runtime = pd.concat([runtime, pd.Series(self.data[d]['Runtime'])], 
                                    axis = 1)            

                if 'blank' in self.data[d]['Sample'] :

                    runtype = pd.concat([runtype, pd.Series(0)], 
                                        axis = 1)
                
                else:

                    runtype = pd.concat([runtype, pd.Series(1)], 
                                        axis = 1)
                    
        runtime.columns = sample_names            
        runtime.name = "Runtime"

        runtype.columns = sample_names        
        runtype.name = "Runtype"

        extracted_features = pd.concat([extracted_features, runtime.astype(str)], axis = 0)

        extracted_features = pd.concat([extracted_features, runtype.astype(int)], axis = 0)

        print(extracted_features.T)
        return extracted_features