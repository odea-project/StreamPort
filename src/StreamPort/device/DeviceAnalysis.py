from ..core.CoreEngine import Analysis

import pandas as pd
import plotly.graph_objects as go
#plotly needs to be added to requirements.txt

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

        ***Note*** : _analysis_type must have same size as data.

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
            else:
                print("Data format must be conform")



    def plot(self, features = False):

        # Initialize traces and buttons
        traces = []          
        identifier = self.name 
        key = 'Pressure Dataframe'
        for analysis_key in self.data:
            if 'Device Pressure Analysis' in analysis_key:
                #data = (self.data[analysis_key])['Curve']
                data = self.data[key]
                time_axis = data['Time']
            if features == True : 
                data = pd.DataFrame({f"{analysis_key}_trend" : (self.data[analysis_key])['Trend'], 
                                     f"{analysis_key}_seasonal" : (self.data[analysis_key])['Seasonal'],
                                     f"{analysis_key}_residual" : (self.data[analysis_key])['Residual']}) 

                if 'Seasonal frequencies' in self.data[analysis_key]:

                    data = pd.merge([data, {f"{analysis_key}_transformed_seasonal" : pd.Series(self.data[analysis_key]['Seasonal frequencies'])}], 
                                    axis = 1)   

            # Iterate over columns (excluding the first one)
            curves = data.columns[1:]
            for sample in curves:
                # Create a scatter trace for each column
                trace = go.Scatter(x=time_axis, y=data[sample], visible=True, name=sample)
                traces.append(trace)

        # Create the layout
        layout = go.Layout(
        title="Pressure/Time Curve(s) - " + identifier, 
        xaxis=dict(title="Time(min)"),
        yaxis=dict(title="Pressure(bar)"),
        showlegend=True
        )
        #the created traces and layouts are used for the final plots

        fig = go.Figure(data=traces, layout=layout)
        fig.show() 
                    
        

    def feature_finder(self, algorithm):
        #this function returns analysis objects that are compatible with the chosen Processing Settings, for further analysis.
        if "pressure_features" or "seasonal_decomposition" in algorithm:
            for key in self.data:
                if 'Device Pressure Analysis' in key:
                    return self    
            
                else:  
                    print(f"Skipping {self.name} because its data is not a dictionary with a 'Pressure Analysis' key.")


            



