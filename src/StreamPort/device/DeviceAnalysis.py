from ..core.CoreEngine import Analysis

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

        feature_finder (self, features_df(DataFrame), features_list(list(str)) : add features extracted by DeviceEngine to DeviceAnalysis object.
            
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



    def plot(self, interactive = False, features = False, decomp = False, transform = False, rolling = False):
        """
        Plots analyses data based on user input. Plots pressure curves by default.
        Args:
            features: input to toggle whether feature plot should be made.
            decomp: input to toggle seasonal components of curves.
            transform: input to toggle fourier transform of raw - and corresponding seasonal - curves.
            interactive: Set interactive or not. Static plots are default, user can choose interactive by setting 'interactive = True' 
        ***Note***features, decomp and transform may only be plotted one at a time. 
        """
        #Initialize traces and buttons
        curves = {}          
        num_figs = 1
        df_keys = []
        for key in list(self.data): 
            if not 'Dataframe' in key:
                df_keys.append(key) 
 
        for analysis_key in df_keys:
            data = self.data[analysis_key]
            sample = data['Sample']
            time_axis = data['Curve']['Time']
            identifier = data['Method'] 

            if features == True:
                decomp = False
                transform = False
                rolling = False
                curves.update({sample : (data['Features'].T)}) 
                title_suffix = 'features'
                time_axis = data['Features'].index
                
            elif decomp == True :
                transform = False
                features = False
                rolling = False
                num_figs = 3
                curves.update({sample : (data['Trend'], 
                                                    data['Seasonal'], 
                                                    data['Residual'])}) 
                title_suffix = 'components'

            elif transform == True:
                features = False
                decomp = False
                rolling = False
                curves.update({sample : (data['Raw curve frequencies'], 
                                         data['Curve seasonal frequencies'])})
                num_figs =  2
                title_suffix = 'frequencies'

            elif rolling == True:
                features = False
                decomp = False
                transform = False
                num_figs = 4
                curves.update({sample : (data['Rolling statistics'].iloc[:, 0], 
                                         data['Rolling statistics'].iloc[:, 1], 
                                         data['Rolling statistics'].iloc[:, 2], 
                                         data['Rolling statistics'].iloc[:, 3])})
                title_suffix = 'rolling statistics'

            else:
                curves.update({sample : data['Curve'][sample]}) 
                title_suffix = 'Curve(s)'

              
        # Create subplots with the specified number of rows
        fig = make_subplots(rows=num_figs, cols=1, shared_xaxes=True)
        for sample_name in list(curves):
            ytext = ["Pressure (bar)"]
            xtext = "Time (min)"
            for i in range(num_figs):
                curve = curves[sample_name]
                if isinstance(curve, tuple):
                    curve = curve[i]
            
                if num_figs == 1:
                    xtext = "Features"
                elif num_figs == 2:
                    ytext = ["Amplitude(Raw)", "Amplitude(Seasonal)"]
                    xtext = "Frequencies"
                elif num_figs == 3:
                    ytext = ["Trend", "Seasonal", "Residual"]
                elif num_figs == 4:
                    ytext = ['rollmin', 'rollmax', 'rollmean', 'rollstd']
                        
                # Create a scatter trace for each column        
                trace = go.Scatter(x=time_axis, y=curve, visible=True, name=sample_name)
                fig.add_trace(trace, row=i + 1, col=1)
                fig.update_yaxes(title_text= ytext[i], row=i + 1, col=1)
                fig.update_xaxes(title_text=xtext, row=i + 1, col=1)
                
                
        # Update the overall layout
        fig.update_layout(
                        title="Pressure/Time " + title_suffix + " - " + identifier,
                        showlegend=True  # Set to True if legend must be visible
                        )

        fig.show()             
        
        

            



