from ..core.CoreEngine import Analysis

import random
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
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

        ***Note*** : _analysis_type must have same size as data. Will influence future functionality of class methods.

        _class_label (str): 'fmt'(First Measurement), 'norm'(Normal), 'dvt'(Deviant) assigned to analyses after feature inspection.
                            This assists in future classification when using supervised learning algorithms.

    Methods: (specified are methods only belonging to child class. For superclass methods, see Analysis)

        validate (self, _analysis_type) Validates the analysis object while allowing for flexibility in handling varying datatypes for each DeviceAnalysis instance.

        plot (self, analyses(DataFrame/list(DataFrame))) : Plots the selected (pressure) curves.

            
    """



    ###
    ###ADD PROVISION TO ENABLE USER-ASSIGNED CLASS LABELS AFTER FEATURE PLOT INSPECTION. MAYBE BY CLICKING.
    ###HOVER FEATURE ON PLOTS TO BRING UP CORRESPONDING LOG PAGE(S). 
    ###



    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type=None, class_label=None):
        
        super().__init__(name, replicate, blank, data)
        self._analysis_type = str(analysis_type) if not isinstance(analysis_type, type(None)) else "Unknown"

        if self.data != {} and '001-blank' in self.data['Sample']:
            self._class_label = 'fmt'
        else: 
            self._class_label = str(class_label) if not isinstance(class_label, type(None)) else "Undefined"



    def validate(self):

        if not isinstance(self.replicate, str):
            pass
        if not isinstance(self.blank, str):
            pass

        for i in list(self.data):
            dict_list = self.data[i]     
            if isinstance(dict_list, dict) or isinstance(dict_list, list):
                pass
            else:
                print("Data format must be conform")



    def __str__(self):
        """
        Returns a string representation of the analysis object.

        """
        if self.data == {}:
            data_str = "  Empty"
        else:
            data_str = '\n'.join(   [
                                        f"{key} : (size {len(str(self.data[key]))})"
                                            if isinstance(self.data[key], int) 
                                            else 
                                        f"{key} : (size {len(self.data[key])})" 
                                            for key in self.data 
                                    ]
                                )
        
        return f"\nAnalysis\n  name: {self.name}\n  replicate: {self.replicate}\n  blank: {self.blank}\n  data:\n{data_str}\n"



    def print(self):
        print(self)
      
        

    def plot(self, interactive = True, features = False, decomp = False, transform = False, type = None, scaled = True):
        """
        Plots analyses data based on user input. Plots pressure curves by default.
        Args:
            features: input to toggle whether feature plot should be made.
            decomp: input to toggle seasonal components of curves.
            transform: input to toggle fourier transform of raw - and corresponding seasonal - curves.
            interactive: Set interactive or not. Defaults to True.
            type: Set 'box' plots or regular 'scatter' plots.
            scaled: Plot scaled version of data. Defaults to True
        ***Note***features, decomp and transform may only be plotted one at a time. 
        """
        #Initialize traces and buttons
        curves = {}          
        num_figs = 1
        feature_flag = 0
        data = self.data    
        
        time_axis = data['Curve']['Time']
        identifier = data['Method'] 
 
        samples = data['Curve'].drop(['Time'], axis=1)
        samples = samples.columns
        num_labels = len(samples)

        #distinct set of high-contrast colors selected for each of the samples
        colors_list = [f'hsv({i*360/num_labels}, 100%, 100%)' for i in range(num_labels)]
        #randomly insert black and gray to improve contrast
        colors_list.insert(random.randint(0, num_labels-1), 'black')
        colors_list.insert(random.randint(0, num_labels-1), 'gray')

        for sample in samples:
            if features == True:
                    decomp = False
                    transform = False
                    curves.update({sample : (data['Features'][sample])})
                    if scaled == True and 'Features scaled' in list(data):
                        curves.update({sample : (data['Features scaled'][sample])}) 
                        title_suffix = 'features(scaled)'
                    else:
                        title_suffix = 'features'
                    feature_flag = 1
                    time_axis = data['Features'].index
                    
            elif decomp == True :
                    transform = False
                    features = False
                    num_figs = 3
                    curves.update({sample : (data['Trend'][sample], 
                                                        data['Seasonal'][sample], 
                                                        data['Residual'][sample])}) 
                    title_suffix = 'components'

            elif transform == True:
                    features = False
                    decomp = False                   
                    curves.update({sample : (data['Raw curve frequencies'][sample], 
                                            data['Curve seasonal frequencies'][sample], 
                                            data['Curve noise frequencies'][sample])})
                    num_figs =  3
                    title_suffix = 'frequencies'

            else:
                    curves.update({sample : data['Curve'][sample]}) 
                    title_suffix = 'Curve(s)'
        
        plot_type = 0
        if type == 'box':
             plot_type = 1     
        # Create subplots with the specified number of rows
        fig = make_subplots(rows=num_figs, cols=1, shared_xaxes=True)
        for index, sample_name in enumerate(list(curves)):
            ytext = ["Pressure (bar)"]
            xtext = "Time (min)"
            for i in range(num_figs):
                curve = curves[sample_name]
                if isinstance(curve, tuple):
                    curve = curve[i]
            
                if feature_flag == 1:
                    xtext = "Features"
                elif num_figs == 3 and 'frequencies' in curve:
                    ytext = ["Amplitude(Raw)", "Amplitude(Seasonal)", "Amplitude(Residual)"]
                    xtext = "Frequencies"
                elif num_figs == 3 and not 'frequencies' in curve:
                    ytext = ["Trend", "Seasonal", "Residual"]
                    xtext = "Time (min)"

                if plot_type != 0:        
                    # Create a scatter trace for each column        
                    fig.add_trace(go.Box(x=time_axis, y=curve, visible=True, name=sample_name, marker=dict(color=colors_list[index], opacity=0.4), legendgroup=f'group{index}'), row=i + 1, col=1) 

                else:
                    fig.add_trace(go.Scatter(x=time_axis, y=curve, visible=True, name=sample_name, mode='lines',
                                    marker=dict(size=5, color=colors_list[index], line=dict(width=0)), legendgroup=f'group{index}'), row=i + 1, col=1)

                
                fig.update_yaxes(title_text= ytext[i], row=i + 1, col=1)
                fig.update_xaxes(title_text=xtext, row=i + 1, col=1)
                
                
        # Update the overall layout
        fig.update_layout(
                        title="Pressure/Time " + title_suffix + " - " + identifier,
                        showlegend=True,  # Set to True if legend must be visible
                        legend = dict(borderwidth = 0)
                        )

        if interactive == True:
            fig.show()             
        
        else:
            pio.show(fig, renderer='png')   
        

            


