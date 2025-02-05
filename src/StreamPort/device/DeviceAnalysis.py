
from src.StreamPort.core.CoreEngine import Analysis

import random
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
        ****Set data as a list of Analysis child objects like pressure analysis, actuals
    
    Instance Attributes:
        _analysis_type (str/list(str), optional): Marker(s) to specify the type of data the current Analysis is related to (pressure, temperature, actuals,..)

        ***Note*** : _analysis_type must have same size as data. Will influence future functionality of class methods.

        _class (str): 'First Measurement', 'Normal', 'Deviant' assigned to analyses after feature inspection.
                            This assists in future classification when using supervised learning algorithms.

        _key_list (list(str)): list of analysis data keys in order of appearance. Maybe different for other data types.
                            
    Methods: (specified are methods only belonging to child class. For superclass methods, see Analysis)

        validate (self, _analysis_type) : Validates the analysis object while allowing for flexibility in handling varying datatypes for each DeviceAnalysis instance.

        plot (self, analyses(DataFrame/list(DataFrame))) : Plots the selected (pressure) curves.

        set_class_label (self, class_label(str)) : Sets class label of analysis object using assigned label from unsupervised learning.
            
    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type=None, class_label=None, key_list=None):
        
        super().__init__(name, replicate, blank, data)
        self._analysis_type = str(analysis_type) if not isinstance(analysis_type, type(None)) else "Unknown"
        self._key_list = key_list if not isinstance(key_list, type(None)) else list(self.data.keys())
        self._class = self.set_class_label(class_label)


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
                                        f"{key} : (size {len(str(self.data[key]))})" 
                                            for key in self.data 
                                    ]
                                )
        
        return f"\nAnalysis\n  name: {self.name}\n  replicate: {self.replicate}\n  blank: {self.blank}\n  class: {self._class}\n  data:\n{data_str}\n"



    def print(self):
        """
        Prints the current object.
        """
        print(self)
      

    
    def plot(self, interactive = True, features = False, decomp = False, transform = False, type = None, scaled = True, transpose=False):
        """
        Plots analyses data based on user input. Plots pressure curves by default.
        Args:
            features: input to toggle whether feature plot should be made.
            decomp: input to toggle seasonal components of curves.
            transform: input to toggle fourier transform of raw - and corresponding seasonal - curves.
            interactive: Set interactive or not. Defaults to True.
            type: Set 'box' plots or regular 'scatter' plots.
            scaled: Plot scaled version of data. Defaults to True
            transpose: Plot transposed dataframe. Applies only to features matrix/dataframe.
        ***Note***features, decomp and transform may only be plotted one at a time. 
        """
        #Initialize traces and buttons. num_figs and feature_flag will be dynamically adapted to desired plot type
        curves = {}          
        num_figs = 1
        feature_flag = 0
        data = self.data    
        
        #set samples and plot structure preliminarily. Absence of arguments prints pressure curves.
        time_axis = data['Curve']['Time']
        identifier = data['Method'] 
        samples = data['Curve'].drop(['Time'], axis=1)
        samples = samples.columns

        #check and set given arguments. Strictness here increases program robustness.
        if features == True:
            decomp = False
            transform = False
            features_df = data['Features']
            time_axis = features_df.index
            feature_flag = 1

            if scaled == True and 'Device Pressure Analysis' not in self.name or '_scaled' in self.name:
                title_suffix = 'features(scaled)'
                
            else:
                title_suffix = 'features'

            if transpose == True:
                samples = features_df.index
                time_axis = features_df.columns
                time_axis = [element.split('|')[-1] for element in time_axis]
                features_df = features_df.T

        elif decomp == True :
            transform = False
            features = False
            transpose = False
            scaled = False
            num_figs = 3
            title_suffix = 'components'
            feature_flag = 2

        elif transform == True:
            features = False
            decomp = False                   
            transpose = False
            scaled = False
            num_figs =  3
            title_suffix = 'frequency components'
            feature_flag = 3

        else:
            title_suffix = 'Curve(s)'
            
        num_labels = len(samples)

        #distinct set of high-contrast colors selected for each label
        colors_list = [f'hsv({i*360/num_labels}, 100%, 100%)' for i in range(num_labels)]
        #randomly insert black and gray to improve contrast
        colors_list.insert(random.randint(0, num_labels-1), 'black')
        colors_list.insert(random.randint(0, num_labels-1), 'gray')

        #populate dictionary with values to be plotted along with argument-related adaptations
        for sample in samples:
            if features == True:
                    feature_dict={sample : (features_df[sample])}
                    curves.update(feature_dict)
                    
            elif decomp == True :
                    curves.update({sample : (data['Trend'][sample], 
                                            data['Seasonal'][sample], 
                                            data['Residual'][sample])}
                                 ) 

            elif transform == True:               
                    curves.update({sample : (data['Raw curve frequencies'][sample], 
                                            data['Curve seasonal frequencies'][sample], 
                                            data['Curve noise frequencies'][sample])}
                                 )

            else:
                    curves.update({sample : data['Curve'][sample]}) 

        # Create subplots with the specified number of rows
        fig = make_subplots(rows=num_figs, cols=1, shared_xaxes=True)
        title_prefix = "Pressure/Time "
        for index, sample_name in enumerate(list(curves)):
            ytext = ["Pressure (bar)"]
            xtext = "Time (min)"
            for i in range(num_figs):
                curve = curves[sample_name]
                if isinstance(curve, tuple):
                    curve = curve[i]
            
                if feature_flag == 1:
                    xtext = "Features"
                    ytext = ["Values"]
                    if scaled == True:
                         ytext = ["Values(scaled)"]
                    if transpose == True:
                         xtext = "Samples"

                elif feature_flag == 3:
                    #frequency bins corresponding to fft results
                    import numpy as np
                    time_axis = np.fft.fftfreq(len(curve))
                    title_prefix = "Magnitudes of "
                    xtext = "Frequency(Hz)"
                    ytext = ["Amp(Raw)", "Amp(Snl)", "Amp(Rsd)"]

                elif feature_flag == 2:
                    ytext = ["Trend", "Seasonal", "Residual"]
                    xtext = "Time (min)"

                if feature_flag==1 and transpose==True:
                    labelname = sample_name.split('|')[-1]
                else:
                    labelname = (sample_name.split('|')[-1]).split(' ')[-2]

                if type == 'box':        
                    # Create a scatter trace for each column        
                    fig.add_trace(go.Box(x=time_axis, y=curve, visible=True, name=labelname, marker=dict(color=colors_list[index], opacity=0.4), hovertext=sample_name, legendgroup=f'group{index}'), row=i + 1, col=1) 

                elif type == 'bar':
                    fig.add_trace(go.Bar(x=time_axis, y=curve, text=round(curve, 2), textposition='auto', visible=True, name=labelname, marker=dict(color=colors_list[index], opacity=0.4), hovertext=sample_name, legendgroup=f'group{index}'), row=i + 1, col=1)

                else:
                    fig.add_trace(go.Scatter(x=time_axis, y=curve, visible=True, name=labelname, mode='lines',
                                    marker=dict(size=5, color=colors_list[index], line=dict(width=0)), text=sample_name, legendgroup=f'group{index}'), row=i + 1, col=1)
                    
                
                fig.update_yaxes(title_text= ytext[i], row=i + 1, col=1)
                fig.update_xaxes(title_text=xtext, row=i + 1, col=1)
                

        # Update the overall layout
        fig.update_layout(
                        title= title_prefix + title_suffix + " - " + identifier,
                        showlegend=True,  # Set to True if legend must be visible
                        legend = dict(borderwidth = 0)
                        )

        if interactive == True:
            fig.show()             
        
        else:
            pio.show(fig, renderer='png')   

        #if making a fullscreen plot
        #fig.write_html('plot.html')

        fig = None

        

    def set_class_label(self, class_label=None):
        """
        Self_assign class labels input from DeviceEngine's classify() function.

        """
        if not isinstance(class_label, type(None)) and not '001-blank' in self.data.get('Sample', []):
            self._class = str(class_label)

        elif isinstance(class_label, list) and isinstance(class_label[0], str):
            self._class = class_label[0]

        else:
            if self.data != {} and 'Sample' in self.data and '001-blank' in self.data['Sample']:
                self._class = 'First measurement'             
            else:
                self._class = "Undefined"
        

"""
***********COMING SOON************
"""

class DevicePressureAnalysis(DeviceAnalysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type='pressure', class_label=None, key_list=None):
        super().__init__(name, replicate, blank, data, analysis_type, class_label, key_list)
        self._analysis_type = analysis_type
        self._key_list = key_list if not isinstance(key_list, type(None)) else list(self.data.keys())



class DeviceActuals(DeviceAnalysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type='actuals', class_label=None, key_list=None):
        super().__init__(name, replicate, blank, data, analysis_type, class_label, key_list)
        self._analysis_type = analysis_type



class DeviceMetadata(DeviceAnalysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None, analysis_type='metadata', class_label=None, key_list=None):
        super().__init__(name, replicate, blank, data, analysis_type, class_label, key_list)
        self._analysis_type = analysis_type


