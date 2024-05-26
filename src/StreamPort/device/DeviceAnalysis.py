from ..core.CoreEngine import Analysis

import plotly.express as px

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

        validate(): Validates the analysis object while allowing for flexibility in handling varying datatypes for each DeviceAnalysis instance.

            
    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, anatype=None):
        
        super().__init__(name, replicate, blank, data)

        self._anatype = str(anatype) if not isinstance(anatype, type(None)) else "Unknown"


    def validate(self):

        if not isinstance(self.replicate, str):
            pass

        if not isinstance(self.blank, str):
            pass           


    def plot(self):

        for i in self.data:
            print(i)
            print(self.data[i])


