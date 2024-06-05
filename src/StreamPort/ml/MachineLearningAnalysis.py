from ..core.Analysis import Analysis
import numpy as np

class MachineLearningAnalysis(Analysis):

    """
    Represents MachineLearningAnalysis class that inherits from Analysis class

    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, classes=None):

        """
        Initializes the MachineLearningAnalysis instance

        Args:
            name (str): The name of the analysis.
            replicate (str): The name of the replicate.
            blank (str): The name of the blank.
            data (dict): The data of the analysis, which is a dict of one dimension numpy arrays.
            class (str): Soon!
        """

        super().__init__(name, replicate, blank, data)
        self.classes = str(classes) if classes else None 
    
    def validate(self):
        """
        Validates the analysis object.

        """
        valid = True  
        
        if not super().validate():
            valid = False
        
        if self.data != {}:
            for key in self.data:
                if not isinstance(self.data[key], np.ndarray):
                    print("Analysis data must be a numpy array!")
                    valid = False
            if len(set([self.data[key].ndim for key in self.data])) != 1:
                print("Analysis data arrays must have only one dimension!")
                valid = False
            if len(set([len(self.data[key]) for key in self.data])) != 1:
                print("Analysis data arrays must have the same length!")
                valid = False
        
        if not isinstance(self.classes, str) and self.classes is not None:
            print("Analysis classes name not conform!")
            valid = False

        if not valid:
            print("Issue/s found with analysis", self.name)
        return valid
            
    