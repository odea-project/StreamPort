from ...src.StreamPort.core import Analyses
import numpy as np

class MachineLearningAnalyses(Analyses):

    """
    Represents MachineLearningAnalyses class that inherits from Analyses class

    Attributes:
        name (str): The name of the Analyses. 
        replicate (str): The name of the replicate.
        blank (str): The name of the blank.
        data (dict): The data of the Analyses.
        classes (str): The name of the classes.

    Methods: 
        validate (self): Validates the Analyses object.

    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, classes=None):

        """
        Initializes the MachineLearningAnalyses instance

        Args:
            name (str): The name of the Analyses.
            replicate (str): The name of the replicate.
            blank (str): The name of the blank.
            data (dict): The data of the Analyses, which is a dict of one dimension numpy arrays.
            classes (str): Analyses class label assigned at runtime based on IsoForest results.
        """

        super().__init__(name, replicate, blank, data)
        self.classes = str(classes) if classes else self.set_class_label() 
    
    def validate(self):
        """
        Validates the Analyses object.

        """
        valid = True  
        
        if not super().validate():
            valid = False
        
        if self.data != {}:
            for key in self.data:
                if not isinstance(self.data[key], np.ndarray):
                    print("Analyses data must be a numpy array!")
                    valid = False
            if len(set([self.data[key].ndim for key in self.data])) != 1:
                print("Analyses data arrays must have only one dimension!")
                valid = False
            if len(set([len(self.data[key]) for key in self.data])) != 1:
                print("Analyses data arrays must have the same length!")
                valid = False
        
        if not isinstance(self.classes, str) and self.classes is not None:
            print("Analyses classes name not conform!")
            valid = False

        if not valid:
            print("Issue/s found with Analyses", self.name)
        return valid
            
    def set_class_label(self, class_label=None):
        """
        Self_assign class labels input from DeviceEngine's get_feature_matrix() and MLEngine's make_iso_forest() functions.

        """
        if not isinstance(class_label, type(None)) and not '001-blank' in self.name:
            self.classes = str(class_label)

        elif isinstance(class_label, list) and isinstance(class_label[0], str):
            self.classes = class_label[0]

        else:
            if self.data != {} and '001-blank' in self.name:
                self.classes = 'First measurement'             
            else:
                self.classes = "Undefined"
        