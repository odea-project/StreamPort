from ..core.CoreEngine import CoreEngine
from ..core.Analyses import Analyses
import pandas as pd

class MachineLearningEngine(CoreEngine):

    """
    A class for running machine learning that inherits from CoreEngine class.
    
    """   
 
    def __init__(self, headers=None, settings=None, analyses=None, results=None):

        """ 
        Initializes the MachineLearningEngine instance

        Args:
            headers (ProjectHeaders, optional): The project headers.
            settings (list, optional): The list of settings.
            analyses (list, optional): The list of analyses.
            results (dict, optional): The dictionary of results.
        """

        super().__init__(headers, settings, analyses, results)

    def read_csv(self, fea_list=None, fea_metadata=None):
        """
        Method for reading the csv file with pandas

        Args:
            fea_list (pd.DataFrame, optional): The dataframe of feature list.
            fea_metadata (pd.DataFrame, optional): The dataframe of feature metadata.
        """
        fea_list = pd.read_csv('feature_list.csv')
        fea_metadata = pd.read_csv('feature_metadata.csv')
        return fea_list, fea_metadata

    def add_analysis(self, analysis):

        """
        Method for adding analysis to the MachineLearningEngine instance.
        
        Args:
            analysis (Analyses): The analysis to be added.
        """
        if self._analyses is None:
            self._analyses = []

        if isinstance(analysis, Analyses):
            if analysis.name not in [a.name for a in self._analyses]:
                self._analyses.append(analysis)
        else:
            raise TypeError("The analyses must be an instance or a list of instances of Analyses class")

