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

    def read_csv(self, path=None, df=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file.
            df (pandas.DataFrame, optional): The dataframe to be read.
        """

        if path is not None:
            df = pd.read_csv(path)
        return df

    def add_analysis(self, df=None):

        """
        Method for adding analysis to the MachineLearningEngine instance. 
        
        Args:
            df (pandas.DataFrame, optional): The dataframe to be read.

        """
    
        columns_count = 4444
        counter = 1
        rows_count = 2

        columns = [] 
        rows = []

        for x, y in df.items():

            if counter <= columns_count: 
                columns.append(x) 

                row_counter = 1 
                for row_value in y: 

                    if row_counter < rows_count and row_counter > 0: 
                        rows.append(row_value) 
                    row_counter += 1 

            counter += 1 

        print("Create a list of analysis object and prints it" )
        anal1 = [
            Analyses(name="Analysis1", data={"x": columns, "y": rows})
        ]

        return anal1