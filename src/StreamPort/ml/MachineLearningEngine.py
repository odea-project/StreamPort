from ..core.CoreEngine import CoreEngine
from ..core.Analysis import Analysis
import pandas as pd
import numpy as np
import os

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

    def read_csv(self, path=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file. (extra details about the csv structure for user)
        """

        if path is not None:
            if os.path.exists(path):
                df = pd.read_csv(path)  
                structure = {
                    "number_of_rows": df.shape[0],
                    "number_of_columns": df.shape[1],
                }

                if structure["number_of_rows"] == 0 or structure["number_of_columns"] == 0:
                    raise ValueError("The structure of the CSV file is not as expected.")
                else:
                    print(f"Structure of the CSV file: {structure}")
            else :
                raise FileNotFoundError(f"The file {path} does not exist.")
        else:
            return None
        
        analyses_name = df.iloc[:,0].tolist()

        if df.duplicated('name', keep='first').any():
            print("Warning: Duplicate analysis names found in the CSV file. Only the first will be added!")

        column_names = df.columns.tolist()[1:] 

        for index, row in df.iterrows():
            row_value = row.tolist()[1:]
            ana = [Analysis(name=analyses_name[index], data={"x": column_names, "y": row_value})]
            self.add_analyses(ana)
     
    def get_data(self):

        # collapse all data arrays from analyses into a matrix for statistics
        # cols are the x (x is all the same in analyses) and rows are the values for each analysis  
     
        if not self._analyses:
            print("No analyses found")
            return None
        
        x_values = self._analyses[0].data["x"]
    
        matrix = []
        for analysis in self._analyses:
            y_values = analysis.data["y"]
            fil_y_values = []
            for value in y_values:
                if value == 0:
                    fil_y_values.append(np.nan)
                else:
                    fil_y_values.append(value)
            matrix.append(fil_y_values)
        
        df_matrix = pd.DataFrame(matrix, columns=x_values)
        
        return df_matrix