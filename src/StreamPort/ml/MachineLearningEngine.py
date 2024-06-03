from ..core.CoreEngine import CoreEngine
from ..ml.MachineLearningAnalysis import MachineLearningAnalysis
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
            ana = MachineLearningAnalysis(name=str(analyses_name[index]), data={"x": np.array(column_names), "y": np.array(row_value)}) 
            if ana.validate():
                self.add_analyses(ana)
            else:
                print(f"Analysis {analyses_name[index]} did not pass validation.")
     
    def get_data(self):
        """
        Method for collapse all data arrays from analyses into a matrix for statistics

        """
     
        if not self._analyses:
            print("No analyses found")
            return None
        
        x_values = self._analyses[0].data["x"]
    
        matrix = []
        for analysis in self._analyses:
            y_values = analysis.data["y"]
            matrix.append(y_values)
        
        df_matrix = pd.DataFrame(matrix, columns=x_values)
        
        return df_matrix
    
    def add_classes(self, classes):

        # added classes (a array of strings) to each analysis and then use it for classification of the PCA results
        return None