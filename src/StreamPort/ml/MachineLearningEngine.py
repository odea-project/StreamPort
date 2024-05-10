from ..core.CoreEngine import CoreEngine
from ..core.Analyses import Analyses
import pandas as pd
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
            # if file exists else warning
            if os.path.exists(path):
            # if check the structure of the csv
                df = pd.read_csv(path)  
                from ..core.CoreEngine import CoreEngine
from ..core.Analyses import Analyses
import pandas as pd
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
            # if file exists else warning
            if os.path.exists(path):
                df = pd.read_csv(path)  
                # if check the structure of the csv
                structure = {
                    "number_of_rows": df.shape[0],
                    "number_of_columns": df.shape[1],
                }
                print(f"Structure of the CSV file: {structure}")         
            else :
                raise FileNotFoundError(f"The file {path} does not exist.")
        else:
            return None

        # collect the names of the analyses  
        #analyses_names = df.items()
        analyses_name = df.iloc[:,0].tolist()

        # if checking if there are no duplicated names in rows, warn the user
        if df.duplicated('name', keep='first').any():
            print("Warning: Duplicate analysis names found in the CSV file.")

        #columns_count = df.shape[1]
        #rows_count = df.shape[0]

        column_names = df.columns.tolist()[1:] # x value for all analyses
        # remove the first column name from column_names (i.e. name)
            
        # loop to each row, as each row is an analysis
        for index, row in df.iterrows():
        # extract the raw values and add it to the y arrayS
            row_value = row.tolist()[1:]
        # each analysis is added with self.add_analyses(anal1)  
            anal1 = [
                Analyses(name=analyses_name[index], data={"x": column_names, "y": row_value})
                ]
    
            self.add_analyses(anal1)
     
            