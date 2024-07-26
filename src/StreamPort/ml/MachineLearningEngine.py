from ..core.CoreEngine import CoreEngine
from ..ml.MachineLearningAnalysis import MachineLearningAnalysis
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px


class MachineLearningEngine(CoreEngine):

    """
    A class for running machine learning that inherits from CoreEngine class.

    Attributes:
        headers (ProjectHeaders, optional): The project headers. Instance of ProjectHeaders class.
        settings (list, optional): The list of settings. Instance or list of instances of ProcessingSettings class.
        analyses (list, optional): The list of analyses. Instance or list of instances of MachineLearningAnalysis class.
        results (dict, optional): The dictionary of results.
    
    Methods:
        __init__ (self, headers=None, settings=None, analyses=None, results=None): Initializes the CoreEngine instance.
        add_analyses_from_csv (self, path): Reads a CSV file and adds analyses to the engine.
        add_classes_from_csv (self, path): Readas a CSV file and adds classes to the engine.
        get_data (self): Get data array from analyses.
        add_classes (self, classes): Adds classes to each analysis for classification.
        get_classes (self): Get the added classes.
        make_pca (self): Perform PCA and collects the results.
        plot_pca (self): Plots the PCA results and classes.
    
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
        self._classes=[]

    def add_analyses_from_csv(self, path=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file. (extra details about the csv structure for user)
        
        Raises:
            ValurError: if the structure of the csv file is not expected
            FileNotFoundError: if the csv file does not exist
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


    def add_classes_from_csv(self, class_path=None):
        """
        Method for reading a csv file, where rows are analyses (obversations) and colums are variables.

        Args:
            path (str, optional): The path to the csv file. (extra details about the csv structure for user)
        
        Raises:
            ValurError: if the structure of the csv file is not expected
            FileNotFoundError: if the csv file does not exist
        """

        if class_path is not None:
            if os.path.exists(class_path):
                df = pd.read_csv(class_path)  
                structure = {
                    "number_of_rows": df.shape[0],
                    "number_of_columns": df.shape[1],
                }

                if structure["number_of_rows"] == 0 or structure["number_of_columns"] == 0:
                    raise ValueError("The structure of the CSV file is not as expected.")
                else:
                    print(f"Structure of the CSV file: {structure}")
            else :
                raise FileNotFoundError(f"The file {class_path} does not exist.")
        else:
            return None       
        
        class_name = df['class'].tolist()
        column_names = df.columns.tolist()[1:]

        for index, row in df.iterrows():
            row_value = row.tolist()[1:]
            ana = MachineLearningAnalysis(name=str(class_name[index]), data={"x": np.array(column_names), "y": np.array(row_value)}) 
            if ana.validate():
                self.add_classes(class_name[index])
            else:
                print(f"Analysis {class_name[index]} did not pass validation.")
     
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
        """
        Adds classes (a array of string) to each analysis and use it for classification of the PCA results
        
        Args:
            classes(str or list[str]): The classes or list of classes to add.
        
        Raises:
            TypeError: if the classes parameter is not an instance o a list of instances of string array
            TypeError: if any element in the list of classes is not an instance of string array
        """
        if not self._analyses:
            print("No analyses found")
            return None

        if self._classes is None:
            self._classes = []

        if isinstance(classes, list):
            for class_list in classes:
                if not isinstance(class_list, str):
                    raise TypeError("Each element in the classes list must be a string")
                if class_list not in self._classes:
                    self._classes.append(class_list)
        else:
            if not isinstance(classes, str):
                raise TypeError("The classes must be an instance or a string")
            if classes not in self._classes:
                self._classes.append(classes)

    def get_classes(self):
        """
        Method to get the added classes.
        """
        return self._classes

    def make_model(self):
        # Create a method in the ML engine to perfom PCA and collect the results
        """
        Method to perform plot and collect the results
        """
        if not self._analyses:
            print("No analyses found")
            return None

        # get the settings for PCA from _settings attribute or get_settings from self
        settings = self.get_settings(settings="MakeModel")
        if settings is None:
            print("No setting found")
            return None
        # to find the number of components
        # settings_obj = _settings[which is class MakePCA], return the first
        for settings in self._settings:
            if settings.call == "MakeModel":  
                settings_obj = settings
                break
            else:
                settings_obj = None
        # result = settings_obj.run(self)
        if settings_obj:
            result = settings_obj.run(self)
        # self.add_results(result)
        
        # create a class for model representation
        # add an attribute with model type: PCA, PLS, etc.

        # get data from the model
        # plot data from model
        # predict with model based on new data

            self.add_results({"model": result})
        else:
            print("No settings object found")
    
    def plot_data(self):
        """
        Method for general plot of data from the analysis using Plotly.
        
        """
        # add argument to optionally choose the x val as rows or cols
        # look at plotly for interactive plotting https://plotly.com/python/

        if not self._analyses:
            print("No analyses found")
            return None

        data = self.get_data()
        if data is None:
            print("No data found")
            return None
        
        fig = go.Figure()
        for i, analysis in enumerate(self._analyses):
            fig.add_trace(go.Scatter(x=data.columns, y=data.iloc[i], name=analysis.name))
        
        fig.update_layout(title='General Data', xaxis_title='Feature', yaxis_title='Value')
        fig.show()    

    def plot_pca(self):
        # make a plot method in the ML engine for the PCA results and classes
        """
        Method to plot the PCA results and classes
        """
        if not self._analyses:
            print("No analyses found")
            return None
        
        feature_names = self._analyses[0].data['x']
        if feature_names is None:
            print("No feature names found")
            return None
        
        pca_results, pca = self.get_results("pca_model")
        # if pca_results.model_type not in "PCA":
        #     return None
        # pca_results.plot() 
        if pca_results is None:
            print("No pca results found")
            return None
        
        classes = self.get_classes()
        if classes is None:
            print("No classes found")
            return None

        # for 2d plot pca scores
        pca_df = pd.DataFrame(data=pca_results[:, :2], columns=['PC1', 'PC2'])
        fig = px.scatter(pca_df,
                        x='PC1', 
                        y='PC2',
                        title='PCA Scores',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                        template='plotly'
        )
        fig.show()   

        # for plot pca loading
        loadings = pd.DataFrame(pca.components_[:2].T, columns=['PC1', 'PC2'], index=feature_names)
        fig = px.scatter(
            loadings, 
            x='PC1', 
            y='PC2',
            text=loadings.index,
            title='PCA Loadings',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            template='plotly'
        )
        fig.update_traces(
            textposition='top center',
            textfont=dict(
                size=12
            ),
            marker=dict(size=10)
        )
        fig.show()

    def plot_dbscan(self):

        if not self._analyses:
            print("No analyses found")
            return None

        dbscan_results = self.get_results("dbscan_model")
        if dbscan_results is None:
            print("No dbscan results found")
            return None

        classes = self.get_classes()
        if classes is None:
            print("No classes found")
            return None
        
        data=self.get_data()
        pca = PCA(n_components=2)  
        dbscan_data = pca.fit_transform(data) 
        
        dbscan_comp1 = dbscan_data[:, 0]
        dbscan_comp2 = dbscan_data[:, 1]

        unique_labels = np.unique(dbscan_results)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                
                color = "black"

            class_member_mask = (dbscan_results == label)
            xy = dbscan_data[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], color=color, edgecolor='k', label=f'Cluster {label}')

        plt.scatter(dbscan_comp1, dbscan_comp2, alpha=0.2)
        plt.xlabel('comp1')
        plt.ylabel('comp2')
        plt.title('DBSCAN Clustering')
        plt.colorbar(label='Cluster Label')
        plt.show()