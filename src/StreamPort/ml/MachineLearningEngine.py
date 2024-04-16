from . import CoreEngine

class MachineLearningEngine(CoreEngine):

    """
    A class for running machine learning that inherits from CoreEngine class.

    Attributes:
        model(object): the machine learning model
        data(object): the data used to train and test the model
    
    """   
 
    def __init__(self, headers=None, settings=None, analyses=None, results=None, model=None, data=None):

        """ 
        Initializes the MachineLearningEngine instance

        Attributes:
            headers (ProjectHeaders, optional): The project headers.
            settings (list, optional): The list of settings.
            analyses (list, optional): The list of analyses.
            esults (dict, optional): The dictionary of results.
            model (object, optiobal): The machine Learning model
            data (object, optional): The dataset used in training and testing the machineLearning model
        """

        super().__init__(headers, settings, analyses, results)
        self.model = model 
        self.data = data

    def train(self):
        """
        Trains the machine learning model using the provided data. 
        """
        print("Training model.")
        
    def predict(self, data):
        """
        Predicts the data using the trained machine learning model.
        Args:
            data : Input data for the prediction.
        """
        print("Make prediction using the trained model.")

   
        