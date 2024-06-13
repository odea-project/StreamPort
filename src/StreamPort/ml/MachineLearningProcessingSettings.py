from ..core.ProcessingSettings import ProcessingSettings
from sklearn.decomposition import PCA

# Processing method specific class
class MakeModel(ProcessingSettings):
    def __init__(self):
        super().__init__()
        self.call = "MakeModel"

    def run(self):
        pass

# Algorithm specific class
class MakeModelPCASKL(MakeModel):
    def __init__(self, n_components = 2, center_data = True):
        self.algorithm = "PCA"
        self.parameters = {
            "n_components": n_components,
            "center_data": center_data
        }
        self.version = "1.4.2"
        self.software = "sklearn"
        super().__init__()

    def run(self, engine):
        data = engine.get_data()

        #if (self.parameters.get("center_data", None)):
            # mean center the data before PCA

        pca = PCA(n_components=self.parameters.get("n_components", None)) 
        pca_results = pca.fit_transform(data)
        
        # the pca_results should be a general model object to be algorithm dependent structure
        return {"model": pca_results}

class MakeModelPLSSKL(MakeModel): # try to aply a different model to evaluate the different between the analyses
    def __init__(self, n_components = 2):
        self.algorithm = "PLS"
        self.parameters = {
            "n_components": n_components
        }
        self.version = "1.4.2"
        self.software = "sklearn"
        super().__init__()

    def run(self, engine):
       pass


# class StatisticModel():
#     def __init__(self, model):
#         self.model_obj = model
#         if isinstance(model, PCA):
#             self.model_type = "PCA"
#         else:
#             self.model_type = None

#     def plot(self):
#         return None
