from ..core.ProcessingSettings import ProcessingSettings
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np


# Processing method specific class
class MakeModel(ProcessingSettings):
    def __init__(self):
        super().__init__()
        self.call = "MakeModel"

    def run(self):
        pass



class MakeModelIsoForest(MakeModel):
    """
    Perform outlier analysis on scaled data using Isolation forest. 
    This function calls the classify() function of its host MLEngine and retrieves data from the linked DeviceEngine object made to conform to MLEngine data structure.

    """
    def __init__(self, device, random_state=None):
        super().__init__()
        self.algorithm = "Isolation forest"
        self.parameters = {
          "random_state": random_state
        }
        self.version = "1.4.2"
        self.software = "sklearn"
        self._device = device

    def run(self, engine):
        (feature_analyses, methods) = engine.get_device_data(device=self._device)
        for ana, method in zip(feature_analyses, methods):
            engine.add_analyses(ana[0])
            features_df = engine.get_data()
            print('Anomaly detection - ' + method)
            prediction_scores = engine.make_iso_forest(features_df, ana[1], random_state=self.parameters['random_state'])
            print(prediction_scores)
            engine.remove_analyses()
        return 




# Algorithm specific class
class MakeModelPCASKL(MakeModel):
    def __init__(self, n_components = 2, center_data = True):
        super().__init__()
        self.algorithm = "PCA"
        self.parameters = {
            "n_components": n_components,
            "center_data": center_data
        }
        self.version = "1.4.2"
        self.software = "sklearn"

    def run(self, engine):
        data = engine.get_data()

        if (self.parameters.get("center_data", None)):
            # mean center the data before PCA
            mean = np.mean(data, axis=0)
            data = data - mean
        
        # Perform PCA directly on uncentered data
        pca = PCA(n_components=self.parameters.get("n_components", None))
        pca_results = pca.fit_transform(data)
            
        # the pca_results should be a general model object to be algorithm dependent structure
        return {"pca_model": (pca_results, pca)}

class MakeModelDBSCANSKL(MakeModel): # try to aply a different model to evaluate the different between the analyses
    def __init__(self, eps = 0.5, min_samples = 5):
        super().__init__()
        self.algorithm = "DBSCAN"
        self.parameters = {
            "eps": eps,
            "min_samples": min_samples
        }
        self.version = "1.4.2"
        self.software = "sklearn"

    def run(self, engine):
        data = engine.get_data()
        eps = self.parameters.get("eps", 0.5)
        min_samples = self.parameters.get("min_samples", 5)
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan_results = dbscan.fit(data)
        labels = dbscan_results.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        return {"dbscan_model": dbscan_results}



# class StatisticModel():
#     def __init__(self, model):
#         self.model_obj = model
#         if isinstance(model, PCA):
#             self.model_type = "PCA"
#         else:
#             self.model_type = None

#     def plot(self):
#         return None
