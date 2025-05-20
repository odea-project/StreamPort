from ...src.StreamPort.core import ProcessingMethod
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
import numpy as np
import warnings


# Processing method specific class
class MakeModel(ProcessingMethod):
    def __init__(self):
        super().__init__()
        self.call = "MakeModel"

    def run(self):
        pass



class MakeModelIsoForest(MakeModel):
    """
    Perform outlier Analyses on scaled data using Isolation forest. 
    This function calls the get_feature_matrix() function of its host MLEngine and retrieves data from the linked DeviceEngine object made to conform to MLEngine data structure.

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
        self._linked_objects = []

    def run(self, engine):
        #mods to pass MLEngine objects for each method_grouped set of results
        #each feature_Analyses in feature_analyses is a tuple of MLEngine object and curve_df
        (feature_analyses, methods) = engine.get_device_data(device=self._device)
        for obj, method in zip(feature_analyses, methods):
            features_df = obj[0].get_data()
            print('This engine: \n')
            obj[0].print()
            print('Anomaly detection - ' + method)
            prediction_scores = obj[0].make_iso_forest(features_df, obj[1], random_state=self.parameters['random_state'])
            print(prediction_scores)
            self._linked_objects.append((obj[0], prediction_scores))

        return self._linked_objects




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
        """
        MOD to handle NA values in data
        """
        data.fillna(0, inplace=True)

        if (self.parameters.get("center_data", None)):
            # mean center the data before PCA
            mean = np.mean(data, axis=0)
            data = data - mean
        
        # Perform PCA directly on uncentered data
        pca = PCA(n_components=self.parameters.get("n_components", None))
        pca_results = pca.fit_transform(data)
            
        # Show PCA characteristics
        #print('Shape before PCA: ', scaled_data.shape)
        print('Shape after PCA: ', pca_results.shape)

        explained_variance_ratio = pca.explained_variance_ratio_
        print("Erklärte Varianz der einzelnen Komponenten:", explained_variance_ratio)

        # Kumulierte erklärte Varianz
        cumulative_variance = explained_variance_ratio.cumsum()
        print("Kumulierte erklärte Varianz:", cumulative_variance)

        # the pca_results should be a general model object to be algorithm dependent structure
        return {"pca_model": (pca_results, pca), "explained_variance_ratio": explained_variance_ratio, "cumulative_variance": cumulative_variance}

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


class MakeModelUMAP(MakeModel):
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, random_state=None):
        super().__init__()
        self.algorithm = "UMAP"
        self.parameters = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "random_state": random_state
        }
        self.version = "1.4.2"
        self.software = "sklearn"

    def run(self, engine):
        data = engine.get_data()
        
        # Hier unterdrücken wir die Warnung während des UMAP-Aufrufs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
            
            # UMAP-Modell ausführen
            umap = UMAP(n_neighbors=self.parameters["n_neighbors"],
                        min_dist=self.parameters["min_dist"],
                        n_components=self.parameters["n_components"],
                        random_state=self.parameters["random_state"])
            
            umap_results = umap.fit_transform(data)

        return {"umap_model" : (umap_results, umap)}

class MakeModelHDBSCAN(MakeModel):
    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean'):
        super().__init__()
        self.algorithm = "HDBSCAN"
        self.parameters = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric
        }
        self.version = "1.4.2"
        self.software = "sklearn"

    def run(self, engine):
        # Retrieve the data
        data = engine.get_data()
        
        # Create and fit the HDBSCAN model
        hdbscan_model = HDBSCAN(min_cluster_size=self.parameters["min_cluster_size"],
                                min_samples=self.parameters["min_samples"],
                                metric=self.parameters["metric"])

        # Fit the model to the data
        hdbscan_model.fit(data)
        
        # Get cluster labels
        cluster_labels = hdbscan_model.labels_
        
        # Return the model and the cluster labels as a dictionary
        return {"hdbscan_model": hdbscan_model, "cluster_labels": cluster_labels}


# class MakeModelRandomForest(MakeModel):
#     def __init__(self, n_estimators=100, max_depth=None, random_state=42):
#         super().__init__()
#         self.algorithm = "RandomForest"
#         self.parameters = {
#             "n_estimators": n_estimators,
#             "max_depth": max_depth,
#             "random_state": random_state
#         }
#         self.version = "1.4.2"
#         self.software = "sklearn"

#     def run(self, engine):
#         data = engine.get_data()
#         target = engine.get_target()

#         rf = RandomForestClassifier(
#             n_estimators=self.parameters["n_estimators"],
#             max_depth=self.parameters["max_depth"],
#             random_state=self.parameters["random_state"]
#         )
#         rf.fit(data, target)
#         rf_results = rf.predict(data)

#         return {"random_forest_model": (rf_results, rf)}


# class StatisticModel():
#     def __init__(self, model):
#         self.model_obj = model
#         if isinstance(model, PCA):
#             self.model_type = "PCA"
#         else:
#             self.model_type = None

#     def plot(self):
#         return None
