from ..core.ProcessingSettings import ProcessingSettings
from sklearn.decomposition import PCA

class MakeModel(ProcessingSettings):
    def __init__(self, call, algorithm, parameters, version, software, developer, contact, link, doi):
        super().__init__(call, algorithm, parameters, version, software, developer, contact, link, doi)
        
    def run(self):
        pass

class MakeModelPCASKL(MakeModel):
    def __init__(self, n_components = 2):
        call="MakePCA"
        algorithm="PCA"
        parameters={
            "n_components": n_components
        }
        version="1.4.2"
        software="sklearn"
        developer=None
        contact=None
        link=None
        doi=None
        super().__init__(call, algorithm, parameters, version, software, developer, contact, link, doi)

    def run(self, engine):
        data = engine.get_data()
        pca = PCA(n_components=2) 
        pca_results = pca.fit_transform(data)
        return pca_results
