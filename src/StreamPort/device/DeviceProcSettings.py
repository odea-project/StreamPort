from src.StreamPort.core.ProcessingSettings import ProcessingSettings
from src.StreamPort.device.DeviceEngine import DeviceEngine


# Processing method specific class
class ExtractFeatures(ProcessingSettings):
  def __init__(self):
    super().__init__()
    self.call = "extract_features"

  def run(self):
    pass
  

# Algorithm specific class
class ExtractPressureFeatures(ExtractFeatures):
  def __init__(self, parameters=None):
    super().__init__()
    self.algorithm = "pressure_features"
    self.parameters = ['min', 'max', 'mean', 'std'] if isinstance(parameters, type(None)) else parameters

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      for analysis in engine._analyses:
        if isinstance(analysis.data, dict) and 'Device Pressure Analysis' in analysis.data:
          results.update({analysis.name: analysis.data})
        else:
          print(f"Skipping {analysis.name} because its data is not a dictionary with a 'Pressure' key.")

    for key in results:
        data = results[key]
        if "Dataframe" in data:
            extracted_features = engine.get_features(data['Dataframe'], self.parameters)
            results[key].update({key: extracted_features})
        
    return results
  

"""
# Algorithm specific class
class NormalizeDataSNV(NormalizeData):
  def __init__(self, liftToZero = True):
    super().__init__()
    self.algorithm = "standard_variance_normalization"
    self.parameters = {"liftToZero": liftToZero}
  
  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      for analysis in engine._analyses:
        if isinstance(analysis.data, dict) and 'y' in analysis.data:
          results.update({analysis.name: analysis.data})
        else:
          print(f"Skipping {analysis.name} because its data is not a dictionary with a 'y' key.")

    for key in results:
      y = np.array(results[key]['y'])
      norm_data = (y - y.mean()) / y.std()

      if self.parameters["liftToZero"]:
        norm_data += abs(norm_data.min())

      results[key].update({'y': norm_data})
    
    return results
"""