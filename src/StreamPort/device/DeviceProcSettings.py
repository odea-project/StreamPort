from src.StreamPort.core.ProcessingSettings import ProcessingSettings


# Processing method specific class
class ExtractFeatures(ProcessingSettings):
  def __init__(self):
    super().__init__()
    self.call = "extract_features"

  def run(self):
    pass
  

# Algorithm specific class
class ExtractPressureFeatures(ExtractFeatures):
  """
  This function will set the conditions to handle feature extraction from pressure data.
  Additional features related to the pressure curves(runtime, runtype) are also added.   
  """
  def __init__(self, parameters=None):
    #user-defined dict/list of features (passed as parameters argument) can be extracted from data.
    super().__init__()
    self.algorithm = "pressure_features"
    self.parameters = ['min', 'max', 'mean', 'std', 'skew', 'kurtosis'] if isinstance(parameters, type(None)) else parameters

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        result = analysis.feature_finder(self.algorithm)
        results.update({result.name: result.data})

    for key in list(results):
      
      data = results[key]
      changed_data = engine.get_features(data, self.parameters)  
      results.update({key: changed_data})
      

    return results
  

# Algorithm specific class
class DecomposeCurves(ExtractFeatures):
  """
  Calculate desired features and return them

  """
  def __init__(self):
    super().__init__()
    self.algorithm = "seasonal_decomposition"
  
  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        result = analysis.feature_finder(self.algorithm)
        results.update({result.name: result.data})

    for key in list(results):
      data = results[key]
      #updates analysis data and returns list of 3 combined dataframes to hold resapective components of all curves
      changed_data = engine.get_seasonal_components(data)
      results.update({key: changed_data})
    
    return results


class FourierTransform(ExtractFeatures):
  """
  Perform Fourier Analysis on data.

  """
  def __init__(self):
    super().__init__()
    self.algorithm = "seasonal_decomposition_transformed"

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        result = analysis.feature_finder(self.algorithm)
        results.update({result.name: result.data})
    
    for key in results:
      data = results[key]
      transformed_data = engine.make_fourier_transform(data)
      results.update({key : transformed_data})
  
    return results
