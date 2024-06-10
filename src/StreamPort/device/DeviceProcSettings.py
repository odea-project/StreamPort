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

    for key in results:
      data = results[key]
      extracted_features = engine.get_features(data, self.parameters)  
      results[key].update({f"{key}_pressure_features": extracted_features})
        
    return results
  

# Algorithm specific class
class DecomposeCurves(ExtractFeatures):
  """
  Calculate desired features, plot and return them

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

    for key in results:
      data = results[key]
      seasonal_components = engine.get_seasonal_components(data)
      results[key].update({f"{key}_pressure_components": seasonal_components})

    #self._plot_results(engine, results)

    return results
"""
  def _plot_results(self, engine, results):
    
"""
    #Private method to be called in run(). Try block in case used otherwise.
"""  
    try:
      for key in results: 
        engine.add_analyses(DeviceAnalysis(name=f"{key}_components", data=results[key]))
        engine.plot_analyses(f"{key}_components")
        engine.remove_analyses(f"{key}_components")
    except TypeError:
      print('No input provided!')
"""      


class FourierTransform(ExtractFeatures):
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
    
    transformed_seasonal = engine.make_fourier_transform(results)
    #self._plot_results(engine, transformed_seasonal)
    return transformed_seasonal  
"""  
  def _plot_results(self, engine, results):
"""
    #Private method to be called in run(). Try block in case used otherwise.
"""  
    try:
      for key in results: 
        engine.add_analyses(DeviceAnalysis(name=f"{key}_transform", data=results[key]))
        engine.plot_analyses(f"{key}_transform")
        engine.remove_analyses(f"{key}_transform")
    except TypeError:
      print('No input provided!')
"""  