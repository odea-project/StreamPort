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
  Smoothed defaults to false. smoothed = True first smoothes the curve by percentage change per datapoint before extracting features.
  Additional features related to the pressure curves(runtime, runtype) are also added.   
  """
  _weighted = None
  def __init__(self, parameters=None, weighted = None):
    #user-defined dict/list of features (passed as parameters argument) can be extracted from data.
    super().__init__()
    self._weighted = weighted 
    self.algorithm = "pressure_features"
    self.parameters = ['min', 'max', 'mean', 'std', 'skew', 'kurtosis'] if isinstance(parameters, type(None)) else parameters

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        results.update({analysis.name: analysis.data})

    for key in list(results):
      
      data = results[key]
      changed_data = engine.get_features(data, self.parameters, self._weighted)  
      results.update({key: changed_data})
      

    return results
  

# Algorithm specific class
class DecomposeCurves(ExtractFeatures):
  """
  Decompose curves into their components (Trend, Seasonal(periods), Residual(noise)). 
  Period defaults to 10.

  """
  _period = None
  def __init__(self, period=None):
    super().__init__()
    self._period = period 
  
  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        results.update({analysis.name: analysis.data})

    for key in list(results):
      data = results[key]
      #updates analysis data and returns list of 3 combined dataframes to hold resapective components of all curves
      changed_data = engine.get_seasonal_components(data, self._period)
      results.update({key: changed_data})
    
    return results


class FourierTransform(ExtractFeatures):
  """
  Perform Fourier Analysis on data.

  """
  def __init__(self):
    super().__init__()

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        results.update({analysis.name: analysis.data})
    
    for key in list(results):
      data = results[key]
      transformed_data = engine.make_fourier_transform(data)
      results.update({key : transformed_data})
  
    return results



class Scaler(ProcessingSettings):
  """
  Scale data based on user input. Defaults to MinMaxScaler

  """
  def __init__(self, parameters= None):
    super().__init__() 
    self.parameters = 'minmax' if isinstance(parameters, type(None)) else parameters
    self.algorithm = "scaling"

  def run(self, engine):
    if engine._results.__len__() > 0:
      results = engine._results
    else:
      results = {}
      analyses = engine.get_analyses([i for i in range(0,len(engine._analyses))])
      for analysis in analyses:
        results.update({analysis.name: analysis.data})

    for ana_name in list(results):
      this_analysis_data = results[ana_name]
      #fix NaN values in extraction or calculation
      updated_data = engine.add_extracted_features(this_analysis_data)
      print('after add_extracted_feats()')
      print(updated_data)
      results.update({ana_name : updated_data})
    #something goes wrong here
    prepared_data = engine.group_analyses(results)
    print('after group_analyses')
    print(f'number of newly grouped analysis objects : {len(prepared_data)}')
    print([ana.data for ana in prepared_data])
    scaled_data = engine.scale_features(prepared_data, type=self.parameters)
    
    return scaled_data
  

