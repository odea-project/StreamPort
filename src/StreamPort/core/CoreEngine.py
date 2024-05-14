from . import ProjectHeaders
from .ProjectHeaders import ProjectHeaders
from .Analysis import Analysis
from .ProcessingSettings import ProcessingSettings

class CoreEngine:
  """
  The CoreEngine class represents the core engine of the StreamPort application.
  It manages project headers, analyses, settings, results, and history.

  Attributes:
    _headers (ProjectHeaders): The project headers.
    _analyses (list): The list of analyses.
    _settings (list): The list of settings.
    _results (dict): The dictionary of results.
    _history (dict): The dictionary of history.

  Methods:
    __init__(self, headers=None, settings=None, analyses=None, results=None): Initializes the CoreEngine instance.
    __str__(self): Returns a string representation of the CoreEngine instance.
    print(self): Prints the CoreEngine instance.
    add_headers(self, headers): Adds project headers.
    get_headers(self, headers): Retrieves project headers.
    remove_headers(self, headers): Removes project headers.
    add_analyses(self, analyses): Adds analyses.
    get_analysis(self, analyses): Retrieves analyses.
    remove_analyses(self, analyses): Removes analyses.
    add_settings(self, settings): Adds settings.
    get_settings(self, settings): Retrieves settings.
    remove_settings(self, settings): Removes settings.
    add_results(self, results): Adds results.
    get_results(self, results): Retrieves results.
    remove_results(self, results): Removes results.
  """

  _headers = ProjectHeaders()
  _analyses = []
  _settings = []
  _results = {}
  _history = {}

  def __init__(self, headers=None, settings=None, analyses=None, results=None):
    """
    Initializes the CoreEngine instance.

    Args:
      headers (ProjectHeaders, optional): The project headers.
      settings (list, optional): The list of settings.
      analyses (list, optional): The list of analyses.
      results (dict, optional): The dictionary of results.
    """
    self._headers = ProjectHeaders()
    self._analyses = []
    self._settings = []
    self._results = {}
    self._history = {}

    if headers is not None:
      self.add_headers(headers)
    if analyses is not None:
      self.add_analyses(analyses)
    if settings is not None:
      self.add_settings(settings)
    if results is not None:
      self.add_results(results)

  def __str__(self):
    """
    Returns a string representation of the CoreEngine instance.
    """
    return f"\n{type(self).__name__} \n" \
      f"  name: {self._headers.headers['name']} \n" \
      f"  author: {self._headers.headers['author']} \n" \
      f"  path: {self._headers.headers['path']} \n" \
      f"  date: {self._headers.headers['date']} \n" \
      f"  analyses: {len(self._analyses)} \n" \
      f"  settings: {len(self._settings)} \n"

  def print(self):
    """
    Prints the CoreEngine instance.
    """
    print(self)

  def add_headers(self, headers):
    """
    Adds headers to the CoreEngine's project headers.

    Args:
      headers (ProjectHeaders or dict): The headers to be added. If `headers` is an instance of `ProjectHeaders`,
        the headers will be updated with the existing headers. If `headers` is a dictionary, the headers will be
        added to the existing headers.

    Returns:
      None
    """
    if self._headers is None:
      self._headers = ProjectHeaders()
    if isinstance(headers, ProjectHeaders):
      self._headers.headers.update(headers.headers)
    elif isinstance(headers, dict):
      self._headers.headers.update(headers)

  def get_headers(self, headers = None):
    """
    Retrieves the specified headers from the internal headers dictionary.

    Args:
      headers (str or list): The headers to retrieve. Can be a single header name (str) or a list of header names.

    Returns:
      dict or str or None: If `headers` is a single header name (str), returns the corresponding header value as a string.
      If `headers` is a list of header names, returns a dictionary containing the header names as keys and their corresponding values as strings.
      If `headers` is not provided or is of an unsupported type, returns the entire headers dictionary.

    """
    if isinstance(headers, str):
      return self._headers.headers.get(headers, None)
    elif isinstance(headers, list):
      return {header: self._headers.headers.get(header, None) for header in headers}
    else:
      return self._headers.headers

  def remove_headers(self, headers = None):
    if self._headers.headers.__len__() == 0:
      return
    if isinstance(headers, list):
      for header in headers:
        if header in self._headers.headers:
          del self._headers.headers[header]
    elif isinstance(headers, str):
      if headers in self._headers.headers:
        del self._headers.headers[headers]

  def add_analyses(self, analyses):
    """
    Adds one or more analyses to the CoreEngine.

    Args:
      analyses (Analysis or list[Analysis]): The analysis or list of analyses to add.

    Raises:
      TypeError: If the analyses parameter is not an instance or a list of instances of the Analysis class.
             If any element in the list is not an instance of the Analysis class.

    """
    if self._analyses is None:
      self._analyses = []
    if isinstance(analyses, list):
      for analysis in analyses:
        if not isinstance(analysis, Analysis):
          raise TypeError("Each element of analyses must be an instance of Analysis class")
        if analysis.name not in [a.name for a in self._analyses]:
          self._analyses.append(analysis)
    else:
      if not isinstance(analyses, Analysis):
        raise TypeError("The analyses must be an instance or a list of instances of Analysis class")
      if analyses.name not in [a.name for a in self._analyses]:
        self._analyses.append(analyses)

  def get_analysis(self, analyses):
    """
    Retrieves the analysis object(s) based on the provided input.

    Parameters:
    - analyses: An integer, string, or list of integers/strings representing the analysis object(s) to retrieve.

    Returns:
    - If `analyses` is an integer and within the range of available analyses, returns the analysis object at the specified index.
    - If `analyses` is a string, returns the analysis object with a matching name.
    - If `analyses` is a list of integers/strings, returns a list of analysis objects corresponding to the provided indices/names.
    - If `analyses` is not provided or of an unsupported type, returns all available analysis objects.

    Note:
    - If `analyses` is an integer or string and no matching analysis object is found, None is returned.
    - If `analyses` is a list and no matching analysis objects are found, an empty list is returned.
    """
    if isinstance(analyses, int) and analyses < len(self._analyses):
      return self._analyses[analyses]
    elif isinstance(analyses, str):
      for analysis in self._analyses:
        if analysis.name == analyses:
          return analysis
    elif isinstance(analyses, list):
      analyses_out = []
      for analysis in analyses:
        if isinstance(analysis, int) and analysis < len(self._analyses):
          analyses_out.append(self._analyses[analysis])
        elif isinstance(analysis, str):
          for a in self._analyses:
            if a.name == analysis:
              analyses_out.append(a)
      return analyses_out
    else:
      return self._analyses

  def remove_analyses(self, analyses = None):
    """
    Removes the specified analyses from the list of analyses.

    Args:
      analyses (int, str, list): The analyses to be removed. It can be an integer index, a string name, or a list of indices or names.

    Returns:
      None

    Raises:
      None
    """
    if self._analyses.__len__() == 0:
      return
    if isinstance(analyses, list):
      for item in analyses:
        if isinstance(item, int) and item < len(self._analyses):
          del self._analyses[item]
        elif isinstance(item, str):
          self._analyses = [analysis for analysis in self._analyses if analysis.name != item]
    elif isinstance(analyses, int) and analyses < len(self._analyses):
      del self._analyses[analyses]
    elif isinstance(analyses, str):
      self._analyses = [analysis for analysis in self._analyses if analysis.name != analyses]
    else:
      self._analyses = []

  def add_settings(self, settings = None):
    """
    Adds the given settings to the list of processing settings.

    Args:
      settings (ProcessingSettings or list[ProcessingSettings]): The settings to be added. 
        It can be a single instance of ProcessingSettings or a list of instances.

    Raises:
      TypeError: If the settings parameter is not an instance or a list of instances of ProcessingSettings class.
      TypeError: If any element in the list of settings is not an instance of ProcessingSettings class.
    """
    if self._settings is None:
      self._settings = []
    if isinstance(settings, list):
      for set in settings:
        if not isinstance(set, ProcessingSettings):
          raise TypeError("Each element of settings must be an instance of ProcessingSettings class")
        if set.call not in [s.call for s in self._settings]:
          self._settings.append(set)
    else:
      if not isinstance(settings, ProcessingSettings):
        raise TypeError("The settings must be an instance or a list of instances of ProcessingSettings class")
      if settings.call not in [s.call for s in self._settings]:
        self._settings.append(settings)

  def get_settings(self, settings = None):
    """
    Retrieves the specified settings from the CoreEngine.

    Parameters:
    - settings: Can be an integer, string, or a list of integers/strings.
          If an integer is provided, it returns the setting at the specified index.
          If a string is provided, it returns the setting with the matching call name.
          If a list is provided, it returns a list of settings corresponding to the provided indices or call names.

    Returns:
    - If settings is an integer or string, returns the corresponding setting.
    - If settings is a list, returns a list of settings.

    If the provided settings parameter does not match any existing settings, it returns all settings.

    """
    if isinstance(settings, int) and settings < len(self._settings):
      return self._settings[settings]
    elif isinstance(settings, str):
      for set in self._settings:
        if set.call == settings:
          return set
    elif isinstance(settings, list):
      settings_out = []
      for item in settings:
        if isinstance(item, int) and item < len(self._settings):
          settings_out.append(self._settings[item])
        elif isinstance(item, str):
          for set in self._settings:
            if set.call == item:
              settings_out.append(set)
      return settings_out
    else:
      return self._settings

  def remove_settings(self, settings = None):
    """
    Removes the specified settings from the internal settings list.

    Args:
      settings: The settings to be removed. It can be either an integer index, a string representing the call name,
            or a list of integers and/or strings.

    Returns:
      None
    """
    if self._settings.__len__() == 0:
      return
    if isinstance(settings, list):
      for item in settings:
        if isinstance(item, int) and item < len(self._settings):
          del self._settings[item]
        elif isinstance(item, str):
          self._settings = [set for set in self._settings if set.call != item]
    elif isinstance(settings, int) and settings < len(self._settings):
      del self._settings[settings]
    elif isinstance(settings, str):
      self._settings = [set for set in self._settings if set.call != settings]
    else:
      self._settings = []	
  
  def add_results(self, results):
    """
    Adds the given results to the internal results dictionary.

    Args:
      results (dict): A dictionary containing the results to be added.

    """
    if self._results is None:
      self._results = {}
    if isinstance(results, dict):
      self._results.update(results)

  def get_results(self, results):
    """
    Retrieves the results from the CoreEngine.

    Args:
      results (str or list): The key(s) of the result(s) to retrieve.

    Returns:
      dict or any: If `results` is a string, returns the corresponding result value.
             If `results` is a list, returns a dictionary with the key-value pairs
             of the requested results. If `results` is neither a string nor a list,
             returns all the results.

    """
    if isinstance(results, str):
      return self._results.get(results, None)
    elif isinstance(results, list):
      out_results = {}
      for result in results:
        out_results[result] = self._results.get(result, None)
      return out_results
    else:
      return self._results
  
  def remove_results(self, results = None):
    """
    Removes the specified results from the internal results dictionary.

    Args:
      results: The results to be removed. It can be an integer, a string, or a list of integers or strings.

    Returns:
      None
    """
    if self._results.__len__() == 0:
      return
    if isinstance(results, int):
      keys = list(self._results.keys())
      if results < len(keys):
        del self._results[keys[results]]
    if isinstance(results, str):
      if results in self._results:
        del self._results[results]
    if isinstance(results, list):
      for result in results:
        if isinstance(result, int):
          keys = list(self._results.keys())
          if result < len(keys):
            del self._results[keys[result]]
        elif isinstance(result, str):
          if result in self._results:
            del self._results[result]
    else:
      self._results = {}

  def run_workflow(self):
    if self._settings.__len__() == 0:
      print("No settings found to run workflow!")
      return
    else:
      for settings in self._settings:
        print(f"Running workflow with settings: {settings.call}")
        results = settings.run(self)
        self.add_results(results)
