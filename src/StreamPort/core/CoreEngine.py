from . import ProjectHeaders
from .ProjectHeaders import ProjectHeaders
from .Analyses import Analyses
from .ProcessingSettings import ProcessingSettings

class CoreEngine:
  """
  The CoreEngine class represents the core engine of the StreamPort application.
  It manages project headers, settings, analyses, history, and results.

  Attributes:
    _headers (ProjectHeaders): The project headers.
    _settings (list): The list of processing settings.
    _analyses (list): The list of analyses.
    _history (None): The history (not implemented yet).
    _results (None): The results (not implemented yet).
  """
  _headers = None
  _settings = None
  _analyses = None
  _history = None
  _results = None

  def __init__(self, headers=None, settings=None, analyses=None, results=None):
    """
    Initializes a new instance of the CoreEngine class.

    Args:
      headers (ProjectHeaders, optional): The project headers. Defaults to None.
      settings (list, optional): The list of processing settings. Defaults to None.
      analyses (list, optional): The list of analyses. Defaults to None.
      results (None, optional): The results (not implemented yet). Defaults to None.
    """
    if headers is None:
      headers = ProjectHeaders()
    if analyses is not None:
      self.add_analyses(analyses)
    self._headers = headers
    self._settings = settings
    self._results = results

  def __str__(self):
    """
    Returns a string representation of the CoreEngine object.

    Returns:
      str: The string representation of the CoreEngine object.
    """
    return f"{type(self).__name__} \n" \
      f"  name: {self._headers.headers['name']} \n" \
      f"  author: {self._headers.headers['author']} \n" \
      f"  path: {self._headers.headers['path']} \n" \
      f"  date: {self._headers.headers['date']} \n"

  def print(self):
    """
    Prints the string representation of the CoreEngine object.
    """
    print(self)

  def add_headers(self, headers):
    """
    Adds headers to the CoreEngine's project headers.

    Args:
        headers (ProjectHeaders or dict): The headers to be added. Can be an instance of ProjectHeaders or a dictionary.

    """
    if self._headers is None:
      self._headers = ProjectHeaders()
    if isinstance(headers, ProjectHeaders):
      self._headers.headers.update(headers.headers)
    elif isinstance(headers, dict):
      self._headers.headers.update(headers)

  def get_headers(self, headers):
    """
    Retrieves the specified headers from the internal headers dictionary.

    Args:
      headers (str or list): The headers to retrieve. Can be a single header as a string or a list of headers.

    Returns:
      dict or None: If `headers` is a string, returns the value of the specified header. 
              If `headers` is a list, returns a dictionary with the specified headers as keys and their values as values.
              If `headers` is None or not provided, returns the entire headers dictionary.

    """
    if self._headers is None:
      return None
    if isinstance(headers, str):
      return self._headers.headers.get(headers, None)
    elif isinstance(headers, list):
      return {header: self._headers.headers.get(header, None) for header in headers}
    else:
      return self._headers.headers

  def remove_headers(self, headers):
    """
    Removes the specified headers from the internal headers dictionary.

    Args:
      headers (str or list): The headers to remove. Can be a single header as a string or a list of headers.

    """
    if self._headers is None:
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
    Add one or more analyses to the CoreEngine.

    Args:
      analyses (Analyses or list[Analyses]): The analysis or list of analyses to add.

    Raises:
      TypeError: If the analyses parameter is not an instance or a list of instances of the Analyses class.
             If any element in the list is not an instance of the Analyses class.

    """
    if self._analyses is None:
      self._analyses = []
    if isinstance(analyses, list):
      for analysis in analyses:
        if not isinstance(analysis, Analyses):
          raise TypeError("Each element of analyses must be an instance of Analyses class")
        if analysis.name not in [a.name for a in self._analyses]:
          self._analyses.append(analysis)
    else:
      if not isinstance(analyses, Analyses):
        raise TypeError("The analyses must be an instance or a list of instances of Analyses class")
      if analyses.name not in [a.name for a in self._analyses]:
        self._analyses.append(analyses)

  def get_analysis(self, analyses):
    """
    Retrieves the analysis object based on the given input.

    Parameters:
    - analyses: Can be an integer, string, or a list of integers/strings.
          If an integer is provided, it returns the analysis object at the specified index.
          If a string is provided, it returns the analysis object with the matching name.
          If a list is provided, it returns a list of analysis objects corresponding to the input.

    Returns:
    - If a single analysis object is found, it is returned.
    - If multiple analysis objects are found, a list of analysis objects is returned.
    - If no analysis object is found, None is returned.

    Note:
    - If the input is an integer or a list of integers, the index should be within the range of available analysis objects.
    - If the input is a string or a list of strings, the name should match the name of an available analysis object.
    """
    if self._analyses is None:
      return None
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
    return self._analyses

  def remove_analyses(self, analyses):
    """
    Remove the specified analyses from the list of analyses.

    Args:
      analyses (list, int, str): The analyses to be removed. It can be a list of indices, a single index, or a name.

    Returns:
      None

    Raises:
      None
    """
    if self._analyses is None:
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

  def add_settings(self, settings):
    """
    Add processing settings to the core engine.

    Args:
      settings (ProcessingSettings or list[ProcessingSettings]): The processing settings to be added.
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

  def get_settings(self, settings):
    """
    Retrieves the specified settings from the CoreEngine.

    Parameters:
    - settings: The settings to retrieve. It can be an integer index, a string representing the call name, or a list of integers or strings.

    Returns:
    - If `settings` is an integer and within the range of available settings, returns the corresponding setting.
    - If `settings` is a string, returns the first setting with a matching call name.
    - If `settings` is a list, returns a list of settings corresponding to the provided indices or call names.
    - If `settings` is None or not found, returns None.
    - If `settings` is not of type int, str, or list, returns all available settings.

    Note: The method assumes that the `_settings` attribute is already populated with the available settings.

    """
    if self._settings is None:
      return None
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
    return self._settings

  def remove_settings(self, settings):
    """
    Removes the specified settings from the internal settings list.

    Args:
      settings (int, str, list): The settings to be removed. It can be an integer index, a string name, or a list of integers or strings.

    Returns:
      None
    """
    if self._settings is None:
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
