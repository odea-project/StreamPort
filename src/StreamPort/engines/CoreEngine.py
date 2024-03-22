import datetime
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

  def add_analyses(self, analyses):
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

  def remove_analyses(self, analyses):
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

  def remove_settings(self, settings):
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
