import pandas as pd

class ProcessingSettings:
  """
  Represents the processing settings for a specific algorithm.

  Attributes:
    call (str): The call entry.
    algorithm (str): The algorithm entry.
    parameters (dict): The parameters entry.
    version (str): The version entry.
    software (str): The software entry.
    developer (str): The developer entry.
    contact (str): The contact entry.
    link (str): The link entry.
    doi (str): The DOI entry.
  """

  def __init__(self, call=None, algorithm=None, parameters={}, version=None, software=None, developer=None, contact=None, link=None, doi=None):
    self.call = str(call)
    self.algorithm = str(algorithm)
    self.parameters = dict(parameters)
    self.version = str(version)
    self.software = str(software)
    self.developer = str(developer)
    self.contact = str(contact)
    self.link = str(link)
    self.doi = str(doi)

  def validate(self):
    """
    Validates the processing settings.

    Returns:
      bool: True if the settings are valid, False otherwise.
    """
    valid = False
    if isinstance(self.call, str) and isinstance(self.algorithm, str) and isinstance(self.parameters, dict) and isinstance(self.version, str):
      valid = True
      if len(self.call) != 1 or not isinstance(self.algorithm, str):
        print("Call entry must be of length 1!")
        valid = False
      if len(self.algorithm) != 1 or not isinstance(self.algorithm, str):
        print("Algorithm entry must be of length 1 and type character!")
        valid = False
      if not isinstance(self.parameters, dict):
        print("Parameters entry must be a list or an S4 class!")
        valid = False
      if len(self.version) != 1 or not isinstance(self.version, str):
        print("Version entry must be of length 1 and type character!")
        valid = False
    else:
      print("Settings elements must be named call, algorithm and parameters!")
    return valid

  def __str__(self):
    """
    Returns a string representation of the processing settings.

    Returns:
      str: The string representation of the processing settings.
    """
    result = "\n"
    result += " ProcessingSettings\n"
    result += " call         " + str(self.call) + "\n"
    result += " algorithm    " + str(self.algorithm) + "\n"
    result += " version      " + str(self.version) + "\n"
    result += " software     " + str(self.software) + "\n"
    result += " developer    " + str(self.developer) + "\n"
    result += " contact      " + str(self.contact) + "\n"
    result += " link         " + str(self.link) + "\n"
    result += " doi          " + str(self.doi) + "\n"
    result += "\n"
    result += " parameters: "
    if len(self.parameters) == 0:
      result += "empty " + "\n"
    else:
      result += "\n"
      for i in range(len(self.parameters)):
        if isinstance(self.parameters[i], pd.DataFrame):
          result += "  - " + str(dict(self.parameters.keys())[i]) + " (only head rows)" + "\n"
          result += "\n"
          result += str(self.parameters[i].head()) + "\n"
          result += "\n"
        elif isinstance(self.parameters[i], dict):
          result += "  - " + str(dict(self.parameters.keys())[i]) + ": " + "\n"
          for j in range(len(self.parameters[i])):
            result += "      - " + str(dict(self.parameters[i].keys())[j]) + str(self.parameters[i][j]) + "\n"
        elif isinstance(self.parameters[i], function):
          result += "  - " + str(dict(self.parameters.keys())[i]) + ":\n"
          result += str(self.parameters[i]) + "\n"
        else:
          result += "  - " + str(dict(self.parameters.keys())[i]) + str(self.parameters[i]) + "\n"
    return result

  def print(self):
    """
    Prints the string representation of the processing settings.
    """
    print(self)
