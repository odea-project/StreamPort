import pandas as pd

class ProcessingSettings:

  def __init__(self, call=None, algorithm=None, parameters=[], version=None, software=None, developer=None, contact=None, link=None, doi=None):
    self.call = str(call)
    self.algorithm = str(algorithm)
    self.parameters = list(parameters)
    self.version = str(version)
    self.software = str(software)
    self.developer = str(developer)
    self.contact = str(contact)
    self.link = str(link)
    self.doi = str(doi)

  def validate(self):
    valid = False
    if isinstance(self.call, str) and isinstance(self.algorithm, str) and isinstance(self.parameters, list) and isinstance(self.version, str):
      valid = True
      if len(self.call) != 1 or not isinstance(self.algorithm, str):
        print("Call entry must be of length 1!")
        valid = False
      if len(self.algorithm) != 1 or not isinstance(self.algorithm, str):
        print("Algorithm entry must be of length 1 and type character!")
        valid = False
      if not isinstance(self.parameters, list):
        print("Parameters entry must be a list or an S4 class!")
        valid = False
      if len(self.version) != 1 or not isinstance(self.version, str):
        print("Version entry must be of length 1 and type character!")
        valid = False
    else:
      print("Settings elements must be named call, algorithm and parameters!")
    return valid

  def __str__(self):
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
          result += "  - " + str(list(self.parameters.keys())[i]) + " (only head rows)" + "\n"
          result += "\n"
          result += str(self.parameters[i].head()) + "\n"
          result += "\n"
        elif isinstance(self.parameters[i], list):
          result += "  - " + str(list(self.parameters.keys())[i]) + ": " + "\n"
          for j in range(len(self.parameters[i])):
            result += "      - " + str(list(self.parameters[i].keys())[j]) + str(self.parameters[i][j]) + "\n"
        elif isinstance(self.parameters[i], function):
          result += "  - " + str(list(self.parameters.keys())[i]) + ":\n"
          result += str(self.parameters[i]) + "\n"
        else:
          result += "  - " + str(list(self.parameters.keys())[i]) + str(self.parameters[i]) + "\n"
    return result

  def print(self):
    print(self)
