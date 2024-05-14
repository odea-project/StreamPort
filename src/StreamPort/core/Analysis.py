import numpy as np

class Analysis:
  """
  Represents an analysis object.

  Attributes:
    name (str): The name of the analysis.
    replicate (str): The name of the replicate.
    blank (str): The name of the blank.
    data (list): The data of the analysis, which is a dict of one dimension numpy arrays.

  Methods:
    validate(): Validates the analysis object.
    print(): Prints the analysis object.
  """

  def __init__(self, name=None, replicate=None, blank=None, data=None):
    self.name = str(name) if name else None
    self.replicate = str(replicate) if replicate else None
    self.blank = str(blank) if blank else None
    self.data = dict(data) if data else {}

  def validate(self):
    """
    Validates the analysis object.

    Prints an error message if any of the attributes are not of type str.
    """
    valid = True
    if not isinstance(self.name, str):
      print("Analysis name not conform!")
      valid = False
    if not isinstance(self.replicate, str):
      print("Analysis replicate name not conform!")
      valid = False
    if not isinstance(self.blank, str):
      print("Analysis blank name not conform!")
      valid = False
    if isinstance(self.data, dict):
      if self.data != {}:
        for key in self.data:
          if not isinstance(self.data[key], np.ndarray):
            print("Analysis data not conform!")
            valid = False
        if len(set([self.data[key].ndim for key in self.data])) != 1:
          print("Analysis data not conform!")
          valid = False
        if len(set([len(self.data[key]) for key in self.data])) != 1:
          print("Analysis data not conform!")
          valid = False
    else:
      print("Analysis data not conform!")
      valid = False
    if not valid:
      print("Issue/s found with analysis", self.name)

  def __str__(self):
    """
    Returns a string representation of the analysis object.

    Returns:
      str: A string representation of the analysis object.
    """
    if self.data == {}:
      data_str = "  Empty"
    else:
      data_str = '\n'.join([f"    {key} (size {len(self.data[key])})" for key in self.data])
    
    return f"\nAnalysis\n  name: {self.name}\n  replicate: {self.replicate}\n  blank: {self.blank}\n  data:\n{data_str}\n"

  def print(self):
    """
    Prints the analysis object.
    """
    print(self)
