
class Analyses:

  def __init__(self, name=None, replicate=None, blank=None):
    self.name = str(name) if name else None
    self.replicate = str(replicate) if replicate else None
    self.blank = str(blank) if blank else None

  def validate(self):
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
    if not valid:
      print("Issue/s found with analysis", self.name)

  def __str__(self):
    return f"\nAnalysis\n  name: {self.name}\n  replicate: {self.replicate}\n  blank: {self.blank}\n"

  def print(self):
    print(self)
