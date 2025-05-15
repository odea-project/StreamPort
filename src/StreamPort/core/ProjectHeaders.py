import datetime

class ProjectHeaders:
  """
  Represents the headers of a project.

  Attributes:
    headers (dict): A dictionary containing the project headers.
      The default keys are 'name', 'author', 'path', and 'date'.

  Methods:
    validate(): Validates the project headers.
    __str__(): Returns a string representation of the project headers.
    print(): Prints the project headers.
    __getitem__(key): Returns the value associated with the given key in the project headers dictionary.
  """

  def __init__(self, **kwargs):
    """
    Initializes a new instance of the ProjectHeaders class.

    Args:
      **kwargs: Keyword arguments representing the project headers.
        The default keys are 'name', 'author', 'path', and 'date'.
    """
    defaults = {'name': None, 'author': None, 'path': None, 'date': datetime.datetime.now()}
    self.headers = defaults
    self.headers.update(kwargs)

  def validate(self):
    """
    Validates the ProjectHeaders object.

    Returns:
      bool: True if the ProjectHeaders object is valid, False otherwise.
    """
    valid = True
    if self.headers['name'] is not None and not isinstance(self.headers['name'], str):
      print("ProjectHeaders entry name must be a non-empty string!")
      valid = False
    if self.headers['author'] is not None and not isinstance(self.headers['author'], str):
      print("ProjectHeaders entry author must be a non-empty string!")
      valid = False
    if self.headers['path'] is not None and not isinstance(self.headers['path'], str):
      print("ProjectHeaders entry path must be a string!")
      valid = False
    if self.headers['date'] is not None and not isinstance(self.headers['date'], datetime.datetime):
      print("ProjectHeaders entry date must be a datetime object!")
      valid = False
    if not valid:
      print("Issue/s found with ProjectHeaders")
    else:
      return True

  def __str__(self):
    """
    Returns a string representation of the project headers.

    Returns:
      str: A string representation of the project headers.
    """
    return "\nProjectHeaders\n" + "\n".join([f"  {key}: {value}" for key, value in self.headers.items()])  
    
  def print(self):
    """
    Prints the project headers.
    """
    print(self)

  def __getitem__(self, key):
    """
    Returns the value associated with the given key in the project headers dictionary.

    Args:
      key (str): The key to retrieve the value for.

    Returns:
      Any: The value associated with the given key, or None if the key is not found.
    """
    return self.headers.get(key, None)
