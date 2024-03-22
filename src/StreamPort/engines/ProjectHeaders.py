import datetime

class ProjectHeaders:

  def __init__(self, **kwargs):
    defaults = {'name': None, 'author': None, 'path': None, 'date': datetime.datetime.now()}
    self.headers = defaults
    self.headers.update(kwargs)

  def validate(self):
    valid = True
    if self.headers['name'] is None or not isinstance(self.headers['name'], str):
      print("ProjectHeaders entry name must be a non-empty string!")
      valid = False
    if self.headers['author'] is None or not isinstance(self.headers['author'], str):
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

  def __str__(self):
    return f"\nProjectHeaders\n  name: {self.headers['name']}\n  author: {self.headers['author']}\n  path: {self.headers['path']}\n  date: {self.headers['date']}\n"

  def print(self):
    print(self)

  def __getitem__(self, key):
    return self.headers.get(key, None)
