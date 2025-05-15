from src.StreamPort.device.DeviceEngine import _2DEngine, TimeSeriesEngine
from .ProjectHeaders import ProjectHeaders
from .ProcessingSettings import ProcessingSettings


"""
Modules for data collection:
To be used by the Selector object to handle data.
"""

#enable file parsing and OS operations with os module
import os

#datetime to handle date and timestamps associated with data
from datetime import datetime

#regular expressions allow pattern matching to find or filter strings
import re

#chardet is used to extract the encoding of a given file to be handled instead of specifying it explicitly. Adds to program robustness
import chardet

#csv allows to inspect input data without reading it. Once the data is read for type, it is passed to Engine for further operations
import csv


"""
GLOBAL VARIABLES:
To be used by every Selector object to handle data. Data across devices and methods will use the same global variables
"""

#format (str): Defines the manner in which Datetime objects/strings are to be parsed and/or formatted.
format = '%H:%M:%S %m/%d/%y'

#datetime_pattern (Regex string): Regular Expression used to search for Datetime strings in experiment logs.
datetime_pattern = re.compile(r'(\d{2}:\d{2}:\d{2} \d{1,2}/\d{1,2}/\d{2})')

#file_format (Regex string): Regular expression to enable filtering through data folders to find pertinent chemstation files
file_format = re.compile(r'^\d{6}_[A-Za-z]+.*? \d{2}-\d{2}-\d{2}$')


class Selector:
    """
    The Selector class functions as a bridge between data collection and data processing. 
    It initializes the remaining classes of the StreamPort application with: 
        - details on the data type of the input stream being handled
        - appropriate ProcessingSettings for the data
        - relevant Header information for the initialization of the Engine (dtype)
    
    Attributes:
        _source (raw string): 
                    Path to source directory of data/analysis. 
                    Can be an individual analysis or parent directory of multiple analyses.
                    Source variable must be a string or raw string or exclude escape characters by using '\\' instad of '\'.
                    User-assigned to each Selector instance at runtime as an argument.
        _dtype (string): The project headers. Currently restricted to '2-D' and 'Matrix'
        _settings (ProcessingSettings): The settings to be applied.
        _results (dict): The dictionary of results.
        _history (dict): The dictionary of history.
        * _history will contain all observed input data while _results will include the datatype that was chosen 

    Methods:
        __init__(self, dtype=None, settings=None): Initializes the Selector instance.

    """


    def __init__(self, source = None, dtype = None, settings = None):
        #unique variable for each device. Stores full path to the source of data to be imported and processed. 
        self._source = r'{}'.format(source) if not isinstance(source, type(None)) else r'{}'.format(os.getcwd())
        self._dtype = '' if not isinstance(dtype, str) else dtype
        self._settings = self.assign_settings() if isinstance(settings, type(None)) else settings
   

    def inspect_csv(self):
        data_path = self._source

        if os.path.isfile(data_path):
            encoding = self.get_encoding(data_path)
            with open(data_path, mode='r', newline='', encoding=encoding) as file:
                # Create a CSV reader object
                reader = csv.reader(file)
                
                # Get the first row (header)
                header = next(reader, None)
                
                if header:
                    print(f"Number of columns: {len(header)}")
                    print(f"Column names: {header}")
                    if len(header) <= 2:
                        bidimensional.append(data_path)
                    else:
                        matrices.append(data_path) 

                else:
                    print("File is empty or has no headers.")
                
                # Inspect data
                num_rows = sum(1 for row in reader)
                print(f"Number of rows: {num_rows}")

        elif os.path.isdir(data_path):
            #initialize stack to iterate over all superfolders
            dir_stack = [data_path]

            while dir_stack:

                #start with the latest folder encountered
                current_folder = dir_stack.pop()

                #add additional checks to ensure that current working directory includes chemstation or pertinent data and does not affect other data.

                #check if superfolder. Directories ending in ' ' can only be superfolders.
                sources_list = []
                runs_list = []
                for file in os.listdir(current_folder):
                    #Split raw path and extension of folders for further operations
                    pathname, extension = os.path.splitext(os.path.join(current_folder, file))

                    if extension == '' and re.match(file_format, str(file)):
                        sources_list.append(file)

                    elif extension == '.D':
                        runs_list.append(file)    
                    

                if sources_list:

                    #add all superfolders to stack for iteration
                    for subfolder in sources_list:
                        dir_stack.append(os.path.join(current_folder, subfolder))

                #if not a superfolder, check whether we are not already within a .D folder   
                else:
                    if not runs_list:
                        os.chdir('..')
                        current_folder = os.getcwd()                                                                             
                        if current_folder in dir_stack:
                            pass

                        else:
                            dir_stack.append(current_folder)
                        continue

                    else:
                        #type of files to read withing each .D folder
                        runtype = r'RUN.LOG'  
                        #acq_type = r'acq.txt' not needed until further updates
                        filetype = r'Pressure.CSV'                        
                        #add latest set of related runs saved as analyses_list
                        final_analyses_list.extend(analyses_list)

            return final_analyses_list


    #function to get encoding of data(UTF-8, UTF-16...) for appropriate applications.
    def get_encoding(self, datafile):
        
        """
        get_encoding(datafile(raw_str)) :
            Extracts encoding of files to be read for RUN and ACQ data.
            Resolves issues of trying to read files with varying encodings, improving program robustness.
        """
        datafile = self._source if isinstance(datafile, type(None)) else datafile
        rawdata = open(datafile, 'rb').read()
        decoding_result = chardet.detect(rawdata)
        character_encoding = decoding_result['encoding']
        rawdata.close()
        return(character_encoding)


    

        
