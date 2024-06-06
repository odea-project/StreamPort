from ..core.CoreEngine import CoreEngine
from ..device.DeviceAnalysis import DeviceAnalysis

#CHECK WHETHER PACKAGES HAVE SUFFICIENT SUPPORT
#Numpy has fft, check also if other stat methods can be implemented without importing tsfresh 

#enable file parsing and OS operations with os module
import os

#datetime to handle date and timestamps associated with data
from datetime import datetime

#regular expressions allow pattern matching to find or filter strings
import re

#chardet is used to extract the encoding of a given file to be handled instead of specifying it explicitly. Adds to program robustness
import chardet

import pandas as pd
import numpy as np


#format (str): Defines the manner in which Datetime objects/strings are to be parsed and/or formatted.
format = '%H:%M:%S %m/%d/%y'

#datetime_pattern (Regex string): Regular Expression used to search for Datetime strings in experiment logs.
datetime_pattern = re.compile(r'(\d{2}:\d{2}:\d{2} \d{1,2}/\d{1,2}/\d{2})')

#file_format (Regex string): Regular expression to enable filtering throught data folders to find pertinent chemstation files
file_format = re.compile(r'^\d{6}_[A-Za-z]+.*? \d{2}-\d{2}-\d{2}$')

class DeviceEngine(CoreEngine):

    """
    DeviceEngine class inherited from CoreEngine class. Handles the actual device-related processing tasks.
    Attributes are declared by passing appropriate class objects to initialization.

    Attr:

    Class Attributes:
        headers (ProjectHeaders, optional): The project headers. Instance of ProjectHeaders class or dict.
        settings (list, optional): The list of settings. Instance or list of instances of ProcessingSettings class.
        analyses (list, optional): The list of analyses. Instance or list of instances of DeviceAnalysis class.
        results (dict, optional): The dictionary of results.
        history (dict, optional): The dictionary of all relational data processed by this object.
        
    Instance Attributes (Unique to each DeviceEngine instance/object):
        _source (raw_str, optional): Path to source directory of data/analysis. 
                                     Can be an individual analysis or parent directory of multiple analyses.
                    Source variable must be a string or raw string or exclude escape characters by using '\\' instad of '\'.
                    User-assigned to each DeviceEngine instance at runtime as an argument.

        _method_ids (list(str)): List of all unique analyses identified by method and date of experiment.


    Methods: (specified are methods only belonging to child class. For superclass methods, see CoreEngine)

        get_analyses(self, analyses(str/int/list(str/int))):
                Overrides superclass method of same name.
                Allows for flexibility in input to find analyses using subwords of identifying strings for analyses(e.g. 'Pac', '08/23', ..).
                Returns only a list of DeviceAnalysis Objects.

        print(self):
                Prints the object and analyses.
        
        find_analyses(self) : 
                Reads and arranges all available (pressure) spectra from a given path to a source directory.
                If source not specified, searches cwd(current working directory) for compatible data.
                Curves are grouped by unique Method ID and date of experiment.

        plot_analyses(self, analyses(str/datetime/list(str/datetime))) : 
                Creates an interactive plot of (pressure) curves against time.
                Unique for unique Method IDs.
                To be expanded to handle other data(e.g. device conditions, curve features).
                (Possible future upgrade)User can specify resolution and/or curve smoothing ratio.

        get_features(self, features_list(str/list, optional)) :
                Extracts features from curves and arranges data appropriately for further processing and ML operations.
                If target features not specified, default list(mean, min, max, std) applied.

        drop_features(self, features_list(str/list of feature/column name(s), optional)) :
                Removes undesired features from prepared dataset. 
                Essentially acts like df.drop(columns), where the columns are features to remove.
                If argument unspecified, defaults to drop all features except identifying features(Date in this case).
    """        
    
    #Private variable for source
    _source = r''

    #list of unique Method IDs 
    _method_ids = []

    

    def __init__(self, headers=None, settings=None, analyses=None, history=None, source =None, method_ids=None):
        
        #initialize superclass attributes for assignment to child class.
        super().__init__(headers, settings, analyses, history)

        #unique variable for each device. Stores full path to the source of data to be imported and processed. 
        self._source = r'{}'.format(source) if not isinstance(source, type(None)) else r'{}'.format(os.getcwd())
        self._method_ids = [] if isinstance(method_ids, type(None)) else method_ids



    def print(self):

        print(self)
        for ana in self._analyses:
            ana.print()



    def find_analyses(self):

        #All instances of DeviceEngine will use the same Datetime methods for analysis.
        #Use predefined global datetime format and search pattern.
        global format 
        global datetime_pattern
        global file_format

        #initialize analysis dictionary to build analysis objects
        analyses_dict = {}

        analyses_list = []

        #function to get encoding of data(UTF-8, UTF-16...) for appropriate applications.
        def get_encoding(datafile):
            
            """
            get_encoding(datafile(raw_str)) :
                Extracts encoding of files to be read for RUN and ACQ data.
                Resolves issues of trying to read files with varying encodings, improving program robustness.
            """

            rawdata = open(datafile, 'rb').read()
            decoding_result = chardet.detect(rawdata)
            character_encoding = decoding_result['encoding']
            return(character_encoding)

        #initialize stack to iterate over all superfolders
        dir_stack = [self._source]

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
                    run_type = r'RUN.LOG'  
                    #acq_type = r'acq.txt' not needed until further updates
                    filetype = r'Pressure.CSV'

                    #individual curves within current method to be later merged into a dataframe against common time feature
                    curves_list = []    

                    #curve headers/names
                    curves = []    

                    #store blank headers
                    blanks = []

                    #store sample headers
                    samples = []

                    #assess number of runs in current directory to set appropriate headers in pressure curve DataFrame
                    num_runs = len(runs_list)

                    #there are expected to be <num_runs> .D folders and 1 .M folder within each experiment folder   
                    suffix_digits = len(str(num_runs))

                    #traverse list of all subdirectories in root/source_dir
                    for d in runs_list:  

                        #path to each encountered folder. Will be traversed based on folder extension
                        current_run_folder = os.path.join(current_folder, d)                   
                        
                        #scan and collect files of type 'Pressure.CSV' from current directory. Expected to return one hit only
                        target_files = [f for f in os.listdir(current_run_folder) if filetype in f] 
                        
                        if not target_files:
                            #(FUTURE)if csv files do not exist, SignalExtraction is called here to create required files 
                            break
                            
                        #Split raw path and extension of folders for further operations
                        pathname, extension = os.path.splitext(current_run_folder)
                        filename = pathname.split('\\')
                            
                        #will act as flag to demarcate blank runs with "-bl" extension in column name
                        blank_identifier = 0    

                        #header suffix to identify type of run. Will be replaced with -bl or -sa based on matrix injected into the instrument
                        run_suffix = ''

                        #choose appropriate header name to identify each run based on currently read folder
                        pressure_suffix = filename[-1].split('--')
                        pressure_suffix = pressure_suffix[-1]

                        method_suffix = filename[-2]

                        #read .log file within each .D folder and extract run_type(blank/sample) for individual runs
                        run_file = os.path.join(current_run_folder, run_type)
                        character_encoding = get_encoding(run_file)

                        #**faulty runs are restarted with same run name after minimum of a day(as seen so far)
                        #**if run within a single .D folder is started and completed more than once, it may indicate an anomaly 
                        times_started = 0

                        with open(run_file, encoding = character_encoding) as f:                                  
                            for line in f:
                                if "blank" in line:
                                    blank_identifier = 1

                                if "Method started" in line:
                                    
                                    times_started = times_started + 1
                                    start_date_string = datetime_pattern.search(line).group()
                                    start_date = datetime.strptime(start_date_string, format)
                                    if not isinstance(start_date, datetime):
                                        start_date = start_date.strftime('%m/%d/%Y %H:%M:%S')

                                elif "Method completed" in line:
                                    
                                    end_date_string = datetime_pattern.search(line).group()
                                    end_date = datetime.strptime(end_date_string, format)
                                    if not isinstance(end_date, datetime):
                                        end_date = end_date.strftime('%m/%d/%Y %H:%M:%S')
                                    
                                    runtime = str(end_date - start_date)
                                    print("Runtime this run : " + runtime)

                            print("Times started(in event of fault) : " + str(times_started))

                        f.close()


                        try:
                            
                            #to account for .D folder without numbering, like 'Irino_kali.D', etc.
                            if not pressure_suffix.isdecimal():                 
                                pressure_suffix = 1
                                suffix_digits = 3
                        
                            #if blank run encountered on reading current run's .LOG file, name run column with '-blank' as identifier
                            if blank_identifier == 1:
                                    
                                run_suffix = '-blank'

                                #add blank run headers to list of blanks
                                curve_header = "Sample - " + (str(pressure_suffix).zfill(suffix_digits) + run_suffix)
                                blanks.append(curve_header)

                            else:

                                #sample runs indicated with sample name
                                run_suffix = filename[-1]
                                    
                                #add sample run headers to samples list, keep trailing pressure suffix for future analysis
                                curve_header = "Sample - " + run_suffix  
                                samples.append(curve_header)

                            curves.append([curve_header, start_date_string])

                            #set current run header based on file number and run type     
                            cols = ["Time", curve_header]

                            #path to 'Pressure.CSV' file found in current directory. Must be only one 
                            target_file = target_files[0]

                            target_file = os.path.join(current_run_folder, target_file)

                            #add pressure curve to list of curves for current method  
                            curves_list.append(pd.read_csv(target_file, 
                                                                sep = ";",
                                                                decimal = ",",
                                                                header = None, 
                                                                names = cols))
                            
                            print(curve_header + ' : \n' + 'start date : ' + start_date_string)
                            print('end date : ' + end_date_string)
                            print('runtime : ' + runtime + '\n')
                                            
                        except FileNotFoundError:

                            print("File doesn't exist")

                        finally:

                            if len(curves_list) > 1:
                                            
                                curves_list[0] = pd.merge(curves_list[0], 
                                                                curves_list[-1], 
                                                                on = 'Time')

                        #experiment date for analysis name. Name is name from Analyses object             
                        analysis_name = "Analysis - " + method_suffix

                        if curve_header:
                            #if pressure curve exists for current analysis, mention in analysis key
                            analysis_key = "Device Pressure Analysis - " + start_date_string
                        
                        else:
                            analysis_key = "Analysis - " + start_date_string

                        analysis_data = {'Method' : method_suffix, 
                                         'Sample' : curve_header, 
                                         'Start date' : start_date_string, 
                                         'Runtime' : runtime, 
                                         'Time since last flush' : "NA",
                                         'Number of Trials' : times_started, 
                                         'Curve' : curves_list[-1]}

                        #build dictionary of analyses per unique method or date of data import
                        analysis = {analysis_key : analysis_data}

                        #analysis data. single item of data attribute of analyses object
                        analyses_dict.update(analysis)

                        #add every encountered analysis to device history
                        self._history.update(analysis)    

                        #dataframe for current identifier(method_suffix and date)
                        merged_df = curves_list[0]

                        #finally add complete dataframe for given method and date to analysis object's data attribute. 
                        #This completes the analysis object.
                        analyses_dict.update({'Pressure Dataframe' : merged_df})

                    #list of analyses populated with individual analysis objects
                    analyses_list.append(DeviceAnalysis(name = analysis_name, data = analyses_dict))

                if not method_suffix:
                    continue

                if method_suffix not in self._method_ids:

                    #add current method ID to list of known/encountered method types, no duplicates
                    self._method_ids.append(method_suffix)

                #update experiments dict with new set of runs
                #self._history.update({method_suffix : merged_df})


                print("Dataframe for method : " + method_suffix)
                print(merged_df.head()) 

        return analyses_list



    def get_analyses(self, analyses):

        result = []

        if analyses is not None:

            if isinstance(analyses, int) and analyses < len(self._analyses):
                result.append(self._analyses[analyses])
            
            elif isinstance(analyses, str):
                for analysis in self._analyses:
                    if analyses in analysis.name:
                        result.append(analysis)
                    
            elif isinstance(analyses, list):
                analyses_out = []
                for analysis in analyses:
                    if isinstance(analysis, int) and analysis < len(self._analyses):
                        analyses_out.append(self._analyses[analysis])
                    elif isinstance(analysis, str):
                        for a in self._analyses:
                            if analysis in a.name:
                                analyses_out.append(a)
                result = analyses_out
            
            else:
                print("Analysis not found!")

            return result

        else:

            print("Provided data is not sufficient or does not exist! Existing analyses will be returned.")
            return self._analyses
        


    def plot_analyses(self, analyses):

        if not isinstance(analyses, type(None)):

            curves_to_plot = self.get_analyses(analyses)
            for ana in curves_to_plot:
                ana.plot()

        else:

            print("No analyses found!\nAdd new analyses to enable plotting")



    def get_features(self, data, features_list):
        #processing settings come in here. 
        #Settings are decided in DeviceProcSettings and passed as data and features_list

        extracted_features = data.iloc[:, 1:].agg(features_list)
        print(extracted_features)
        return(extracted_features)



    def drop_features(self, features_list):

        return()
    

