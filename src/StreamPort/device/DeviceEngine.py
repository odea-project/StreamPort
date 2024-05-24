from ..core.CoreEngine import CoreEngine
from ..device.DeviceAnalysis import DeviceAnalysis

import os
from datetime import datetime
import re
import chardet
import pandas as pd
import numpy as np

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
        
    Instance Attributes (Unique to each DeviceEngine instance/object):
        _source (raw_str, optional): Path to source directory of data/analysis. 
                                     Can be an individual analysis or parent directory of multiple analyses.
                    Source variable must be a string or raw string or exclude escape characters by using '\\' instad of '\'.

    Methods:

        read_device_spectra(self, source(raw_str, optional)) : 
                Reads and arranges all available (pressure) spectra from a given path to a source directory.
                If source not specified, searches cwd(current working directory) for compatible data.
                Curves are grouped by unique Method ID and date of experiment.

        get_encoding(datafile(raw_str)) :
                Extracts encoding of files to be read for RUN and ACQ data.
                Resolves issues of trying to read files with varying encodings, improving program robustness.

        plot_spectra(self) : 
                Creates an interactive plot of (pressure) curves against time.
                Unique for unique Method IDs.
                To be expanded to handle other data(e.g. device conditions, curve features).
                (Possible future upgrade)User can specify resolution and/or curve smoothing ratio.

        add_features(self, features_list(str/list, optional)) :
                Extracts features from curves and arranges data appropriately for further processing and ML operations.
                If target features not specified, default list(mean, min, max, std) applied.

        get_features(self) :
                Returns/prints result dataset of extracted features over all samples.

        drop_features(self, features_list(str/list of feature/column name(s), optional)) :
                Removes undesired features from prepared dataset. 
                Essentially acts like df.drop(columns), where the columns are features to remove.
                If argument unspecified, defaults to drop all features except identifying features(Date in this case).
    """         

    #Private variable for source
    _source = r''

    #list of unique Method IDs 
    method_ids = []

    #dictionary of all recorded experiments for current DeviceEngine object. Can be (possibly) replaced by super()._history
    experiments = {}

    def __init__(self, headers=None, settings=None, analyses=None, results=None, source =None):
        
        #initialize superclass attributes for assignment to child class.
        super().__init__(headers, settings, analyses, results)

        #unique variable for each device. Stores full path to the source of data to be imported and processed. 
        self._source = r'{}'.format(source) if not isinstance(source, type(None)) else r'{}'.format(os.getcwd())
        self.method_ids = []
        self.experiments = {}

    def read_device_spectra(self):

        #All instances of DeviceEngine will use the same Datetime methods for analysis.
        #format (str): Defines the manner in which Datetime objects/strings are to be parsed and/or formatted.
        format = '%H:%M:%S %m/%d/%y'

        #datetime_pattern (Regex string): Regular Expression used to search for Datetime strings in experiment logs.
        datetime_pattern = re.compile(r'(\d{2}:\d{2}:\d{2} \d{1,2}/\d{1,2}/\d{2})')

        #initialize 1-D analysis dictionary to build analysis objects
        analyses_dict = {}

        #list of analysis objects to pass to DeviceAnalysis
        analyses_list = []

        #function to get encoding of data(UTF-8, UTF-16...) for appropriate applications.
        def get_encoding(datafile):

            rawdata = open(datafile, 'rb').read()
            decoding_result = chardet.detect(rawdata)
            character_encoding = decoding_result['encoding']
            return(character_encoding)

        #initialize stack to iterate over all superfolders
        dir_stack = [self._source]

        while dir_stack:

            #start with the latest folder encountered
            current_folder = dir_stack.pop()

            #check if superfolder. Directories ending in ' ' can only be superfolders.
            sources_list = []
            runs_list = []
            for file in os.listdir(current_folder):
                #Split raw path and extension of folders for further operations
                pathname, extension = os.path.splitext(os.path.join(current_folder, file))
                filename = pathname.split('\\')
                if extension == '':
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
                    acq_type = r'acq.txt'
                    filetype = r'Pressure.CSV'

                    #individual curves within current method to be later merged into a dataframe against common time feature
                    curves_list = []    

                    #curve headers/names
                    curves = []    

                    #will later hold "Time - method_id on which all curves under a method will be merged into one dataframe"
                    merge_target = ''

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
                            break
                            
                        #Split raw path and extension of folders for further operations
                        pathname, extension = os.path.splitext(current_run_folder)
                        filename = pathname.split('\\')
                            
                        #will act as flag to demarcate blank runs with "-bl" extension in column name
                        blank_identifier = 0    

                        #header suffix to identify type of run. Will be replaced with -bl or -sa based on matrix injected into the instrument
                        run_suffix = ''

                        #choose appropriate header name to identify each run based on currently read folder
                        pressure_suffix = filename[-1][-2:]


                        #read acq.txt file within each .D folder and extract method name(and other run info later)
                        acq_file = os.path.join(current_run_folder, acq_type)
                        character_encoding = get_encoding(acq_file)

                        with open(acq_file, encoding = character_encoding) as a:                                  
                            for line in a:
                                if "Acq. Method" in line:
                                    line = line.split(' ')[-1]
                                    line = line.split('.')
                                    method_suffix = line[0]
                                    break
                        a.close()


                        #read .log file within each .D folder and extract run_type(blank/sample) for individual runs
                        run_file = os.path.join(current_run_folder, run_type)
                        character_encoding = get_encoding(run_file)

                        with open(run_file, encoding = character_encoding) as f:                                  
                            for line in f:
                                if "blank" in line:
                                    blank_identifier = 1

                                if "Method started" in line:
                    
                                    start_date_string = datetime_pattern.search(line).group()
                                    start_date = datetime.strptime(start_date_string, format)
                                    if not isinstance(start_date, datetime):
                                        start_date = start_date.strftime('%m/%d/%Y %H:%M:%S')

                                elif "Instrument run completed" in line:
                                    
                                    end_date_string = datetime_pattern.search(line).group()
                                    end_date = datetime.strptime(end_date_string, format)
                                    if not isinstance(end_date, datetime):
                                        end_date = end_date.strftime('%m/%d/%Y %H:%M:%S')
                                    
                            runtime = str(end_date - start_date)

                            print('start date : ' + start_date_string)
                            print('end date : ' + end_date_string)
                            print('runtime : ' + runtime)
                            
                        f.close()

                        try:
                            
                            #to account for .D folder without numbering, like 'Irino_kali.D', etc.
                            if not pressure_suffix.isdecimal():                 
                                pressure_suffix = 1
                        
                            #if blank run encountered on reading current run's .LOG file, name run column with '-bl' as identifier
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
                                                                header = None, names = cols))
                                            
                        except FileNotFoundError:

                            print("File doesn't exist")

                        finally:

                            if len(curves_list) > 1:
                                            
                                curves_list[0] = pd.merge(curves_list[0], 
                                                                curves_list[-1], 
                                                                on = 'Time')
                                
                        #experiment date for analysis name. Name is name from Analyses object             
                        analysis_name = "Analysis - " + method_suffix + " " + start_date_string

                        if curve_header:
                            #if pressure curve exists for current analysis, mention in analysis key
                            analysis_key = "Device Pressure Analysis - " + start_date_string
                        
                        else:
                            analysis_key = "Analysis - " + start_date_string

                        analysis_data = np.array(['Method : ' + method_suffix, 'Sample : ' + curve_header, 'Start date : ' + start_date_string, 'Runtime : ' + runtime, 'Time since last flush : ' + "NA"])

                        #individual dictionary entry for an analysis object
                        analysis = {analysis_key : analysis_data}

                        #analysis data. data attribute of analyses object
                        analyses_dict.update(analysis)
                        
                        #list of analyses objects
                        analyses_list.append(DeviceAnalysis(name = analysis_name, data = analysis))

                    merged_df = curves_list[0]

                    merge_target = "Time - " + method_suffix

                    merged_df.rename(columns = {'Time' : merge_target}, inplace = True)

                if not method_suffix:
                    continue

                if method_suffix not in self.method_ids:

                    #add current method ID to list of known/encountered method types, no duplicates
                    self.method_ids.append(method_suffix)
                    self.experiments.update({method_suffix : merged_df})

                else:

                    #merge dfs with the same method ID
                    existing_df = self.experiments[method_suffix]
                    existing_curves = len(existing_df.columns[1:])

                    new_curves = merged_df.columns[1:]
                    num_new_curves = len(new_curves)

                    updated_curves = []

                    for i in range(1, num_new_curves + 1):
                        updated_curves.append(new_curves[i-1] + '_' + str( i + existing_curves))

                    updated_curves.insert(0, merge_target)
                    merged_df.columns = updated_curves

                    updated_method = pd.merge(existing_df,                                           
                                            merged_df,                                               
                                            on = merge_target)
                    
                    self.experiments.update({method_suffix : updated_method}) 

                print("Dataframe for method : " + method_suffix)
                print(merged_df.head()) 

        return(analyses_list)

    def plot_spectra(self):

        return()
    


    def add_features(self, features_list):

        return()
    


    def get_features(self):

        return()
    


    def drop_features(self, features_list):

        return()