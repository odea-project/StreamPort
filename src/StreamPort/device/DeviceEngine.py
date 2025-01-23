from src.StreamPort.core.CoreEngine import CoreEngine
from src.StreamPort.device.DeviceAnalysis import DeviceAnalysis
from src.StreamPort.ml.MachineLearningAnalysis import MachineLearningAnalysis
from src.StreamPort.ml.MachineLearningEngine import MachineLearningEngine

#CHECK WHETHER PACKAGES HAVE SUFFICIENT SUPPORT

#enable file parsing and OS operations with os module
import os

#datetime to handle date and timestamps associated with data
from datetime import datetime

#regular expressions allow pattern matching to find or filter strings
import re

#chardet is used to extract the encoding of a given file to be handled instead of specifying it explicitly. Adds to program robustness
import chardet

#import modules to handle data
import pandas as pd
import numpy as np

#module to enable time decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

#import ML packages
#to scale data
from sklearn import preprocessing as scaler



"""
GLOBAL VARIABLES:
To be used by every DeviceEngine object to handle data. Data across devices and methods will use the same global variables
"""

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
        results (dict, optional): The dictionary of results computed using varying Processing Settings.
        history (dict, optional): The dictionary of all relational data processed by this object.
        
    Instance Attributes (Unique to each DeviceEngine instance/object):
        _source (raw_str, optional): Path to source directory of data/analysis. 
                                     Can be an individual analysis or parent directory of multiple analyses.
                    Source variable must be a string or raw string or exclude escape characters by using '\\' instad of '\'.
                    User-assigned to each DeviceEngine instance at runtime as an argument.

        _method_ids (dict(tup(datetime))): Dict of all unique analyses (identified by method and date of experiment)
                                            and associated value-pair of (start date, end date) for the respective batch.

        _components(str/list(str)): List of all components involved in runs on this device.

    Methods: (specified are methods only belonging to child class or modified to fit child class specifications. For superclass methods, see CoreEngine)

        print(self):
                Prints the object and analyses.
        
        find_analyses(self) : 
                Reads and arranges all available (pressure) spectra from a given path to a source directory.
                If source not specified, searches cwd(current working directory) for compatible data.
                Curves are grouped by unique Method ID and date of experiment.

        get_analyses(self, analyses(str/int/list(str/int))):
                Overrides superclass method of same name.
                Allows for flexibility in input to find analyses using subwords of identifying strings for analyses(e.g. 'Pac', '08/23', ..).
                Returns only a list of DeviceAnalysis Objects.
                
        plot_analyses(self, analyses(str/datetime/list(str/datetime))) : 
                Creates an interactive plot of (pressure) curves against time.
                Unique for unique Method IDs.
                To be expanded to handle other data(e.g. device conditions, curve features).
                (Possible future upgrade)User can specify resolution and/or curve smoothing ratio.

        get_features(self, data(dict), features_list(list(str), optional)) :
                Extracts features from (pressure) curves and arranges data appropriately for further processing and ML operations.
                If target features not specified, default list(mean, min, max, std, skew, kurtosis) applied.

        get_seasonal_components(self, data(dict)) :
                Breaks (pressure) curves down into their respective trend, seasonal and residual components for further analysis.
                        
        drop_features(self, features_list(str/list of feature/column name(s), optional)) :
                Removes undesired features from prepared dataset. 
                Essentially acts like df.drop(columns), where the columns are features to remove.
                If argument unspecified, defaults to drop all features except identifying features(Date in this case).
    """        
    
    #Private variable for source
    _source = r''

    #list of unique Method IDs 
    _method_ids = {}

    #list of unique components
    _components = {}

    #list of analysis data_keys
    _data_keyList = []

    def __init__(self, headers=None, settings=None, analyses=None, history=None, source =None, method_ids=None, components=None):
        
        #initialize superclass attributes for assignment to child class.
        super().__init__(headers, settings, analyses, history)

        #unique variable for each device. Stores full path to the source of data to be imported and processed. 
        self._source = r'{}'.format(source) if not isinstance(source, type(None)) else r'{}'.format(os.getcwd())
        self._method_ids = {} if isinstance(method_ids, type(None)) else method_ids
        self._components = {} if isinstance(components, type(None)) else components
        self._data_keyList = [] 


    def print(self):

        print(self)
        for ana in self._analyses:
            ana.print()



    def find_analyses(self):
        """
        Searches for analyses in the given source or current working directory if source is not provided.
        Prepares the available analyses and return a list of analyses objects that conform to data processing format.

        """
        #All instances of DeviceEngine will use the same Datetime methods for analysis.
        #Use predefined global datetime format and search pattern.
        global format 
        global datetime_pattern
        global file_format

        final_analyses_list = []

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

        #default last end date for first recorded batch. Expects file containing experiment batches to be sorted in ascending order of start dates
        last_known_end_date = datetime(1,1,1,0,0,0,0)
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

                    #individual curves within current method to be later merged into a dataframe against common time feature
                    curves_list = []    

                    #assess number of runs in current directory to set appropriate headers in pressure curve DataFrame
                    num_runs = len(runs_list)

                    #there are expected to be <num_runs> .D folders and 1 .M folder within each experiment folder   
                    suffix_digits = len(str(num_runs))

                    #preliminal list to carry objects of all runs in one directory idetified by method name and date
                    analyses_list = []

                    current_run_of_batch = datetime.now()
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

                        #read .log file within each .D folder and extract runtype(blank/sample) for individual runs
                        run_file = os.path.join(current_run_folder, runtype)
                        character_encoding = get_encoding(run_file)

                        """
                        #**faulty runs are restarted with same run name after minimum of a day(as seen so far)
                        #**if run within a single .D folder is started and completed more than once, it may indicate an anomaly 

                        """
                        times_started = 0
                        log_data = []
                        with open(run_file, encoding = character_encoding) as f:                                  
                            component_number = 0
                            comp_ids = []
                            found_postrun = 0
                            for line in f:
                                log_data.append(line)
                     
                                #get component data(detector, ppump, sampler...)
                                component_name = 'Detector'
                                if line[0] == 'G':
                                    component_id = line.split(' ')[0]
                                    if 'Detector' in line:
                                        component_name = 'Detector'
                                    #elif
                            #check log data
                                    components = list(self._components.keys())
                                    if component_id not in components:
                                        self._components.update({component_name : component_id})
                                    if component_id not in comp_ids:
                                        component_number = component_number + 1
                                        comp_ids.append(component_id)

                                if "blank" in line:
                                    blank_identifier = 1

                                if "Run" in line:
                                    
                                    times_started = times_started + 1
                                    start_date_string = datetime_pattern.search(line).group()
                                    start_date = datetime.strptime(start_date_string, format)
                                    if not isinstance(start_date, datetime):
                                        start_date = start_date.strftime('%m/%d/%Y %H:%M:%S')

                                elif "Postrun" in line and found_postrun != 1:
                                    found_postrun = 1
                                    end_date_string = datetime_pattern.search(line).group()
                                    end_date = datetime.strptime(end_date_string, format)
                                    if not isinstance(end_date, datetime):
                                        end_date = end_date.strftime('%m/%d/%Y %H:%M:%S')
                                    
                                    runtime = str(end_date - start_date)

                                    #convert runtime string into absolute number of seconds
                                    runtime = datetime.strptime(runtime, '%H:%M:%S')
                                    runtime = runtime.second + runtime.minute * 60 + runtime.hour * 3600
                                    
                            print("Times started(in event of fault) : " + str(times_started))

                        f.close()

                        print('This files logs')
                        print(log_data)

                        try:
                            
                            #to account for .D folder without numbering, like 'Irino_kali.D', etc.
                            #Batch idle time for first analysis duplicated for every analysis of that (untrimmed) method
                            #first batch has 0 idle time
                            idle_time = 0
                            if not pressure_suffix.isdecimal():                 
                                pressure_suffix = 1
                                suffix_digits = 3
                                class_label = 'First measurement'
                                #start date of first run 001 is marked. 
                                # This also indicates that the current batch is complete and the last recorded end_date is the end date of the final run of the batch. 
                                first_run_of_batch = start_date
                                last_run_of_batch = current_run_of_batch

                                if last_known_end_date != datetime(1,1,1,0,0,0,0):
                                    idle_time = first_run_of_batch - last_known_end_date
                                    # Get the total number of seconds from the absolute value of the timedelta
                                    idle_time = abs(idle_time).total_seconds()
                                
                            else:
                                #cycle through end dates of all previous analyses. Last end date saved in variable will be the last run recorded before 001.
                                # Confirm correct time stamp, 'postrun' or 'method completed'       
                                current_run_of_batch = end_date
                                class_label = None
                        
                            #if blank run encountered on reading current run's .LOG file, name run column with '-blank' as identifier
                            if blank_identifier == 1:
                                    
                                run_suffix = '-blank'

                                #assign runtype 0 to blanks                               
                                run_type = 0 

                                #add blank run headers to list of blanks
                                curve_header = "Sample - " + (str(pressure_suffix).zfill(suffix_digits) + run_suffix)     

                            else:

                                #assign runtype 1 to samples 
                                run_type = 1

                                #add sample run headers to samples list, keep trailing pressure suffix for future analysis
                                #sample runs indicated with sample name
                                curve_header = "Sample - " + filename[-1]  
                            
                            #sample name/header holds its start date and time
                            curve_header = curve_header + '|' + start_date_string

                            #run-number from filename is used as batch position ('.D' = 1, '002.D' = 2, '003.D' = 3...)    
                            batch_position = pressure_suffix

                            #set current run header based on file number and run type     
                            cols = ["Time", curve_header]

                            #path to 'Pressure.CSV' file found in current directory. Must be only one 
                            target_file = target_files[0]

                            target_file = os.path.join(current_run_folder, target_file)

                            """
                            align_data handles data using varying decimal identifiers('.' or ',') 
                            """
                            def align_data(file_path):
                                decimal = '.'
                                with open(file_path, 'r') as file:
                                    first_line = file.readline()
                                    if ',' in first_line:
                                        decimal = ','                                    
                                    
                                file.close()

                                return decimal

                            decimal = align_data(target_file)        

                            pressure_file = pd.read_csv(target_file, 
                                                                sep = ";",
                                                                decimal = decimal,
                                                                header = None, 
                                                                names = cols)
                            
                            curve_runtime = (int(pressure_file['Time'].max()) - int(pressure_file['Time'].min())) * 60 # convert to seconds

                            #add pressure curve to list of curves for current method  
                            curves_list.append(pressure_file)
                            
                            print(curve_header)
                            print('end date : ' + end_date_string)
                            print('runtime : ' + str(runtime))
                            print('runtime observed from curve : ' + str(curve_runtime))
                            print('batch idle time : ' + str(idle_time))

                            #find deviation between true and observed runtimes 
                            runtime_delta = runtime - curve_runtime

                            #find percent deviation between true and observed runtimes
                            runtime_percent_error = (abs(runtime_delta) / curve_runtime) * 100

                        except FileNotFoundError:
                            
                            """
                            This area is to be extended to handle other device data(Temp, Actual Logs).
                            Future implementation includes calling SignalExtraction when the required csv files are absent.
                            """
                            print(f"File {target_file} doesn't exist")

                        finally:

                            if len(curves_list) > 1:
                                            
                                curves_list[0] = pd.merge(curves_list[0], 
                                                                curves_list[-1], 
                                                                on = 'Time')

                        if curve_header:
                            #if pressure curve exists for current analysis, mention in analysis key
                            #experiment date and method ID as identifier for analysis name. Name is name from Analyses object             
                            analysis_name = f"Device Pressure Analysis - {method_suffix}| Start time: {start_date_string}"
                            analysis_type = 'pressure'

                        else:
                            #individual analyses(runs) in data are identified by their start date.
                            #The last successful start date is the final start date.
                            analysis_name = f"Analysis - {method_suffix} | Start time: {start_date_string}"

                        #each 'Device Pressure Analysis' dict like below refers to an individual run within a set of consecutive runs for a particular method on a particular date.
                        #This will be updated during operations concerning the particular runs such as feature extraction.
                        #the larger dict holding these individual dicts is the data of a single DeviceAnalysis object.
                        analysis_data = {'Method' : method_suffix, 
                                         'Sample' : curve_header, 
                                         'Curve' : curves_list[-1], 
                                         'Runtype' : run_type,
                                         'Batch position' : batch_position,
                                         'Runtime' : runtime,
                                         'Runtime delta' :  runtime_delta,
                                         'Runtime percent error' : runtime_percent_error,
                                         'Idle time' : idle_time,
                                         'Number of Trials' : times_started, 
                                         'Log' : log_data}
                        
                        
                        for index in range(component_number):
                                analysis_data.update({f'Component {index+1}' : comp_ids[index]}) 

                        data_keyList = list(analysis_data.keys())

                        #add every encountered analysis to device history
                        self._history.update({analysis_name : analysis_data})    

                        #make idle times conform across related analyses grouped by method name and date
                        if idle_time != 0:#that means its sample 1 blank
                            for ana in analyses_list:
                                if ana.data['Idle time'] == 0:
                                    ana.data['Idle time'] = idle_time

                        #create new DeviceAnalysis object.
                        new_object = DeviceAnalysis(name = analysis_name, data = analysis_data, analysis_type = analysis_type, class_label=class_label, key_list=data_keyList)

                        #list of analyses populated with individual analysis objects
                        analyses_list.append(new_object)

                    #add latest set of related runs saved as analyses_list
                    final_analyses_list.extend(analyses_list)

                if not method_suffix:
                    continue
                
                method_ids = list(self._method_ids.keys())
                if method_suffix not in method_ids:
                    if method_ids != []:
                        last_known_end_date = self._method_ids[method_ids[-1]][-1]
                        
                    #add current method ID to list of known/encountered method types, no duplicates
                    self._method_ids.update({method_suffix : (first_run_of_batch, last_run_of_batch)})

        return final_analyses_list



    def get_analyses(self, analyses=None):
        """
        Identical superclass method is modified here to return only a list, 
        and also to return any value with matching subwords rather than the exact key.

        """
        ana_list = self._analyses

        if analyses != None:
            ana_list = []
            if isinstance(analyses, int) and analyses < len(self._analyses):
                ana_list.append(self._analyses[analyses])
            
            elif isinstance(analyses, str):
                for analysis in self._analyses:
                    if analyses in analysis.name:
                        ana_list.append(analysis)
            
            elif isinstance(analyses, DeviceAnalysis):
                ana_list.append(analyses)
                    
            elif isinstance(analyses, list):
                analyses_out = []
                for analysis in analyses:
                    if isinstance(analysis, int) and analysis < len(self._analyses):
                        analyses_out.append(self._analyses[analysis])
                    elif isinstance(analysis, str):
                        for a in self._analyses:
                            if analysis in a.name:
                                analyses_out.append(a)
                    elif isinstance(analysis, DeviceAnalysis):
                        analyses_out.append(analysis)
                ana_list = analyses_out
            
            else:
                print("Analysis not found!")

        else:
            print("Provided data is not sufficient or does not exist! Existing analyses will be returned.")
            
        return ana_list        



    def get_features(self, data, features_list, weighted):
        """
        #Settings are decided in DeviceProcSettings and passed as data(dict) and features_list(list(str)) and weighted(bool).
        #additional features that complement information from given set of features are runtime and runtype/class label.
        
        """
        weighted = weighted if isinstance(weighted, bool) else False

        #runtime of each sample indicates possible faults with the run
        runtime = {}
        runtime_delta = {}
        runtime_percent_error = {}
        idle_time = {}

        #runtype describes whether run was a blank or a sample run
        runtype = {}

        #batch position and component features.
        bpos = {}
        #comp_features = {}

        #update each Device Pressure Analysis with its features 
        curve = data['Curve']
        sample = data['Sample']

        #features extracted from 'pct_change' curves hepl better model curve behaviour.
        curve_features = curve.iloc[:, 1].agg(features_list)

        #'pct_change' transformation is first used(optional) on pressure curves to emphasise focus on changes in the curve over time.        
        if weighted == True:    
            weighted_curve_features = (curve.iloc[:, 1].agg('pct_change')*100).replace([np.inf, -np.inf], np.nan)
            weighted_curve_features = (weighted_curve_features.fillna(0)).agg(features_list)
            weighted_curve_features.index = [f"{i}_percent_change" for i in weighted_curve_features.index]
            curve_features = weighted_curve_features

        #add additional run features         
        runtime.update({sample : data['Runtime']})
        runtime_delta.update({sample : data['Runtime delta']})
        runtime_percent_error.update({sample : data['Runtime percent error']})
        runtype.update({sample : data['Runtype']})
        bpos.update({sample : data['Batch position']})
        idle_time.update({sample : data['Idle time']})

        """
        logs = data['Log']
        
        component_number = 0
        comp_ids = []
        #get component data(detector, pump, sampler...)
        for line in logs:
            if line[0] == 'G':
                component_id = line.split(' ')[0]
                if component_id not in comp_ids:
                    component_number = component_number + 1
                    comp_ids.append(component_id)
        for index in range(component_number):
            comp_features.update({f'Component {index+1}' : comp_ids[index]}) 

        """
            
        run_features = pd.DataFrame([runtime, runtime_delta, runtime_percent_error, runtype, bpos, idle_time], 
                                    columns=runtype.keys())


        run_features.index = ['Runtime(Rt)', 'RtDelta', 'RtPctError', 'Runtype', 'BatPos', 'IdleTime']

        curve_features = pd.concat([curve_features, run_features], axis = 0)

        data.update({'Features' : curve_features})

        return data
    


    def bin_frequencies(self, data):
        """
        Bin frequency results of fourier transform on decomposed curves.
        Aggregate(mean) of the values in these bins will be additional features to identify the most active frequencies within these discrete bins. 
        Bins represent windows of size on the time axis of the curve.
        Args: (self : DeviceEngine object, data(DeviceAnalysis.data) : data entry of an individual analysis object)  

        """       

        def find_split(freq_bins):
            res = (round(abs(min(freq_bins)) + abs(max(freq_bins)), 2))
            for i in [3, 4, 5, 7]:
                if res%i == 0:
                    return i
            return 4

        #keys for bin dict
        feature_names = []
        bins_list = []

        """frequency bins corresponding to seasonal and noise components"""
        seasonal = data['Seasonal']
        seasonal_freq_bins = np.fft.fftfreq(len(seasonal))
        #
        noise = data['Residual']
        noise_freq_bins = np.fft.fftfreq(len(noise))
        
        """add mean and sum residual feature"""
        mean_noise = noise.agg(['mean'])
        sum_noise = noise.agg(['sum'])
        max_noise = noise.agg(['max'])
        bins_list.append({data['Sample'] : mean_noise})
        bins_list.append({data['Sample'] : sum_noise})
        bins_list.append({data['Sample'] : max_noise})
        feature_names.append('mean_noise')
        feature_names.append('sum_noise')
        feature_names.append('max_noise')
        """magnitudes of seasonal and noise frequency components"""
        seasonal_freqs = data['Curve seasonal frequencies']
        #
        noise_freqs = data['Curve noise frequencies']

        """dynamically set number of bins edges to split frequencies into"""
        num_seasonal_bins = find_split(seasonal_freq_bins)   
        #
        num_noise_bins = find_split(noise_freq_bins)

        """create bin edges"""
        seasonal_bin_edges = np.linspace(min(seasonal_freq_bins), max(seasonal_freq_bins), num=num_seasonal_bins)
        #
        noise_bin_edges = np.linspace(min(noise_freq_bins), max(noise_freq_bins), num=num_noise_bins)

        """create bins based on preset edges and determine grouping of data"""
        seasonal_bin_indices = np.digitize(seasonal_freqs, seasonal_bin_edges, right=True)
        #
        noise_bin_indices = np.digitize(noise_freqs, noise_bin_edges, right=True)

        """average data"""
        mean_binned_seasonal_magnitudes = pd.Series(
                                                [np.mean(seasonal_freqs[seasonal_bin_indices == i]) 
                                                        for i in range(1, len(seasonal_bin_edges))
                                                ]
                                              ).fillna(0)
        """add data"""
        sum_binned_seasonal_magnitudes = pd.Series(
                                            [np.sum(seasonal_freqs[seasonal_bin_indices == i]) 
                                             for i in range(1, len(seasonal_bin_edges))]
                                             ).fillna(0)
        #
        #
        mean_binned_noise_magnitudes = pd.Series(
                                                [np.mean(noise_freqs[noise_bin_indices == i]) 
                                                        for i in range(1, len(noise_bin_edges))
                                                ]
                                              ).fillna(0)

        sum_binned_noise_magnitudes = pd.Series(
                                            [np.sum(noise_freqs[noise_bin_indices == i]) 
                                             for i in range(1, len(noise_bin_edges))]
                                             ).fillna(0)

        #concatenate the series head to tail (along axis=0)
        combined_analysis_mags = pd.concat([mean_binned_seasonal_magnitudes, 
                                            sum_binned_seasonal_magnitudes, 
                                            mean_binned_noise_magnitudes, 
                                            sum_binned_noise_magnitudes], ignore_index=True)
        
        #create a DataFrame with a single column
        bins_df = combined_analysis_mags.to_frame(name=data['Sample'])
        new_feat_names = []
        #round bin eedge values
        seasonal_bin_edges = [round(i, 2) for i in seasonal_bin_edges]
        noise_bin_edges = [round(i, 2) for i in noise_bin_edges]
        #create index/feature names
        for feat in ['snl', 'noise']:
            if feat == 'snl':
                bin_edges = seasonal_bin_edges
            else:
                bin_edges = noise_bin_edges
            for i, j in zip(bin_edges[:-1], bin_edges[1:]):
                new_feat_names.extend([f'mean_{feat}_{i}-{j}', f'sum_{feat}_{i}-{j}'])
        bins_df.index = new_feat_names

        """average and std of whole frequency magnitude arrays"""
        mean_seasonal_magnitudes = seasonal_freqs.agg(['mean']) 
        std_seasonal_magnitudes = seasonal_freqs.agg(['std'])
        bins_list.append({data['Sample'] : std_seasonal_magnitudes})
        bins_list.append({data['Sample'] : mean_seasonal_magnitudes})
        feature_names.append('std_snl_mag')
        feature_names.append('mean_snl_mag')
        #
        mean_noise_magnitudes = noise_freqs.agg(['mean']) 
        std_noise_magnitudes = noise_freqs.agg(['std'])
        bins_list.append({data['Sample'] : std_noise_magnitudes})
        bins_list.append({data['Sample'] : mean_noise_magnitudes})
        feature_names.append('std_noise_mag')
        feature_names.append('mean_noise_mag')
        
        #whole array aggregates 
        whole_df = pd.DataFrame(bins_list, index=feature_names)

        #bin aggregates
        feature_df = pd.concat([whole_df, bins_df], axis = 0)
        feature_df[data['Sample']] = feature_df[data['Sample']].astype('float64')

        return feature_df
    


    def add_extracted_features(self, data):
        """
        Add features engineered from seasonal decomposition and fourier transform to features-matrix to improve classification

        """
        new_features_df = self.bin_frequencies(data)
        features_df = data['Features']
        features_df = pd.concat([features_df, new_features_df], axis=0)
        features_df[data['Sample']] = features_df[data['Sample']].astype('float64')
        data.update({'Features' : features_df})
        return data
    


    def scale_features(self, prepared_data, type):
        """
        Scale data according to user input. Default values take over if no input.
        Returns list of changed data dicts. 

        """ 
        results = {}

        #once features are completely engineered and added, data is scaled for final ML
        for ana in prepared_data:
            #each ana is an analysis object
            data = ana.data
            features_df = data['Features']
            samples = list(features_df.columns)
            features = list(features_df.index)
            
            print(features_df)
            if type == 'minmax':
                    mm = scaler.MinMaxScaler()
                    scaled_data = mm.fit_transform(features_df.T)

            elif type == 'std':
                        std = scaler.StandardScaler()
                        scaled_data = std.fit_transform(features_df.T)

            elif type == 'robust':
                        rob = scaler.RobustScaler()
                        scaled_data = rob.fit_transform(features_df.T)

            elif type == 'maxabs':
                        mabs = scaler.MaxAbsScaler()
                        scaled_data = mabs.fit_transform(features_df.T)

            elif type == 'norm':
                        norm = scaler.Normalizer()
                        scaled_data = norm.fit_transform(features_df.T)
             
            scaled_df = pd.DataFrame(scaled_data.T, columns= samples, index= features)
            
            data.update({'Features' : scaled_df})

            results.update({f"{ana.name}_scaled" : data})
            
        #return transformed data. 
        return results



    def get_seasonal_components(self, data, period):
        """
        Break each sample's time-series curve down into its components : Trend, Seasonal, and Residual(Noise)
        Return value is the updated data containing seasonal components of all curves of the dict originally passed as input to this function. 
        Now each 'Device Pressure Analysis' dict has been modified with additional seasonal component data for each individual curve
    
        """
        period = period if not isinstance(period, type(None)) else 10

        sample_name = data['Sample']
        sample_curve = data['Curve']
        decomp = seasonal_decompose(pd.to_numeric(sample_curve[sample_name]), 
                                    model = 'additive', 
                                    period = period,  
                                    extrapolate_trend = 10)
            
        #components for current curve
        trend, seasonal, residual = pd.Series(decomp.trend, name=sample_name), pd.Series(decomp.seasonal, name=sample_name), pd.Series(decomp.resid, name=sample_name)
  
        data.update({'Trend' : trend, 
                                'Seasonal' : seasonal,
                                'Residual' : residual})    
    
        return data



    def make_fourier_transform(self, data):
        """
        Transform raw curves and seasonal component of results from get_seasonal_components() using Fast Fourier Transform.
        Behaviour of (pressure) curves can now be better analysed by inspecting them in the frequency domain.
        
        """
        curve = data['Curve'].iloc[:, 1]
        #np.fft.fft can only be performed on array so use curve.values
        transformed_curve = np.fft.fft(curve.values)
        #convert result frequencies array into absolute values to get magnitudes of frequency components
        transformed_curve = abs(transformed_curve)
        transformed_curve = pd.Series(transformed_curve, name=curve.name)
        data.update({'Raw curve frequencies' : transformed_curve})

        #if 'Seasonal' in data[analysis_key]:
        seasonal = data['Seasonal']
        transformed_seasonal = np.fft.fft(seasonal.values)
        transformed_seasonal = abs(transformed_seasonal)
        transformed_seasonal = pd.Series(transformed_seasonal, name=curve.name)
        data.update({'Curve seasonal frequencies' : transformed_seasonal})

        residual = data['Residual']
        transformed_residual = np.fft.fft(residual.values)
        transformed_residual = abs(transformed_residual)
        transformed_residual = pd.Series(transformed_residual, name=curve.name)
        data.update({'Curve noise frequencies' : transformed_residual})
        #return transformed data. 
        return data



    def get_results(self, results=None, scaled=False):
        """
        Retrieves the results from CoreEngine.

        Args:
        results (str or list): The key(s) of the result(s) to retrieve. Absence of an argument returns all known results for the current device.
        scaled (bool): filter output dict by choosing whether to return scaled features or not.
        Mods : int type input returns the results entry that lies on a list-like index within the dictionary. 
            e.g: input 4 returns the 5th entry of the results dict.

        Returns:
        dict or any: If `results` is a string, returns the corresponding result value.
                If `results` is a list, returns a dictionary with the key-value pairs of the requested results. 
                If `results` is neither a string nor a list, returns all the results.   

        """
        result_dict = self._results

        if isinstance(results, str):
            result_dict = {}            
            for key in list(self._results):
                if results in key:
                    result_dict.update({key : self._results[key]}) 
            
        elif isinstance(results, int) and results < len(self._results):
            key_list = list(self._results.keys())
            result_dict = {key_list[results] : self._results[key_list[results]]}
        
        elif isinstance(results, list):
            results_out = {}
            for result in results:
                if isinstance(result, str):
                    for key in list(self._results):
                        if result in key:
                            results_out.update({key : self._results[key]})
                elif isinstance(result, int) and result < len(self._results):
                    key_list = list(self._results.keys())
                    results_out.update({key_list[result] : self._results[key_list[result]]})
            result_dict = results_out

        elif results == None:
            print('Invalid Input! Returning all existing results!')
        
        #All existing or selected results have been retrieved till this point

        #Now filter based on user selection to return scaled or unscaled features

        found_results = list(result_dict.keys())
        found_dict = {}
        #if scaled, only return scaled results
        if scaled == True:
            for resname in found_results:    
                if '_scaled' in resname or 'Device Pressure Analysis' not in resname:
                    found_dict.update({resname : result_dict[resname]}) 

            if len(list(found_dict.keys())) == 0:
                print('No scaled results available!!')

        #if scaled False or unspecified, return unscaled data
        else:
            for resname in found_results:    
                if '_scaled' not in resname or 'Device Pressure Analysis' in resname:
                    found_dict.update({resname : result_dict[resname]}) 
    
        return found_dict
                


    def remove_results(self, results, features=''):
        """
        Removes the specified results from the internal results dictionary.
        Mods : Drop select features using 'features' argument from chosen results.

        Args:
        results: The results to be removed. It can be an integer, a string, or a list of integers or strings.

        Returns:    
        None
        """
        if features == 'base':
            feature_string = ['Features']
        elif features == 'decompose':
            feature_string = ['Trend', 'Seasonal', 'Residual']
        elif features == 'transform':
            feature_string = ['Raw curve frequencies', 'Curve seasonal frequencies', 'Curve noise frequencies']

        for feature in feature_string:

            if self._results.__len__() == 0:
                return
            if isinstance(results, int):
                keys = list(self._results.keys())
                if results < len(keys):
                    result = self._results[keys[results]]
                    ana_keys = [key for key in result if feature in key]
                    for key in ana_keys:
                        del result[key]
                    self._results.update({keys[results] : result})
                    
            if isinstance(results, str):
                if results in self._results:
                    result = self._results[results]
                    ana_keys = [key for key in result if feature in key]
                    for key in ana_keys:
                        del result[key]
                    self._results.update({results : result})

            if isinstance(results, list):
                for result in results:
                    if isinstance(result, int):
                        keys = list(self._results.keys())
                        if result < len(keys):
                            result_dict = self._results[keys[result]]
                            ana_keys = [key for key in result_dict if feature in key]
                            for key in ana_keys:
                                del result_dict[key]
                            self._results.update({keys[result] : result_dict})
                            
                    elif isinstance(result, str):
                        if result in self._results:
                            result_dict = self._results[result]
                            ana_keys = [key for key in result_dict if feature in key]
                            for key in ana_keys:
                                del result_dict[key]
                            self._results.update({result : result_dict})    

            else:
                self._results = {}


        
    def trim_method_name(self, method_name=None):
        """
        Method to trim Analysis/Method names for comparison and grouping. 

        """
        #pattern to find date extension to trim
        pattern = r'\d{6}'
        
        if isinstance(method_name, type(None)):
            return [self.trim_method_name(name) for name in self._method_ids]

        # Base case: If the string doesn't start or end with an underscore, return it as long as it has been trimmed.
        if (method_name[0] != '_' and method_name[-1] != '_') and not re.search(pattern, method_name):
            if re.search(r'^Mix-.*_training-data_', method_name):
                    method_name = re.sub(r'^Mix-.*_training-data_', '', method_name)
            return method_name
        
        # Recursive cases:
        elif method_name[0] == '_':
            # Trim leading underscore
            return self.trim_method_name(method_name[1:])  
        
        elif method_name[-1] == '_':
            # Trim trailing underscore
            return self.trim_method_name(method_name[:-1])  
        
        else:
            #whole method names have no trailing or leading '_' but have length > 10 and always have a six-digit date extension
            #get part of string containing method name without date
            new_method_name = method_name.split(' ')[0]
            # Substitute the 6-digit numbers with an empty string
            result = re.sub(pattern, '', new_method_name)   
            return self.trim_method_name(result)
        


    def group_analyses(self, data_list, group_by):
        """
        Method to compare and group related analyses based on Method ID or Date of experiment.
        """
        #helper functions
        def initialize_group(analysis, group_by):
            # Initialize a new group with the first analysis data
            method_name = analysis.data['Method']
            if group_by == 'method':            
                method_name = self.trim_method_name(method_name)
            return {'Method': method_name, 'Data': analysis.data}

        def can_merge(current_group, new_analysis, group_by):
            # Logic to determine if two analyses can be merged
            if group_by == 'method':
                return (self.trim_method_name(current_group['Method']) == self.trim_method_name(new_analysis.data['Method']))
            else:
                return (current_group['Method'] == new_analysis.data['Method'])

        def merge_groups(current_group, new_analysis):
            # Logic to merge the data from new_analysis into current_group
            new_data = new_analysis.data
            #print('new data:', new_data)
            current_data = current_group['Data']
            #print('current data:', current_data)
            current_method = current_group['Method']

            merged_data = {}

            merged_data.update({'Method' : current_method})
            for data_item in ['Curve', 
                              'Sample', 
                              'Features', 
                              'Trend', 
                              'Seasonal', 
                              'Residual', 
                              'Raw curve frequencies', 
                              'Curve seasonal frequencies', 
                              'Curve noise frequencies']:
                
                if data_item == 'Curve':
                    # to handle new data with varying time entries
                    df = pd.merge_asof(current_data[data_item], new_data[data_item], on='Time', direction='nearest')
                    # for old HPLC data
                    #df = pd.merge(current_data[data_item],new_data[data_item], on='Time')
                    merged_data.update({data_item : df})
                
                elif data_item == 'Sample':
                    current_samples = current_data[data_item]
                    new_samples = new_data[data_item]
                    merged_samples = []
                    if not isinstance(current_samples, list):
                        merged_samples.append(current_samples)
                    else:
                        merged_samples = current_samples
                    if not isinstance(new_samples, list):
                        merged_samples.append(new_samples)
                    else:
                        merged_samples.extend(new_samples)
                    merged_data.update({data_item : merged_samples})    
                
                else:
                    if data_item in list(current_data.keys()):
                        #seems to be fixed. KEEP AN EYE ON THIS
                        merged_data.update({data_item : pd.concat([current_data[data_item],new_data[data_item]], axis = 1)})
            # Handle DataFrame concatenation and updating as needed
            current_group.update({'Data' : merged_data})
            return current_group
        

        if not isinstance(data_list, (list, dict)):
            raise ValueError('Invalid input data type. Expected list or dict.')

        scaled_anas, anas_to_plot = [], []

        if isinstance(data_list, dict):
            for key, value in data_list.items():
                if '_scaled' in key or 'Device Pressure Analysis' not in key:
                    print(f'{key} is already prepared!')
                    scaled_anas.append(DeviceAnalysis(name=key, data=value))
                else:
                    anas_to_plot.append(DeviceAnalysis(name=key, data=value))

            if not anas_to_plot:
                return scaled_anas
        
        elif isinstance(data_list, list):
            anas_to_plot = data_list

        num_analyses = len(anas_to_plot)
        if num_analyses == 0:
            print('No analyses found!')
            return []

        # Grouping logic
        objects_list = []
        current_group = initialize_group(anas_to_plot[0], group_by)

        for ana in anas_to_plot[1:]:
            if can_merge(current_group, ana, group_by):
                current_group = merge_groups(current_group, ana)
            else:
                current_method = current_group['Method']
                print('now saving current group... '  + current_method)
                #print('Diag:', current_group['Data'])
                names_list = [ana.name for ana in objects_list]
                if current_method not in names_list:
                    objects_list.append(DeviceAnalysis(name=current_method, data=current_group['Data']))
                else:
                    #match the index of the analysis that has the same name as the current one
                    unikey_index = names_list.index(current_method)
                    #group the new grouped data with older storage
                    print(f'Merging this group with older {current_method} data..')
                    grouped = self.group_analyses([objects_list[unikey_index], DeviceAnalysis(name = current_method, data = current_group['Data'])], group_by)
                    #replace old object with the same name
                    del objects_list[unikey_index]
                    objects_list.extend(grouped)

                current_group = initialize_group(ana, group_by)
        
        current_method = current_group['Method']
        names_list = [ana.name for ana in objects_list]
        if current_method not in names_list:
            objects_list.append(DeviceAnalysis(name=current_method, data=current_group['Data']))
        else:
            #match the index of the analysis that has the same name as the current one
            unikey_index = names_list.index(current_method)
            #group the new grouped data with older storage
            print(f'Merging this group with older {current_method} data..')
            grouped = self.group_analyses([objects_list[unikey_index], DeviceAnalysis(name = current_method, data = current_group['Data'])], group_by)
            #replace old object with the same name
            del objects_list[unikey_index]
            objects_list.extend(grouped)  # Append the last group

        if group_by != 'method':
            for ana in objects_list:
                if (ana.data['Method'][0] != '_' and ana.data['Method'][-1] != '_') and len(ana.data['Method']) <= 10:
                    del ana
        return objects_list



    def plot_analyses(self, analyses=None, interactive=True, group_by='method'):
        """
        Plots each analysis dataframe by calling plot() function of respective DeviceAnalysis objects

        """
        #retrieve list of analysis objects based on user input 
        anas_to_plot = self.get_analyses(analyses)

        objects_list = self.group_analyses(anas_to_plot, group_by=group_by)    

        for ana in objects_list:
            ana.plot(interactive=interactive)
            del ana
    
    

    def plot_results(self, results=None, features='', type=None, scaled=True, interactive=True, transpose=False, group_by='method'):
        """
        Plot the computed (and added) results of feature extraction, seasonal decomposition, fourier transform.
    
        """
        result_dict = self.get_results(results)

        #incase scaled is activated
        if features == 'base' and scaled == True:
            result_dict = self.get_results(results, scaled=scaled)
       
        objects_list = self.group_analyses(result_dict, group_by=group_by)
        
        for ana in objects_list:        

            if features == 'base': 
                ana.plot(features = True, type = type, scaled = scaled, interactive=interactive, transpose=transpose) 
                            
            elif features == 'decompose':
                ana.plot(decomp = True, type = type, interactive=interactive) 

            elif features == 'transform':
                ana.plot(transform = True, type = type, interactive=interactive)

            else:
                ana.plot(interactive=interactive)

            del ana


  
    def get_feature_matrix(self, results=None):
        """
        3-way function that uses ML engines that each run host DeviceEngine methods to classify data from a particular method.
        ML object with an iteration of this function exists for each method in DeviceEngine.  
        
        """
        #retrieve scaled features data of desired result for classification
        result_dict = self.get_results(results=results, scaled=True)
        result_keys = list(result_dict.keys())
        
        #setup train and test sets
        train_feature_dfs = []
        test_feature_dfs = []

        curve_dfs = []
        print(result_keys)
        #feature_dfs = [result_dict[result_keys[0]]['Features']]
        #new_df = feature_dfs[0]
        # ´join all curves, split features of training and test sets
        #curve_dfs = [result_dict[result_keys[0]]['Curve']]
        #for (this_key, next_key) in zip(result_keys[:-1], result_keys[1:]):
        for key in result_keys:
            curve_dfs.append(result_dict[key]['Curve'])
            if 'basis' in key:
                train_feature_dfs.append(result_dict[key]['Features'])
            
            else:
                if 'cell' in key:
                    test_feature_dfs.append(result_dict[key]['Features'])

        train_features = pd.concat(train_feature_dfs, axis=1)
        test_features = pd.concat(test_feature_dfs, axis=1)
        
        new_curve_df = curve_dfs[0]
        for i in range(1, len(curve_dfs)):
            new_curve_df = pd.merge_asof(new_curve_df, curve_dfs[i], on='Time', direction='nearest')

        #transpose to enable ML
        train_features = train_features.T
        test_features = test_features.T
        
        #migrate prepared and scaled features matrix as MachineLearningAnalysis data to MachineLearningEngine
        #ml_objects = []
        #x_values = list(new_df.columns)
        #samples = list(new_df.index)
        #for i in range(len(new_df)):
        #    name = samples[i]
        #    data = {'x' : x_values, 
        #            'y' : new_df.iloc[i]}
        #    ml_objects.append(MachineLearningAnalysis(name=name, data=data))

        #create MLEngine objects for each method-grouped set of results like 'Pac' or 'Doc'
   
        #ml_engine = MachineLearningEngine(headers = {'name': f'ML{results}', 'author': 'Sandeep H.'}, analyses = ml_objects)
        #return (ml_engine, new_curve_df)
        return(train_features, test_features, new_curve_df)
        


        
        