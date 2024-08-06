from ..core.CoreEngine import CoreEngine
from ..device.DeviceAnalysis import DeviceAnalysis

#CHECK WHETHER PACKAGES HAVE SUFFICIENT SUPPORT
#Numpy has fft, check also if other stats methods can be implemented without importing tsfresh 

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

#module to enable time decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

#Module to perform fourier transformation on seasonal components of curves
from scipy import fftpack

"""
GLOBAL VARIABLES
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

        _method_ids (list(str)): List of all unique analyses identified by method and date of experiment.


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
        """
        Searches for analyses in the given source or current working directory if source is not provided.
        Prepares the available analyses and return a list of analyses objects that conform to data processing format.

        """
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
                    runtype = r'RUN.LOG'  
                    #acq_type = r'acq.txt' not needed until further updates
                    filetype = r'Pressure.CSV'

                    #individual curves within current method to be later merged into a dataframe against common time feature
                    curves_list = []    

                    #curve headers/names
                    curves = []    

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
                            for line in f:
                                log_data.append(line)

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

                                    #convert runtime string into absolute number of seconds
                                    runtime = datetime.strptime(runtime, '%H:%M:%S')
                                    runtime = runtime.second + runtime.minute * 60 + runtime.hour * 3600

                                    print("Runtime this run(seconds) : " + str(runtime))

                            print("Times started(in event of fault) : " + str(times_started))

                        f.close()

                        print('This files logs')
                        print(log_data)

                        try:
                            
                            #to account for .D folder without numbering, like 'Irino_kali.D', etc.
                            if not pressure_suffix.isdecimal():                 
                                pressure_suffix = 1
                                suffix_digits = 3
                                #assign class label 2 to first run of every experiment 
                                run_type = 2
                        
                            #if blank run encountered on reading current run's .LOG file, name run column with '-blank' as identifier
                            if blank_identifier == 1:
                                    
                                run_suffix = '-blank'

                                #assign class label 0 to blanks
                                if pressure_suffix != 1:
                                    run_type = 0 

                                #add blank run headers to list of blanks
                                curve_header = "Sample - " + (str(pressure_suffix).zfill(suffix_digits) + run_suffix)
                                

                            else:

                                #sample runs indicated with sample name
                                run_suffix = filename[-1]

                                #assign class label 1 to samples
                                if pressure_suffix != 1:
                                    run_type = 1

                                #add sample run headers to samples list, keep trailing pressure suffix for future analysis
                                curve_header = "Sample - " + run_suffix  
                                

                            curves.append([curve_header, start_date_string])

                            #set current run header based on file number and run type     
                            cols = ["Time", curve_header]

                            #path to 'Pressure.CSV' file found in current directory. Must be only one 
                            target_file = target_files[0]

                            target_file = os.path.join(current_run_folder, target_file)

                            #add pressure curve to list of curves for current method  
                            curves_list.append(pd.read_csv(target_file, 
                                                                sep = ";",
                                                                #decimal = ",",
                                                                header = None, 
                                                                names = cols))
                            
                            print(curve_header + ' : \n' + 'start date : ' + start_date_string)
                            print('end date : ' + end_date_string)
                            print('runtime : ' + str(runtime) + '\n')
                                            
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
                            analysis_name = f"Device Pressure Analysis - {method_suffix} | Start time: {start_date_string}"
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
                                         'Runtype' : run_type,
                                         'Start date' : start_date_string, 
                                         'Runtime' : runtime, 
                                         'Idle time' : "NA",
                                         'Number of Trials' : times_started, 
                                         'Curve' : curves_list[-1], 
                                         'Log' : log_data}

                        #add every encountered analysis to device history
                        self._history.update({analysis_name : analysis_data})    

                        #create new DeviceAnalysis object.
                        new_object = DeviceAnalysis(name = analysis_name, data = analysis_data, analysis_type = analysis_type)

                        #list of analyses populated with individual analysis objects
                        analyses_list.append(new_object)

                if not method_suffix:
                    continue

                if method_suffix not in self._method_ids:

                    #add current method ID to list of known/encountered method types, no duplicates
                    self._method_ids.append(method_suffix)

        return analyses_list



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
                    
            elif isinstance(analyses, list):
                analyses_out = []
                for analysis in analyses:
                    if isinstance(analysis, int) and analysis < len(self._analyses):
                        analyses_out.append(self._analyses[analysis])
                    elif isinstance(analysis, str):
                        for a in self._analyses:
                            if analysis in a.name:
                                analyses_out.append(a)
                ana_list = analyses_out
            
            else:
                print("Analysis not found!")

        else:
            print("Provided data is not sufficient or does not exist! Existing analyses will be returned.")
            
        return ana_list        



    def get_features(self, data, features_list, smoothed):
        """
        #Settings are decided in DeviceProcSettings and passed as data(dict) and features_list(list(str)) and smoothed(bool).
        #additional features that complement information from given set of features are runtime and runtype/class label.
        
        """
        smoothed = smoothed if isinstance(smoothed, bool) else False

        #runtime of each sample indicates possible faults with the run
        runtime = {}

        #runtype describes whether run was a flush(blank), a sample run or the first run of the day.
        runtype = {}

        #update each Device Pressure Analysis with its features in addition to creating the combined dataframe
        curve = data['Curve']
        #'pct_change' transformation is first used on pressure curves to emphasise focus on changes in the curve over time.
        #features extracted from 'pct_change' curves hepl better model curve behaviour.
        curve_features = curve.iloc[:, 1].agg(features_list)
                
        if smoothed == True:    
            smoothed_curve_features = (curve.iloc[:, 1].agg('pct_change')*100).agg(features_list)
            smoothed_curve_features.index = [f"{i}_percent_change" for i in smoothed_curve_features.index]
            curve_features = smoothed_curve_features
                 
        runtime.update({data['Sample'] : data['Runtime']})
        runtype.update({data['Sample'] : data['Runtype']})

        run_features = pd.DataFrame([runtime, runtype], 
                                    columns=runtype.keys())
                
        run_features.index = ['Runtime', 'Runtype']

        curve_features = pd.concat([curve_features, run_features], axis = 0)

        data.update({'Features' : curve_features})

        return data
    


    def get_seasonal_components(self, data, period):
        """
        Break each sample's time-series curve down into its components : Trend, Seasonal, and Residual(Noise)
        Return value is the updated data containing seasonal components of all curves of the dict originally passed as input to this function. 
        Now each 'Device Pressure Analysis' dict has been modified with additional seasonal component data for each individual curve
    
        """
        period = period if not isinstance(period, type(None)) else 10

        sample_name = data['Sample']
        sample_curve = data['Curve']
        decomp = seasonal_decompose(sample_curve[sample_name], 
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
        transformed_curve = fftpack.fft(curve.values)
        #convert result frequencies array into absolute values for better readability
        transformed_curve = abs(transformed_curve)
        transformed_curve = pd.Series(transformed_curve, name=curve.name)
        data.update({'Raw curve frequencies' : transformed_curve})

        #if 'Seasonal' in data[analysis_key]:
        seasonal = data['Seasonal']
        transformed_seasonal = fftpack.fft(seasonal.values)
        transformed_seasonal = abs(transformed_seasonal)
        transformed_seasonal = pd.Series(transformed_seasonal, name=curve.name)
        data.update({'Curve seasonal frequencies' : transformed_seasonal})

        residual = data['Residual']
        transformed_residual = fftpack.fft(residual.values)
        transformed_residual = abs(transformed_residual)
        transformed_residual = pd.Series(transformed_residual, name=curve.name)
        data.update({'Curve noise frequencies' : transformed_residual})
        #return transformed data. 
        return data



    def get_rolling_stats(self, data, features_list, period):
        """
        Compute rolling features of (pressure) curves over a specified window. Period defaults to 5
        """
        period = period if isinstance(period, int) else 5
        #roll_stats = pd.DataFrame()
        
        
        smoothed_curves = pd.DataFrame() 
            
        curve = data['Curve'].iloc[:, 1]
        for feature in features_list:

            if feature == 'mean':
                smoothed_curve = curve.rolling(window=period).mean()

            elif feature == 'min':
                smoothed_curve = curve.rolling(window=period).min()

            elif feature  == 'max':
                smoothed_curve = curve.rolling(window=period).max()

            elif feature == 'std':
                smoothed_curve = curve.rolling(window=period).std()

            elif feature == 'ema':
                smoothed_curve = curve.ewm(span=period).mean()

            feature = f"{curve.name}_roll_{feature}"
            smoothed_curve.name = feature
            smoothed_curves = pd.concat([smoothed_curves, smoothed_curve], axis = 1)

        data.update({'Rolling statistics' : smoothed_curves})
        
        #return transformed data. 
        return data



    def get_results(self, results=None):
        """
        Retrieves the results from CoreEngine.

        Args:
        results (str or list): The key(s) of the result(s) to retrieve. Absence of an argument returns all known results for the current device.
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
            result_pattern = re.compile('^.+' + results + '.+$')
            for key in self._results:
                if re.match(result_pattern, key):
                    result_dict.update({key : self._results[key]}) 
            
        elif isinstance(results, int) and results < len(self._results):
            key_list = list(self._results.keys())
            result_dict = {key_list[results] : self._results[key_list[results]]}
        
        elif isinstance(results, list):
            results_out = {}
            for result in results:
                if isinstance(result, str):
                    result_pattern = re.compile('^.+' + result + '.+$')
                    for key in self._results:
                        if re.match(result_pattern, key):
                            results_out.update({key : self._results[key]}) 
                elif isinstance(result, int) and result < len(self._results):
                    key_list = self._results.keys()
                    results_out.update({key_list[results] : self._results[key_list[results]]})
            result_dict = results_out

        elif results == None:
            print('Invalid Input! Returning all existing results!')

        return result_dict
    

                
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
        elif features == 'rolling':
            feature_string = ['Rolling statistics']

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



    def prepare_plots(self, plot_list):
        """
        Group and organize data appropriately into unique set of runs/experiments using method ids and dates.
        Args:
            list/dict of desired data.
        Returns:
            prepared list/dict of newly created analyses objects grouped appropriately.

        """
        if isinstance(plot_list, list):
            anas_to_plot = plot_list 

        elif isinstance(plot_list, dict):
            anas_to_plot = [DeviceAnalysis(name = res, data = plot_list[res]) for res in list(plot_list)]
        
        num_analyses = len(anas_to_plot)
        
        #create list of distinct analyses data dicts present in found analyses. 
        curves, samples, features, trends, seasonals, residuals, raw_freqs, seasonal_freqs, noise_freqs, methods = [], [], [], [], [], [], [], [], [], []

        for ana in anas_to_plot:
            method = ana.data
            curves.append(method['Curve'])
            samples.append(method['Sample'])
            features.append(method['Features'])
            trends.append(method['Trend'])
            seasonals.append(method['Seasonal'])
            residuals.append(method['Residual'])
            raw_freqs.append(method['Raw curve frequencies'])
            seasonal_freqs.append(method['Curve seasonal frequencies'])
            noise_freqs.append(method['Curve noise frequencies'])
            methods.append(method['Method'])
        
        plot_data = anas_to_plot[0]
        df = curves[0]
        new_method =  methods[0]
        these_samples = [samples[0]]

        #list of newly created analysis objects to be plotted
        objects_list = []
        
        for i in range(1, num_analyses):            
            next_curve = curves[i]

            next_method = methods[i]

            if next_method == new_method:
                df = pd.concat([df, next_curve], axis = 0)

                features[0] = pd.concat([features[0], features[i]], axis = 0)

                trends[0] = pd.concat([trends[0], trends[i]], axis = 1)
                seasonals[0] = pd.concat([seasonals[0], seasonals[i]], axis = 1)
                residuals[0] = pd.concat([residuals[0], residuals[i]], axis = 1)

                raw_freqs[0] = pd.concat( [ raw_freqs[0], raw_freqs[i] ], 
                                               axis = 1)
                seasonal_freqs[0] = pd.concat( [ seasonal_freqs[0], seasonal_freqs[i] ], 
                                               axis = 1)
                noise_freqs[0] = pd.concat([noise_freqs[0], noise_freqs[i]], 
                                           axis = 1)
                
                these_samples.append(samples[i])

            else:
                plot_data.data['Curve'] = df

                plot_data.data['Features'] = features[0]

                trends[0].columns = these_samples
                seasonals[0].columns = these_samples
                residuals[0].columns = these_samples
                plot_data.data['Trend'] = trends[0]
                plot_data.data['Seasonal'] = seasonals[0]
                plot_data.data['Residual'] = residuals[0]

                raw_freqs[0].columns = these_samples
                seasonal_freqs[0].columns = these_samples
                noise_freqs[0].columns = these_samples
                plot_data.data['Raw curve frequencies'] = raw_freqs[0]
                plot_data.data['Curve seasonal frequencies'] = seasonal_freqs[0]
                plot_data.data['Curve noise frequencies'] = noise_freqs[0]                

                objects_list.append(DeviceAnalysis(name = plot_data.name, data = plot_data.data))

                plot_data = anas_to_plot[i]
                df = next_curve
                these_samples = []

            new_method = next_method
        
        #loop ends before adding last item(when mismatched). Add this item to list of objects
        plot_data.data['Curve'] = df
        plot_data.data['Features'] = features[0]
       
        trends[0].name = samples[-1]
        seasonals[0].name = samples[-1]
        residuals[0].name = samples[-1]
        plot_data.data['Trend'] = trends[0]
        plot_data.data['Seasonal'] = seasonals[0]
        plot_data.data['Residual'] = residuals[0]

        raw_freqs[0].name = samples[-1]
        seasonal_freqs[0].name = samples[-1]
        noise_freqs[0].name = samples[-1]
        plot_data.data['Raw curve frequencies'] = raw_freqs[0]
        plot_data.data['Curve seasonal frequencies'] = seasonal_freqs[0]
        plot_data.data['Curve noise frequencies'] = noise_freqs[0] 
                
        objects_list.append(DeviceAnalysis(name = plot_data.name, data = plot_data.data))
        
        return objects_list



    def plot_analyses(self, analyses=None):
        """
        Plots each analysis dataframe by calling plot() function of respective DeviceAnalysis objects

        """
        #retrieve list of analysis objects based on user input 
        anas_to_plot = self.get_analyses(analyses)

        objects_list = self.prepare_plots(anas_to_plot)    

        for ana in objects_list:
            ana.plot()
            del ana
    
    

    def plot_results(self, results=None, features='', type=None):
        """
        Plot the computed (and added) results of feature extraction, seasonal decomposition, fourier transform or rolling statistics.
        ***FIX DUPLICATES AND FEATURE PLOTS BUG. 
        """
        result_dict = self.get_results(results)

        objects_list = self.prepare_plots(result_dict)

        for ana in objects_list:        

            if features == 'base':    
                ana.plot(features = True, type = type) 
                            
            elif features == 'decompose':
                ana.plot(decomp = True, type = type) 

            elif features == 'transform':
                ana.plot(transform = True, type = type)
                    
            elif features == 'rolling':
                ana.plot(rolling = True)

            else:
                ana.plot()

            del ana