from ..core.CoreEngine import CoreEngine
import os

class DeviceEngine(CoreEngine):

    """
    DeviceEngine class inherited from CoreEngine class. Handles the actual device-related processing tasks.
    Attributes are declared by passing appropriate class objects to initialization.

    Attr:

        headers (ProjectHeaders, optional): The project headers. Instance of ProjectHeaders class or dict.
        settings (list, optional): The list of settings. Instance or list of instances of ProcessingSettings class.
        analyses (list, optional): The list of analyses. Instance or list of instances of DeviceAnalysis class.
        results (dict, optional): The dictionary of results.

        _source (raw_str, optional): Path to source directory of data/analysis. 
                                     Can be an individual analysis or parent directory of multiple analyses.
                    Source variable must be a string or raw string or exclude escape characters by using '\\' instad of '\'.

    Methods:

        read_device_spectra(self, source(raw_str, optional)) : 
                Reads and arranges all available (pressure) spectra from a given path to a source directory.
                If source not specified, searches cwd(current working directory) for compatible data.
                Curves are grouped by unique Method ID and date of experiment.

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

    _source = r''

    def __init__(self, headers=None, settings=None, analyses=None, results=None, source =None):
        
        super().__init__(headers, settings, analyses, results)
        self._source = r'{}'.format(source) if not isinstance(source, None) else os.getcwd()
    

    def read_device_spectra(self):

        #initialize stack to iterate over all superfolders
        dir_stack = [self._source]

        return()
    


    def plot_spectra(self):

        return()
    


    def add_features(self, features_list):

        return()
    


    def get_features(self):

        return()
    


    def drop_features(self, features_list):

        return()