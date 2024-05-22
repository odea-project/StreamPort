from ..core.CoreEngine import CoreEngine


class DeviceEngine(CoreEngine):

    """
    DeviceEngine class inherited from CoreEngine class. Handles the actual device-related processing tasks.
    Attributes are declared by passing appropriate class objects to initialization.

    Attr:

        headers (ProjectHeaders, optional): The project headers. Instance of ProjectHeaders class or dict.
        settings (list, optional): The list of settings. Instance or list of instances of ProcessingSettings class.
        analyses (list, optional): The list of analyses. Instance or list of instances of DeviceAnalysis class.
        results (dict, optional): The dictionary of results.

    Methods:

        read_device_spectra(source(str, optional)) : 
                Reads and arranges all available (pressure) spectra from a given path to a source directory.
                If source not specified, searches cwd(current working directory) for compatible data.
                Curves are grouped by unique Method ID and date of experiment.

        plot_spectra() : 
                Creates an interactive plot of (pressure) curves against time.
                Unique for unique Method IDs.
                To be expanded to handle other data(e.g. device conditions, curve features).
                (Possible future upgrade)User can specify resolution and/or curve smoothing ratio.

        add_features(features_list(str/list, optional)) :
                Extracts features from curves and arranges data appropriately for further processing and ML operations.
                If target features not specified, default list(mean, min, max, std) applied.

        get_features() :
                Returns/prints result dataset of extracted features over all samples.

        drop_features(features_list(str/list of feature/column name(s), optional)) :
                Removes undesired features from prepared dataset. 
                Essentially acts like df.drop(columns), where the columns are features to remove.
                If argument unspecified, defaults to drop all features.
    """         


    def __init__(self, headers=None, settings=None, analyses=None, results=None):
        
        super().__init__(headers, settings, analyses, results)

    

    def read_device_spectra(self, path = None):

        return()
    


    def plot_spectra():

        return()
    


    def add_features():

        return()
    


    def get_features():

        return()
    


    def drop_features():

        return()