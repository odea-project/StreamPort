from ..core.Analyses import Analyses

class MachineLearningAnalysis(Analyses):

    """
    Represents MachineLearningAnalysis class that inherits from Analysis class

    """

    def __init__(self, name=None, replicate=None, blank=None, data=None):

        """
        Initializes the MachineLearningAnalysis instance

        Args:
            name (str): The name of the analysis.
            replicate (str): The name of the replicate.
            blank (str): The name of the blank.
            data (list): The data of the analysis, which is a dict of one dimension numpy arrays.
        """

        super().__init__(name, replicate, blank, data)
    