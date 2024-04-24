from ..core.Analyses import Analyses

class SensorAnalysis(Analyses):

    """
    Represents SensorAnalysis class that inherits from Analysis class

    """

    def __init__(self, name=None, replicate=None, blank=None):

        """
        Initializes the SensorAnalysis instance

        Attributes:
            name (str): The name of the analysis.
            replicate (str): The name of the replicate.
            blank (str): The name of the blank.
        """
        super().__init__(name, replicate, blank)