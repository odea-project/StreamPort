from ..core.Analysis import Analysis

class MachineLearningAnalysis(Analysis):

    """
    Represents MachineLearningAnalysis class that inherits from Analysis class

    """

    def __init__(self, name=None, replicate=None, blank=None, data=None, classes=None):

        """
        Initializes the MachineLearningAnalysis instance

        Args:
            name (str): The name of the analysis.
            replicate (str): The name of the replicate.
            blank (str): The name of the blank.
            data (list): The data of the analysis, which is a dict of one dimension numpy arrays.
            classes (str): Soon!
        """

        super().__init__(name, replicate, blank, data)
        self.classes = str(classes) if classes else None
    