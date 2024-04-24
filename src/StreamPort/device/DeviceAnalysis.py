from ..core.CoreEngine import Analyses


class DeviceAnalysis(Analyses):

    def __init__(self, name=None, replicate=None, blank=None, data=None):
        
        super().__init__(name, replicate, blank, data)