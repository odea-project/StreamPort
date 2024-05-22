from ..core.CoreEngine import Analysis


class DeviceAnalysis(Analysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None):
        
        super().__init__(name, replicate, blank, data)