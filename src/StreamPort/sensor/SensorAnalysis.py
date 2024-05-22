from ..core.Analysis import Analysis

class SensorAnalysis(Analysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None, type=None):

        super().__init__(name, replicate, blank, data)
        
        self.type = str(type) if type else None