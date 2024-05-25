from ..core.CoreEngine import Analysis


class DeviceAnalysis(Analysis):

    def __init__(self, name=None, replicate=None, blank=None, data=None):
        
        super().__init__(name, replicate, blank, data)

    def validate(self):

        if not isinstance(self.replicate, str):
            pass

        if not isinstance(self.blank, str):
            pass            