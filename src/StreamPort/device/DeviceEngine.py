from ..core.CoreEngine import CoreEngine


class DeviceEngine(CoreEngine):

    def __init__(self, headers=None, settings=None, analyses=None, results=None):
        
        super().__init__(headers, settings, analyses, results)