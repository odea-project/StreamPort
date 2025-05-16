from src.StreamPort.core import Analyses

class SensorAnalyses(Analyses):
    """
    A class for reading and deploying sensor data streams.
    """
    def __init__(self):
        super().__init__(data_type="Sensor", formats=None, data=None)