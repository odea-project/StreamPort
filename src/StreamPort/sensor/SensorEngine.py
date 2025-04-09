import time
from ..core.CoreEngine import CoreEngine

# Added as lads_client.pth in site-packages folder of venv sp
from lads_opcua_client import main as lc

class SensorEngine(CoreEngine):

    """
    A class for reading and deploy sensor data streams.
    
    """
 
    def __init__(self, headers=None, settings=None, analyses=None, results=None):

        super().__init__(headers, settings, analyses, results)

    def read_from_ladsopcua(self, config = "config.json"):
        
        connections = lc.Connections(config)

        while not connections.initialized:
            time.sleep(0.1)

        functional_units = connections.functional_units

        print("Number of func_units: ", functional_units.__len__())

        functional_unit_names = list(map(lambda functional_unit: functional_unit.at_name, functional_units))

        print(functional_unit_names)

        print(functional_units[0].current_state.value_str)
