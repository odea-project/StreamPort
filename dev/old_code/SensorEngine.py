import time
from src.StreamPort.core import Engine

# Added as lads_client.pth in site-packages folder of venv sp
import lads_client as lc


class SensorEngine(Engine):
    """
    A class for reading and deploy sensor data streams.

    """

    def read_from_ladsopcua(self, config="config.json"):
        """
        Reads data from LADS OPC UA server.
        :param config: Path to the configuration file.
        :return: None
        """
        connections = lc.Connections(config)
        while not connections.initialized:
            time.sleep(0.1)
        functional_units = connections.functional_units
        print("Number of func_units: ", functional_units.__len__())
        functional_unit_names = list(
            map(lambda functional_unit: functional_unit.at_name, functional_units)
        )
        print(functional_unit_names)
        print(functional_units[0].current_state.value_str)
