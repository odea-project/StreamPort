"""
This module contains processing methods for device analyses data.
"""

import datetime
from src.StreamPort.core import ProcessingMethod
from src.StreamPort.device.analyses import PressureCurves


class PressureCurvesMethodAssignBatchPositionNative(ProcessingMethod):
    """
    Assigns batch position and calculated iddle time to pressure curves using a native algorithm.
    """

    def __init__(self):
        super().__init__()
        self.data_type = "PressureCurves"
        self.method = "AssignBatchPosition"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {}

    def run(self, analyses: PressureCurves):
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        data.sort(
            key=lambda x: (x["timestamp"] if x["timestamp"] else datetime.datetime.min)
        )

        for i, pc in enumerate(data):
            if i == 0:
                pc["idle_time"] = 0
                pc["batch_position"] = 1
                continue

            pc["idle_time"] = (
                pc["timestamp"] - data[i - 1]["timestamp"]
            ).total_seconds()

            if (
                pc["method"] == data[i - 1]["method"]
                and pc["batch"] == data[i - 1]["batch"]
            ):
                pc["batch_position"] = data[i - 1]["batch_position"] + 1
            else:
                pc["batch_position"] = 1

            data[i] = pc

        analyses.data = data
        return analyses


class PressureCurvesMethodExtractFeaturesNative(ProcessingMethod):
    """
    Method to extract features from pressure curves using a native algorithm.
    """

    def __init__(self):
        super().__init__()
        self.data_type = "PressureCurves"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {}

    def run(self, analyses: PressureCurves):
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        features_template = {
            "method": "",
            "batch_position": 0,
            "run_type": "",
            "idle_time": 0,
            "pressure_max": 0,
            "pressure_min": 0,
            "pressure_mean": 0,
            "pressure_std": 0,
            "pressure_range": 0,
            "runtime_delta_percentage": 0,
        }

        unique_methods = set()
        for pc in data:
            if pc["method"] not in unique_methods:
                unique_methods.add(pc["method"])

        for i, pc in enumerate(data):
            feati = features_template.copy()

            for i, method in enumerate(unique_methods):
                if pc["method"] == method:
                    feati["method"] = i
                    break

            feati["batch_position"] = pc["batch_position"]

            if pc["sample"] == "Blank":
                feati["run_type"] = 0
            else:
                feati["run_type"] = 1

            feati["idle_time"] = pc["idle_time"]
            feati["pressure_max"] = max(pc["pressure_var"])
            feati["pressure_min"] = min(pc["pressure_var"])
            feati["pressure_mean"] = sum(pc["pressure_var"]) / len(pc["pressure_var"])
            feati["pressure_std"] = (
                sum((x - feati["pressure_mean"]) ** 2 for x in pc["pressure_var"])
                / len(pc["pressure_var"])
            ) ** 0.5
            feati["pressure_range"] = feati["pressure_max"] - feati["pressure_min"]
            feati["runtime_delta_percentage"] = (
                ((max(pc["time_var"]) * 60 - min(pc["time_var"]) * 60) - pc["runtime"])
                / pc["runtime"]
                * 100
            )

            pc["features"] = feati

            data[i] = pc

        analyses.data = data
        return analyses
