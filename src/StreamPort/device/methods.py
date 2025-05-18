"""
This module contains processing methods for device analyses data.
"""

import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
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

    Args:
        period (int): The period for seasonal decomposition. Default is 10.
        bins (int): The number of bins for Fast Fourier Transformation (FFT). Default is 4.
    """

    def __init__(self, period: int = 10, bins: int = 4):
        super().__init__()
        self.data_type = "PressureCurves"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"period": period, "bins": bins}

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
            "runtime": 0,
            "runtime_delta_percentage": 0,
            "residual_mean": 0,
            "residual_std": 0,
            "residual_sum": 0,
            "residual_max": 0,
        }

        for i in range(self.parameters["bins"] + 1):
            features_template[f"seasonal_fft_mean_{i}"] = 0
            features_template[f"seasonal_fft_sum_{i}"] = 0
            features_template[f"residual_fft_mean_{i}"] = 0
            features_template[f"residual_fft_sum_{i}"] = 0

        features_raw_transform = {
            "trend": [],
            "seasonal": [],
            "residual": [],
            "seasonal_fft": [],
            "residual_fft": [],
            "sample_spacing": 0,
            "freq_bins": [],
            "freq_bin_edges": [],
            "freq_bins_indices": [],
        }

        unique_methods = set()
        for pc in data:
            if pc["method"] not in unique_methods:
                unique_methods.add(pc["method"])

        for i, pc in enumerate(data):
            feati = features_template.copy()
            featrawi = features_raw_transform.copy()

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
            feati["runtime"] = (max(pc["time_var"]) - min(pc["time_var"])) * 60
            feati["runtime_delta_percentage"] = (
                abs(feati["runtime"] - pc["runtime"]) / pc["runtime"] * 100
            )

            decomp = seasonal_decompose(
                pd.to_numeric(pc["pressure_var"]),
                model="additive",
                period=self.parameters["period"],
                extrapolate_trend=10,
            )

            featrawi["trend"] = decomp.trend
            featrawi["seasonal"] = decomp.seasonal
            featrawi["residual"] = decomp.resid

            transformed_seasonal = np.fft.fft(decomp.seasonal)
            transformed_seasonal = abs(transformed_seasonal)

            transformed_residual = np.fft.fft(decomp.resid)
            transformed_residual = abs(transformed_residual)

            time_var = np.array(pc["time_var"])
            if len(time_var) > 1:
                sample_spacing = np.mean(np.diff(time_var))
            else:
                sample_spacing = 1.0

            freq_bins = np.fft.fftfreq(len(decomp.seasonal), d=sample_spacing)

            positive_freqs = freq_bins > 0
            freq_bins = freq_bins[positive_freqs]
            transformed_seasonal = transformed_seasonal[positive_freqs]
            transformed_residual = transformed_residual[positive_freqs]

            featrawi["seasonal_fft"] = transformed_seasonal
            featrawi["residual_fft"] = transformed_residual
            featrawi["sample_spacing"] = sample_spacing
            featrawi["freq_bins"] = freq_bins

            feati["residual_mean"] = np.mean(transformed_residual)
            feati["residual_std"] = np.std(transformed_residual)
            feati["residual_sum"] = np.sum(transformed_residual)
            feati["residual_max"] = np.max(transformed_residual)

            num_bins = 4
            freq_bin_edges = np.histogram_bin_edges(freq_bins, bins=num_bins)
            featrawi["freq_bin_edges"] = freq_bin_edges

            freq_bins_indices = np.digitize(freq_bins, freq_bin_edges, right=True)
            unique_bins_indices = np.unique(freq_bins_indices)

            featrawi["freq_bins_indices"] = freq_bins_indices

            mean_binned_seasonal_magnitudes = np.array(
                [
                    (
                        np.mean(transformed_seasonal[freq_bins_indices == i])
                        if np.any(freq_bins_indices == i)
                        else 0
                    )
                    for i in unique_bins_indices
                ]
            )

            sum_binned_seasonal_magnitudes = np.array(
                [
                    (
                        np.sum(transformed_seasonal[freq_bins_indices == i])
                        if np.any(freq_bins_indices == i)
                        else 0
                    )
                    for i in unique_bins_indices
                ]
            )

            mean_binned_residual_magnitudes = np.array(
                [
                    (
                        np.mean(transformed_residual[freq_bins_indices == i])
                        if np.any(freq_bins_indices == i)
                        else 0
                    )
                    for i in unique_bins_indices
                ]
            )

            sum_binned_residual_magnitudes = np.array(
                [
                    (
                        np.sum(transformed_residual[freq_bins_indices == i])
                        if np.any(freq_bins_indices == i)
                        else 0
                    )
                    for i in unique_bins_indices
                ]
            )

            for j, bin_index in enumerate(unique_bins_indices):
                feati[f"seasonal_fft_mean_{bin_index}"] = (
                    mean_binned_seasonal_magnitudes[j]
                )
                feati[f"seasonal_fft_sum_{bin_index}"] = sum_binned_seasonal_magnitudes[
                    j
                ]
                feati[f"residual_fft_mean_{bin_index}"] = (
                    mean_binned_residual_magnitudes[j]
                )
                feati[f"residual_fft_sum_{bin_index}"] = sum_binned_residual_magnitudes[
                    j
                ]

            pc["features"] = feati
            pc["features_raw"] = featrawi

            data[i] = pc

        analyses.data = data
        return analyses


class PressureCurvesMethodScaleFeaturesNative(ProcessingMethod):
    """
    Scales features of pressure curves using a native algorithm.
    """

    def __init__(self):
        super().__init__()
        self.data_type = "PressureCurves"
        self.method = "ScaleFeatures"
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

        unique_methods = set()
        for pc in data:
            if pc["features"]["method"] not in unique_methods:
                unique_methods.add(pc["features"]["method"])

        feature_template = data[0]["features"].copy()
        for key in feature_template.keys():
            feature_template[key] = 0

        unique_methods_max = []
        for i, method in enumerate(unique_methods):
            unique_methods_max.append(feature_template.copy())
            unique_methods_max[i]["method"] = method

        for i, pc in enumerate(data):
            feat = pc["features"]
            for j, method in enumerate(unique_methods):
                if feat["method"] == method:
                    for key, item in feat.items():
                        if key == "method":
                            continue
                        if item > unique_methods_max[j][key]:
                            unique_methods_max[j][key] = item

        for i, pc in enumerate(data):
            feat = pc["features"]
            for j, method in enumerate(unique_methods):
                if feat["method"] == method:
                    for key, item in feat.items():
                        if key == "method":
                            continue
                        if unique_methods_max[j][key] != 0:
                            feat[key] = item / unique_methods_max[j][key]
                        else:
                            feat[key] = 0
                    break

            pc["features"] = feat
            data[i] = pc

        analyses.data = data
        return analyses
