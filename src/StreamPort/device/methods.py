"""
This module contains processing methods for device analyses data.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import preprocessing as scaler
from src.StreamPort.core import ProcessingMethod
from src.StreamPort.device.analyses import PressureCurvesAnalyses


class PressureCurvesMethodAssignBatchPositionNative(ProcessingMethod):
    """
    Assigns batch position and calculated iddle time to pressure curves using a native algorithm.
    """

    def __init__(self):
        super().__init__()
        self.data_type = "PressureCurvesAnalyses"
        self.method = "AssignBatchPosition"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {}

    def run(self, analyses: PressureCurvesAnalyses) -> PressureCurvesAnalyses:
        """
        Assigns batch position and calculated iddle time to pressure curves.
        Args:
            analyses (PressureCurvesAnalyses): The PressureCurvesAnalyses instance to process.
        Returns:
            PressureCurvesAnalyses: The processed PressureCurvesAnalyses instance with batch position and idle time assigned.
        """
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        data = sorted(data, key=lambda x: x["timestamp"])

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

    Details:
        The method extract features from pressure curves using seasonal decomposition and FFT, adding entries named "features" and "features_raw" to each dict in the data list of the PressureCurvesAnalyses instance.
        The "features" include:
            - batch_position: The position of the batch in the analysis.
            - run_type: The type of run (0 for Blank, 1 for Sample).
            - idle_time: The time between the current and previous pressure curve.
            - pressure_max: The maximum pressure value.
            - pressure_min: The minimum pressure value.
            - pressure_mean: The mean pressure value.
            - pressure_std: The standard deviation of the pressure values.
            - pressure_range: The range of the pressure values.
            - runtime: The runtime of the analysis.
            - residual_mean: The mean of the residuals from seasonal decomposition.
            - residual_std: The standard deviation of the residuals from seasonal decomposition.
            - residual_sum: The sum of the residuals from seasonal decomposition.
            - residual_max: The maximum value of the residuals from seasonal decomposition.
            - seasonal_fft_mean_{i}: The mean of the seasonal FFT for bin {i}.
            - seasonal_fft_sum_{i}: The sum of the seasonal FFT for bin {i}.
            - residual_fft_mean_{i}: The mean of the residual FFT for bin {i}.
            - residual_fft_sum_{i}: The sum of the residual FFT for bin {i}.
        The "features_raw" include:
            - trend: The trend component from seasonal decomposition.
            - seasonal: The seasonal component from seasonal decomposition.
            - residual: The residual component from seasonal decomposition.
            - seasonal_fft: The FFT of the seasonal component.
            - residual_fft: The FFT of the residual component.
            - sample_spacing: The sample spacing used for FFT.
            - freq_bins: The frequency bins used for FFT.
            - freq_bin_edges: The edges of the frequency bins.
            - freq_bins_indices: The indices of the frequency bins.

    """

    def __init__(self, period: int = 10, bins: int = 4):
        super().__init__()
        self.data_type = "PressureCurvesAnalyses"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"period": period, "bins": bins}

    def run(self, analyses: PressureCurvesAnalyses) -> PressureCurvesAnalyses:
        """
        Extracts features from pressure curves using seasonal decomposition and FFT.
        Args:
            analyses (PressureCurvesAnalyses): The PressureCurvesAnalyses instance to process.
        Returns:
            PressureCurvesAnalyses: The processed PressureCurvesAnalyses instance with features extracted.
        """
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        features_template = {
            "batch_position": 0,
            "run_type": "",
            "idle_time": 0,
            "pressure_max": 0,
            "pressure_min": 0,
            "pressure_mean": 0,
            "pressure_std": 0,
            "pressure_range": 0,
            "runtime": 0,
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

        for i, pc in enumerate(data):
            feati = features_template.copy()
            featrawi = features_raw_transform.copy()

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
            feati["runtime"] = pc["runtime"]

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


class PressureCurvesMethodScaleFeaturesScalerSklearn(ProcessingMethod):
    """
    Scales features of pressure curves using a scaler from sklearn.

    Args:
        scaler_type (str): The type of scaler to use. Options are:
            - "MinMaxScaler"
            - "StandardScaler"
            - "RobustScaler"
            - "MaxAbsScaler"
            - "MaxNormalizer"
    """

    def __init__(self, scaler_type: str = "MinMaxScaler"):
        super().__init__()
        self.data_type = "PressureCurvesAnalyses"
        self.method = "ScaleFeatures"
        self.algorithm = "Sklearn"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"type": scaler_type}

    def run(self, analyses: PressureCurvesAnalyses) -> PressureCurvesAnalyses:
        """
        Scales features of pressure curves using a scaler from sklearn.
        Args:
            analyses (PressureCurvesAnalyses): The PressureCurvesAnalyses instance to process.
        Returns:
            PressureCurvesAnalyses: The processed PressureCurvesAnalyses instance with scaled features.
        """
        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses

        scaler_type = self.parameters["type"]
        if scaler_type == "MinMaxScaler":
            scaler_model = scaler.MinMaxScaler()
        elif scaler_type == "StandardScaler":
            scaler_model = scaler.StandardScaler()
        elif scaler_type == "RobustScaler":
            scaler_model = scaler.RobustScaler()
        elif scaler_type == "MaxAbsScaler":
            scaler_model = scaler.MaxAbsScaler()
        elif scaler_type == "MaxNormalizer":
            scaler_model = scaler.Normalizer(norm="max")
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        methods_set = analyses.get_methods()
        mt = analyses.get_metadata()
        df = analyses.get_features()

        for col in df.columns:
            df[col] = df[col].astype(float)

        for method in methods_set:
            mask = mt["method"] == method
            df.loc[mask, :] = scaler_model.fit_transform(df.loc[mask, :])

        for i, pc in enumerate(data):
            pc["features"] = df.iloc[i].to_dict()
            data[i] = pc

        analyses.data = data
        return analyses
