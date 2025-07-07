"""
This module contains processing methods for device analyses data.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
#from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter 
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
        #bins (int): The number of bins for Fast Fourier Transformation (FFT). Default is 4.

    Details:
        The method extracts features from pressure curves using seasonal decomposition and FFT, adding entries named "features" and "features_raw" to each dict in the data list of the PressureCurvesAnalyses instance.
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
            #- seasonal_fft_mean_{i}: The mean of the seasonal FFT for bin {i}.
            #- seasonal_fft_sum_{i}: The sum of the seasonal FFT for bin {i}.
            #- residual_fft_mean_{i}: The mean of the residual FFT for bin {i}.
            #- residual_fft_sum_{i}: The sum of the residual FFT for bin {i}.
        The "features_raw" include:
            - trend: The trend component from seasonal decomposition.
            - seasonal: The seasonal component from seasonal decomposition.
            - residual: The residual component from seasonal decomposition.
            #- seasonal_fft: The FFT of the seasonal component.
            #- residual_fft: The FFT of the residual component.
            #- sample_spacing: The sample spacing used for FFT.
            #- freq_bins: The frequency bins used for FFT.
            #- freq_bin_edges: The edges of the frequency bins.
            #- freq_bins_indices: The indices of the frequency bins.

    """
    
    def __init__(self, period: int = 10, bins: int = 4):
    #def __init__(self, period: int = 10):
        super().__init__()
        self.data_type = "PressureCurvesAnalyses"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"period": period, "bins": bins}
        #self.parameters = {"period": period}
        
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
            # "batch_position": 0,
            # "run_type": "",
            # "idle_time": 0,
            "area": 0,
            "pressure_max": 0,
            "pressure_min": 0,
            "pressure_mean": 0,
            "pressure_std": 0,
            "pressure_range": 0,
            "runtime": 0,
            # "residual_median": 0,
            "residual_noise": 0,
            "residual_std": 0,
            # "residual_sum": 0,
            # "residual_max": 0,
            # "seasonal_amplitude": 0,
            # "skewness": 0,
            # "kurtosis": 0,
        }

        #for i in range(self.parameters["bins"] + 1):
        #    if i == 0:
        #        continue
            # features_template[f"seasonal_fft_max_{i}"] = 0
        #    features_template[f"residual_fft_max_{i}"] = 0

        for i in range(1, self.parameters["bins"] + 1):
            features_template[f"max_amplitude_bin_{i}"] = 0
            features_template[f"min_amplitude_bin_{i}"] = 0
            features_template[f"amplitude_range_bin_{i}"] = 0

        features_raw_transform = {
            "trend": [],
            "seasonal": [],
            "residual": [],
            #"pressure_baseline_corrected": [],
            #"seasonal_fft": [],
            #"residual_fft": [],
            #"sample_spacing": 0,
            #"freq_bins": [],
            #"freq_bin_edges": [],
            #"freq_bins_indices": [],
        }

        for i, pc in enumerate(data):
            feati = features_template.copy()
            featrawi = features_raw_transform.copy()

            # feati["batch_position"] = pc["batch_position"]

            # if pc["sample"] == "Blank":
            #     feati["run_type"] = 0
            # else:
            #     feati["run_type"] = 1

            # feati["idle_time"] = pc["idle_time"]
            feati["pressure_max"] = max(pc["pressure_var"])
            feati["pressure_min"] = min(pc["pressure_var"])
            feati["pressure_mean"] = sum(pc["pressure_var"]) / len(pc["pressure_var"])
            feati["pressure_std"] = (
                sum((x - feati["pressure_mean"]) ** 2 for x in pc["pressure_var"])
                / len(pc["pressure_var"])
            ) ** 0.5
            feati["pressure_range"] = feati["pressure_max"] - feati["pressure_min"]

            feati["runtime"] = pc.get("runtime", 0)

            pressure_vector = np.array(pc["pressure_var"])

            # Apply baseline correction. Comparing multiple algorithms to choose the best one.
            # """
            # 1. Simple Moving Average
            # - smoothed version of the original signal, with each point replaced by the average of itself and its <window_size - 1> nearest neighbors
            # - this smoothed vector is then subtracted from the original pressure vector to obtain the baseline corrected vector while retaining noise
            # """
            # window_size = self.parameters["period"] if self.parameters["period"] % 2 != 0 else self.parameters["period"] - 1
            # edges = window_size // 2
            # smoothed_vector = np.convolve(pressure_vector, np.ones(window_size) / window_size, mode='same')
            # # baseline correction by subtracting the smoothed vector from the original pressure vector
            # baseline_corrected_vector = pressure_vector - smoothed_vector
            # # Remove the elements from the beginning and end to avoid edge effects. Typically <window_size // 2>
            # baseline_corrected_vector = baseline_corrected_vector[edges:-edges]

            """
            2. Savitzky-Golay Filter
            - applies a polynomial smoothing filter to the data, which is particularly effective for preserving features of the data while reducing noise
            - uses a sliding window to fit a polynomial to the data points within the window, and then replaces the central point with the value of the polynomial at that point
            """
            #from scipy.signal import savgol_filter - to be manually implemented
            window_size = self.parameters["period"] if self.parameters["period"] % 2 != 0 else self.parameters["period"] + 1  # Must be odd
            poly_order = 2  # Polynomial order
            smoothed_vector = savgol_filter(pressure_vector, window_size, poly_order)
            baseline_corrected_vector = pressure_vector - smoothed_vector
            # Remove the elements from the beginning and end to avoid edge effects. Typically <window_size // 2>
            # baseline_corrected_vector = baseline_corrected_vector[1:-1]

            # featrawi["pressure_baseline_corrected"] = baseline_corrected_vector

            # np.array_split puts extra elements in the first bin if the length of the vector is not divisible by the number of bins
            num_bins = np.array_split(baseline_corrected_vector, self.parameters["bins"])
            for i, vector_bin in enumerate(num_bins):
                if len(vector_bin) == 0:
                    continue
                max_amplitude = np.max(vector_bin)
                min_amplitude = np.min(vector_bin)
                amplitude_range = max_amplitude - min_amplitude
                #print(f"Method: {pc['method']}, Sample: {pc['sample']}, Bin {i + 1}: Max: {max_amplitude}, Min: {min_amplitude}, Range: {amplitude_range}")
                feati[f"max_amplitude_bin_{i + 1}"] = max_amplitude
                feati[f"min_amplitude_bin_{i + 1}"] = min_amplitude
                feati[f"amplitude_range_bin_{i + 1}"] = amplitude_range
            
            # Confirm whether pressure vector is clipped after baseline correction and amplitude binning
            pressure_vector = pressure_vector[1:-1]

            time_var = np.array(pc["time_var"])
            time_var = time_var[1:-1]

            feati["area"] = np.trapz(pressure_vector, x=pc["time_var"][1:-1])

            # feati["skewness"] = skew(pressure_vector)
            # feati["skewness"] = feati["skewness"] + abs(feati["skewness"]) + 1

            # feati["kurtosis"] = kurtosis(pressure_vector)
            # feati["kurtosis"] = feati["kurtosis"] + abs(feati["kurtosis"]) + 1

            decomp = seasonal_decompose(
                pd.to_numeric(pressure_vector),
                model="additive",
                period=self.parameters["period"],
                extrapolate_trend="freq",
            )

            featrawi["trend"] = decomp.trend
            featrawi["seasonal"] = decomp.seasonal
            featrawi["residual"] = decomp.resid

            #transformed_seasonal = np.fft.fft(decomp.seasonal)
            #transformed_seasonal = abs(transformed_seasonal)
            #transformed_residual = np.fft.fft(decomp.resid)
            #transformed_residual = abs(transformed_residual)

            residual = decomp.resid
            # raise residual to all positive values
            # transformed_residual = np.abs(transformed_residual)
            valid_mask = ~np.isnan(residual) & ~np.isnan(time_var)
            residual_derivative = np.gradient(residual[valid_mask])
            time_var_derivative = np.gradient(time_var[valid_mask])

            feati["residual_noise"] = np.std(residual_derivative / time_var_derivative)
            # feati["residual_median"] = np.mean(residual[valid_mask])
            feati["residual_std"] = np.std(residual[valid_mask])
            # feati["residual_sum"] = np.sum(transformed_residual)
            # feati["residual_max"] = np.max(residual[valid_mask])

            # seasonal = decomp.seasonal
            # feati["seasonal_amplitude"] = np.max(seasonal) - np.min(seasonal)

            #if len(time_var) > 1:
            #    sample_spacing = np.mean(np.diff(time_var))
            #else:
            #    sample_spacing = 1.0

            #freq_bins = np.fft.fftfreq(len(decomp.resid))  # d=sample_spacing

            #positive_freqs = freq_bins > 0
            #freq_bins = freq_bins[positive_freqs]
            #transformed_seasonal = transformed_seasonal[positive_freqs]
            #transformed_residual = transformed_residual[positive_freqs]

            #featrawi["seasonal_fft"] = transformed_seasonal
            #featrawi["residual_fft"] = transformed_residual

            #featrawi["sample_spacing"] = sample_spacing
            #featrawi["freq_bins"] = freq_bins

            #num_bins = 4
            #freq_bin_edges = np.histogram_bin_edges(freq_bins, bins=num_bins)
            #featrawi["freq_bin_edges"] = freq_bin_edges

            #freq_bins_indices = np.digitize(freq_bins, freq_bin_edges, right=True)
            #unique_bins_indices = np.unique(freq_bins_indices)
            #featrawi["freq_bins_indices"] = freq_bins_indices

            #for bin_index in unique_bins_indices:
            #    if bin_index == 0:
            #        continue
                # transformed_seasonal_bin = transformed_seasonal[
                #     freq_bins_indices == bin_index
                # ]
                # max_seasonal_bin = np.max(transformed_seasonal_bin)
                # feati[f"seasonal_fft_max_{bin_index}"] = max_seasonal_bin

                #transformed_residual_bin = transformed_residual[
                #    freq_bins_indices == bin_index
                #]
                #max_residual_bin = np.max(transformed_residual_bin)
                #feati[f"residual_fft_max_{bin_index}"] = max_residual_bin

            # mean_binned_seasonal_magnitudes = np.array(
            #     [
            #         (
            #             np.max(transformed_seasonal[freq_bins_indices == i])
            #             if np.any(freq_bins_indices == i)
            #             else 0
            #         )
            #         for i in unique_bins_indices
            #     ]
            # )

            # # sum_binned_seasonal_magnitudes = np.array(
            # #     [
            # #         (
            # #             np.sum(transformed_seasonal[freq_bins_indices == i])
            # #             if np.any(freq_bins_indices == i)
            # #             else 0
            # #         )
            # #         for i in unique_bins_indices
            # #     ]
            # # )

            # mean_binned_residual_magnitudes = np.array(
            #     [
            #         (
            #             np.max(transformed_residual[freq_bins_indices == i])
            #             if np.any(freq_bins_indices == i)
            #             else 0
            #         )
            #         for i in unique_bins_indices
            #     ]
            # )

            # # sum_binned_residual_magnitudes = np.array(
            # #     [
            # #         (
            # #             np.sum(transformed_residual[freq_bins_indices == i])
            # #             if np.any(freq_bins_indices == i)
            # #             else 0
            # #         )
            # #         for i in unique_bins_indices
            # #     ]
            # # )

            # for bin_index in unique_bins_indices:
            #     feati[f"seasonal_fft_max_{bin_index}"] = (
            #         mean_binned_seasonal_magnitudes[bin_index]
            #     )
            #     # feati[f"seasonal_fft_sum_{bin_index}"] = sum_binned_seasonal_magnitudes[
            #     #     bin_index
            #     # ]
            #     feati[f"residual_fft_max_{bin_index}"] = (
            #         mean_binned_residual_magnitudes[bin_index]
            #     )
            #     # feati[f"residual_fft_sum_{bin_index}"] = sum_binned_residual_magnitudes[
            #     #     bin_index
            #     # ]

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
