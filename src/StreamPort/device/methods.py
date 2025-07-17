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
        window_size (int): The window size/resolution for baseline correction. Default is 7.
        bins (int): The number of bins for Fast Fourier Transformation (FFT). Default is 4.
        crop (int): The number of elements to crop from the beginning and end of the pressure vector to remove unwanted artifacts. Default is 2.

    Details:
        The method extracts features from pressure curves using seasonal decomposition and FFT, adding entries named "features" and "features_raw" to each dict in the data list of the PressureCurvesAnalyses instance.
        The "features" include:
            - area: The area under the pressure curve.
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
            - abs_deviation: The absolute deviation of the pressure values in each bin.
        The "features_raw" include:
            - trend: The trend component from seasonal decomposition.
            - seasonal: The seasonal component from seasonal decomposition.
            - residual: The residual component from seasonal decomposition.
    """

    def __init__(self, period: int = 10, window_size: int = 7, bins: int = 4, crop: int = 2):
        super().__init__()
        self.data_type = "PressureCurvesAnalyses"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {"period": period, "window_size": window_size, "bins": bins, "crop": crop}
        
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
        
        #remove StandBy samples in case of error-lc
        data = [pc for pc in data if pc["sample"] != "StandBy"]

        features_template = {
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

        features_raw_transform = {
            "trend": [],
            "seasonal": [],
            "residual": [],
            "pressure_baseline_corrected": [],
        }

        # A small subset of curves from each method is missing a datapoint. Could indicate an anomaly, may also be reflected in the true runtimes. 
        # Solution is to pad all shorter curves for each unique method with zeros to indicate the run ended uncharacteristically and handle missing values to enforce a unified time axis.
        method_time_vars = {}

        # Find correct length per method
        for pc in data:
            method = pc["method"]
            time_var = np.array(pc["time_var"])

            if method not in method_time_vars:
                method_time_vars[method] = time_var
            elif len(time_var) > len(method_time_vars[method]):
                method_time_vars[method] = time_var

        for i, pc in enumerate(data):

            # Zeros in place of actual values in these anomalous curves would set them apart in the bin amplitude values calculated below.
            method = pc["method"]
            time_var = np.array(pc["time_var"])
            pressure_vector = np.array(pc["pressure_var"])
            target_time_var = method_time_vars[method]

            if len(time_var) < len(target_time_var):
                pc["pressure_var"] = np.concatenate([
                    pressure_vector,
                    np.zeros(len(target_time_var) - len(pressure_vector))
                ])
                pc["time_var"] = target_time_var
            
            feati = features_template.copy()
            featrawi = features_raw_transform.copy()

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
            time_var = np.array(pc["time_var"])
            
            # Crop the pressure vector to remove unwanted artifacts before processing
            pressure_vector = pressure_vector[self.parameters["crop"]:-self.parameters["crop"]] 
            time_var = time_var[self.parameters["crop"]:-self.parameters["crop"]]

            pc["pressure_var"] = pressure_vector
            pc["time_var"] = time_var

            if len(pressure_vector) != len(time_var):
                raise ValueError(
                    "Pressure vector and time variable must have the same length!"
                )

            # Apply baseline correction
            # """
            # 1. Simple Moving Average
            # - smoothed version of the original signal, with each point replaced by the average of itself and its <window_size - 1> nearest neighbors
            # - this smoothed vector is then subtracted from the original pressure vector to obtain the baseline corrected vector while retaining noise
            # """
            # smoothed_vector = np.convolve(pressure_vector, np.ones(window_size) / window_size, mode='same')
            # # baseline correction by subtracting the smoothed vector from the original pressure vector
            # baseline_corrected_vector = pressure_vector - smoothed_vector
            # # Remove the elements from the beginning and end to avoid edge effects. Typically <window_size // 2>
            # baseline_corrected_vector = baseline_corrected_vector[crop:-crop]

            """
            2. Savitzky-Golay Filter
            - applies a polynomial smoothing filter to the data, which is particularly effective for preserving features of the data while reducing noise
            - uses a sliding window to fit a polynomial to the data points within the window, and then replaces the central point with the value of the polynomial at that point
            """
            #from scipy.signal import savgol_filter - to be manually implemented
            if self.parameters["window_size"] % 2 == 0:
                self.parameters["window_size"] + 1  # Must be odd

            poly_order = 2  # Polynomial order
            smoothed_vector = savgol_filter(pressure_vector, self.parameters["window_size"], poly_order)
            baseline_corrected_vector = pressure_vector - smoothed_vector
            
            #baseline_correction or any such operation may introduce NaN values
            baseline_corrected_vector = np.nan_to_num(baseline_corrected_vector, 
                                                      posinf = np.max(baseline_corrected_vector), 
                                                      neginf = np.min(baseline_corrected_vector))

            featrawi["pressure_baseline_corrected"] = baseline_corrected_vector

            # np.array_split puts extra elements in the first bin if the length of the vector is not divisible by the number of bins
            vector_bins = np.array_split(baseline_corrected_vector, self.parameters["bins"])
            bin_edges = []
            start_edge = 0
            for bin in vector_bins:
                end_edge = start_edge + len(bin) - 1

                key = f"abs_deviation_{time_var[start_edge].round(3)}_{time_var[end_edge].round(3)}" 
                feati[key] = np.max(bin) - np.min(bin)
                
                bin_edges.append([start_edge, end_edge])

                start_edge = end_edge + 1  

            featrawi["bin_edges"] = bin_edges

            feati["area"] = np.trapz(pressure_vector, x=time_var)

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
