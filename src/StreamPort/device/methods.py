"""
This module contains processing methods for device analyses data.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
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
        Assigns batch position and calculated idle time to pressure curves.
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

        features_template = {}

        features_raw_transform = {
            "trend": [],
            "seasonal": [],
            "residual": [],
            "pressure_baseline_corrected": [],
        }

        # a small subset of curves from each method/batch is missing a datapoint. Could indicate an anomaly, may also be reflected in the true runtimes. 
        # pad all shorter curves for each unique method with zeros to indicate the run ended uncharacteristically and handle missing values to enforce a unified time axis.
        method_time_vars = {}

        # Find correct length per method/batch
        for pc in data:
            method = pc["method"]
            time_var = np.nan_to_num(np.array(pc["time_var"]), nan=0.0)

            if method not in method_time_vars:
                method_time_vars[method] = time_var
            elif len(time_var) > len(method_time_vars[method]):
                method_time_vars[method] = time_var

        for i, pc in enumerate(data):

            # padding with zeros in place of actual values in these anomalous curves would set them apart in the bin amplitude values calculated below.
            method = pc["method"]
            time_var = np.array(pc["time_var"])
            pressure_vector = np.nan_to_num(np.array(pc["pressure_var"]), nan=0.0)
            target_time_var = method_time_vars[method]

            if len(time_var) < len(target_time_var):
                pc["pressure_var"] = np.concatenate([
                    pressure_vector,
                    np.zeros(len(target_time_var) - len(time_var))
                ])
                pc["time_var"] = target_time_var
            
            # padding done. Calculate features
            feati = features_template.copy()
            featrawi = features_raw_transform.copy()
            
            # crop the pressure vector to remove unwanted artifacts before processing
            pressure_vector = np.array(pc["pressure_var"])
            time_var = np.array(pc["time_var"])

            pressure_vector = pressure_vector[self.parameters["crop"]:-self.parameters["crop"]] 
            time_var = time_var[self.parameters["crop"]:-self.parameters["crop"]]
            
            pc["pressure_var"] = pressure_vector
            pc["time_var"] = time_var

            feati["runtime"] = pc.get("runtime", 0)

            if len(pressure_vector) != len(time_var):
                raise ValueError(
                    "Pressure vector and time variable must have the same length!"
                )

            # """
            # Savitzky-Golay Filter
            # - applies a polynomial smoothing filter to the data, which is particularly effective for preserving features of the data while reducing noise
            # - uses a sliding window to fit a polynomial to the data points within the window, and then replaces the central point with the value of the polynomial at that point
            # """
            # #savgol_filter - to be manually implemented
            # if self.parameters["window_size"] % 2 == 0:
            #     self.parameters["window_size"] += 1  # Must be odd

            # poly_order = 2  # Polynomial order
            # smoothed_vector = savgol_filter(pressure_vector, self.parameters["window_size"], poly_order)

            """
            SNIP(Statistical Non-linear Iterative Peak)
            -  
            """
            # apply a double logarithm transformation to the pressure vector
            lls_vector = np.log(np.log(np.sqrt(pressure_vector + 1) + 1) + 1)
            # Define a function to compute the minimum filter
            def min_filter(lls_vector, m):
                """Applies the SNIP minimum filter"""
                lls_filtered = np.copy(lls_vector)
                for i in range(m, len(lls_vector) - m):
                    lls_filtered[i] = min(lls_vector[i], (lls_vector[i-m] + lls_vector[i + m])/2)
                return lls_filtered

            # Apply the filter for the first 100 iterations
            lls_filtered = np.copy(lls_vector)
            for m in range(5):
                lls_filtered = min_filter(lls_vector, m)

            smoothed_vector = (np.exp(np.exp(lls_filtered) - 1) - 1) ** 2 - 1
            baseline_corrected_vector = pressure_vector - smoothed_vector
            
            #baseline_correction or any such operation may introduce NaN values
            baseline_corrected_vector = np.nan_to_num(baseline_corrected_vector, 
                                                      posinf = np.max(baseline_corrected_vector), 
                                                      neginf = np.min(baseline_corrected_vector))

            featrawi["pressure_baseline_corrected"] = baseline_corrected_vector
            #baseline correction done

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

            # define bin edges by index
            split_idxs = np.linspace(0, len(target_time_var), self.parameters["bins"] + 1, dtype=int)

            bin_edges = []

            for i in range(self.parameters["bins"]):
                start_idx = split_idxs[i]
                end_idx = split_idxs[i + 1] - 1

                if end_idx >= len(pressure_vector):
                    end_idx = len(pressure_vector) - 1  

                start_time = round(target_time_var[start_idx], 3)
                end_time = round(target_time_var[end_idx], 3)

                # area under curve
                key_area = f"area_{start_time}_{end_time}"
                feati[key_area] = np.trapz(pressure_vector[start_idx:end_idx + 1], x = target_time_var[start_idx:end_idx + 1])

                # base statisical features
                key_min = f"min_{start_time}_{end_time}"
                feati[key_min] = min(pressure_vector[start_idx:end_idx + 1])
                
                key_max = f"max_{start_time}_{end_time}"
                feati[key_max] = max(pressure_vector[start_idx:end_idx + 1])

                key_mean = f"mean_{start_time}_{end_time}"
                feati[key_mean] = sum(pressure_vector[start_idx:end_idx + 1]) / (end_idx + 1 -start_idx)

                key_std = f"std_{start_time}_{end_time}"
                feati[key_std] = (
                sum((x - feati[key_mean]) ** 2 for x in pressure_vector[start_idx:end_idx + 1])
                / (end_idx + 1 -start_idx)
                ) ** 0.5

                key_range = f"range_{start_time}_{end_time}"
                feati[key_range] = feati[key_max] - feati[key_min]

                # std in noise component of curve
                key_res_std = f"residual_std_{start_time}_{end_time}"
                resid_bin = residual[start_idx:end_idx + 1]
                valid_bin_mask = ~np.isnan(resid_bin) & ~np.isnan(target_time_var[start_idx:end_idx + 1])
                feati[key_res_std] = np.std(resid_bin[valid_bin_mask]) if np.any(valid_bin_mask) else np.nan

                """
                Out for now, residual_noise shows the most deviation among features. Could hinder visibility of other features. 
                """
                # absolute noise 
                # key_res_noise = f"residual_noise_{start_time}_{end_time}"
                # if np.sum(valid_bin_mask) >= 2:  # gradient needs at least 2 points
                #     resid_deriv = np.gradient(resid_bin[valid_bin_mask])
                #     time_deriv = np.gradient(target_time_var[start_idx:end_idx + 1][valid_bin_mask])
                #     res_noise = np.std(resid_deriv / time_deriv)
                # else:
                #     res_noise = np.nan
                # feati[key_res_noise] = res_noise

                if pressure_vector[start_idx] == 0:
                    pressure_vector[start_idx] = 0.01 #avoid division by 0

                # relative change
                key_relc = f"relative_change_{start_time}_{end_time}"  # deviation in curves caused by e.g. Open Oven lost in smoothing. Pressure curve RoC in bins to id exact moment error causes change
                feati[key_relc] = (pressure_vector[end_idx] - pressure_vector[start_idx]) / pressure_vector[start_idx] #abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / pressure_vector[start_idx]) 
                
                """
                Out for now, roc shows the most deviation among features. Could hinder visibility of other features. 
                """
                # rate of change. Absolute value used to avoid negative roc and relative change values
                # key_roc = f"roc_{start_time}_{end_time}"
                # feati[key_roc] = abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / target_time_var[end_idx] - target_time_var[start_idx]) #abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / target_time_var[end_idx] - target_time_var[start_idx])               
                
                # absolute value of minute fluctuations after baseline correction 
                key_dev = f"abs_deviation_{start_time}_{end_time}"  # absolute fluctuation in the bin without baseline pressure value to catch small deviations
                feati[key_dev] = np.nanmax(baseline_corrected_vector[start_idx:end_idx + 1]) - np.nanmin(baseline_corrected_vector[start_idx:end_idx + 1])

                bin_edges.append([start_time, end_time])

            featrawi["bin_edges"] = bin_edges
      
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
