"""
This module contains processing methods for device analyses data.
"""

import pandas as pd
import numpy as np
import re
from statsmodels.tsa.seasonal import seasonal_decompose
#from scipy.signal import savgol_filter
from scipy.signal import convolve2d 
from scipy.optimize import curve_fit
#from scipy.optimize import least_squares #better alternative to curve_fit in case of high noise content in MSD
from sklearn import preprocessing as scaler
from core import ProcessingMethod
from device.analyses import PressureCurvesAnalyses
from device.analyses import MassSpecAnalyses


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
        bins (int): The number of bins for feature extraction. Default is 4.
        crop (int): The number of elements to crop from the beginning and end of the pressure vector to remove unwanted artifacts. Default is 2.

    Details:
        The method extracts features from pressure curves using seasonal decomposition and binning, adding entries named "features" and "features_raw" to each dict in the data list of the PressureCurvesAnalyses instance.
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
            - relc: Relative change of the pressure at time intervals.
            - roc: Rate of change of the pressure at time intervals.
            - abs_deviation: The absolute deviation of the pressure values in each bin.
        The "features_raw" include:
            - trend: The trend component from seasonal decomposition.
            - seasonal: The seasonal component from seasonal decomposition.
            - residual: The residual component from seasonal decomposition.
            - pressure_baseline_corrected: The pressure curve with its baseline removed.
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

            """
            Savitzky-Golay Filter
            - applies a polynomial smoothing filter to the data, which is particularly effective for preserving features of the data while reducing noise
            - uses a sliding window to fit a polynomial to the data points within the window, and then replaces the central point with the value of the polynomial at that point
            """
            # if self.parameters["window_size"] % 2 == 0:
            #     self.parameters["window_size"] += 1  # Must be odd

            # poly_order = 2  # Polynomial order
            # smoothed_vector = savgol_filter(pressure_vector, self.parameters["window_size"], poly_order)

            """
            SNIP(Statistical Non-linear Iterative Peak-clipping)
            - Iteratively estimates and removes the broad background trends from sharp spectral features of a signal, like peaks
            - Useful for spectra with overlapping peaks or variable baselines.
            ## Baseline Correction is expensive and unused except for calculating abs_deviation. Uncomment when needed.
            """
            # # apply a double logarithm transformation to the pressure vector
            # lls_vector = np.log(np.log(np.sqrt(pressure_vector + 1) + 1) + 1)
            # # Define a function to compute the minimum filter
            # def min_filter(lls_vector, m):
            #     """Applies the SNIP minimum filter"""
            #     lls_filtered = np.copy(lls_vector)
            #     for i in range(m, len(lls_vector) - m):
            #         lls_filtered[i] = min(lls_vector[i], (lls_vector[i-m] + lls_vector[i + m])/2)
            #     return lls_filtered

            # # Apply the filter for the first 5 iterations
            # lls_filtered = np.copy(lls_vector)
            # for m in range(5):
            #     lls_filtered = min_filter(lls_vector, m)

            # smoothed_vector = (np.exp(np.exp(lls_filtered) - 1) - 1) ** 2 - 1

            # # subtract the smoothed vector from the original vector to remove the baseline
            # baseline_corrected_vector = pressure_vector - smoothed_vector
            
            # # baseline correction or any such operation may introduce NaN values
            # baseline_corrected_vector = np.nan_to_num(baseline_corrected_vector, 
            #                                           posinf = np.max(baseline_corrected_vector), 
            #                                           neginf = np.min(baseline_corrected_vector))

            # featrawi["pressure_baseline_corrected"] = baseline_corrected_vector
            # # baseline correction done

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
                # # absolute noise 
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

                # # relative change
                # key_relc = f"relative_change_{start_time}_{end_time}"  # deviation in curves caused by e.g. Open Oven lost in smoothing. Pressure curve RoC in bins to id exact moment error causes change
                # feati[key_relc] = (pressure_vector[end_idx] - pressure_vector[start_idx]) / pressure_vector[start_idx] #abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / pressure_vector[start_idx]) 
                
                """
                Out for now, roc shows the most deviation among features. Could hinder visibility of other features. 
                """
                # # rate of change. Absolute value used to avoid negative roc and relative change values
                key_roc = f"roc_{start_time}_{end_time}"
                feati[key_roc] = abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / target_time_var[end_idx] - target_time_var[start_idx]) #abs((pressure_vector[end_idx] - pressure_vector[start_idx]) / target_time_var[end_idx] - target_time_var[start_idx])               
                
                # # absolute value of minute fluctuations after baseline correction 
                # key_dev = f"abs_deviation_{start_time}_{end_time}"  # absolute fluctuation in the bin without baseline pressure value to catch small deviations
                # feati[key_dev] = np.nanmax(baseline_corrected_vector[start_idx:end_idx + 1]) - np.nanmin(baseline_corrected_vector[start_idx:end_idx + 1])

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


class MassSpecMethodExtractFeaturesNative(ProcessingMethod):
    """
    Method to extract features from ms data using a native algorithm.

    Args:
        data (str): "sim" or "tic" based on user's choice. Defaults to sim, and targets the closest mz if the mz input parameter is invalid.
        rt (float): The retention time for the target to be analysed.
        mz (float): The mz for the target.
        rt_window (int): The minimum distance by seconds/number of rt entries before and after current one to be considered when finding peaks (default = 8s). Any peaks within this distance from each other will be disregarded.
        mz_window (float): Range of adjacent mz value to be considered for 2D tile/window creation. Defaults to 1.0 Da (mz(s) within 1.0 to the current one, here, totalling 1 mz before and after the target).
        smooth (int | bool): User may provide a window size and choose whether the signal must be pre-treated using smoothing. Default is None(False). Note: If an int n is passed, the window will be nxn over the 2D intensity array, where n is always automatically limited to the values 3, 5, and 7.
        exclude (list | str): Choice of whether to exclude "Flush", "Blank" or other such runs from the analysis. This will set the features for the respective samples to None, without removing the samples.

    Methods:
        - _apply_gaussian_filter: Performs gaussian smoothing on the 2D intensity matrix before feature extraction with a default 3x3 kernel.
        - _gaussian_2d: Performs gaussian fitting for a 2D peak space.
        - _fit_gaussian: Fits the 1D signal to a gaussian curve and optimizes the values A, mu, sigma using the ADAM optimizer.      
        - _get_r2: Calculates the rsquared value from the optimized fit values.
        - _show_tile_debug_plot: Plots the target window/tile selected by the user, along with the gaussian surface (red) fit to it. Returns plots for all failed fits for debugging and a plot of the last sample when fit is successful.
        - run: Uses the helper functions to extract features from the analyses based on the target selected by the user and returns the analyses with the features.

    Details:
        The method extracts features from ms data matrix, for each selected target (rt, mz), adding entries named "features" to each dict in the data list of the MassSpecAnalyses instance.
        The "features" per target include:
            2D features:
            - peak_height: Height of the peak. 
            - peak_rt: Location/rt at which the peak occurs.
            - peak_mz: Location/mz at which the peak occurs.
            - peak_area_2d: Area under the peak within the region.
            - peak_volume: Volume of the peak region.
            - peak_fit_2d: Quality of to the 2d gaussian.
            - peak_a/n_error: Ratio between Volume and 2D Area as a measure of noise. 
            - peak_s/n_2d: 2D S/N ratio.
            - peak_mean: Mean/Center of the signal around the peak.
            1D features:
            - peak_fwhm: Full Width at Half Maximum of the peak along rt.
            - peak_s/n: Signal to Noise (S/N) ratio of the signal around the peak.
            - peak_fit_quality: The rsquared error of a gaussian fit on the signal.
    """

    def __init__(self, 
                 data: str = "sim", 
                 rt: float = None, 
                 mz: float = None, 
                 rt_window_size: int = 8, 
                 mz_window_size: float = 1.0, 
                 smooth: int|bool = None,
                 exclude: str = None): # regex for this. 
        super().__init__()
        self.data_type = "MassSpecAnalyses"
        self.method = "ExtractFeatures"
        self.algorithm = "Native"
        self.input_instance = dict
        self.output_instance = dict
        self.number_permitted = 1
        self.parameters = {
            "data" : data,
            "target_rt" : rt,
            "target_mz" : mz,
            "rt_window_size" : rt_window_size,
            "mz_window_size" : mz_window_size,
            "smooth" : smooth,
            "exclude" : exclude
            }
    
    def _apply_gaussian_filter(self, matrix, kernel_size, sigma, mode = 0): # scipy alternative is much faster.

        assert kernel_size % 2 == 1, "Kernel size must be odd."

        # generate a 2D gaussian kernel
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        
        if mode == 0: # isotropic Gaussian if the resolutions along rt and mz axes are comparable
            gauss = np.exp(-ax**2 / (2. * sigma**2))
            kernel = np.outer(gauss, gauss)
        else:
            sigma_rt = sigma + 0.5 # sim data has higher resolution along rt, so larger sigma
            sigma_mz = max(0.1, sigma - 0.5)

            gauss_rt = np.exp(-0.5 * (ax / sigma_rt) ** 2)
            gauss_mz = np.exp(-0.5 * (ax / sigma_mz) ** 2)

            kernel = np.outer(gauss_rt, gauss_mz)

        # normalize kernel
        kernel /= np.sum(kernel)

        result = convolve2d(matrix, kernel, mode='same', boundary='symm') # mode=same maintains the shape of the matrix, boundary=symm mirrors edge values within and without the kernel boundary
        
        return result 

    def _gaussian_2d(self, coords, A, mux, muy, sigx, sigy):
        x, y = coords
        return A * np.exp(
            -(((x - mux) ** 2) / (2 * sigx ** 2) + ((y - muy) ** 2) / (2 * sigy ** 2))
        )

    def _fit_gaussian(self, x, y, A_init, mu_init, sigma_init):
        # adam optimizer parameters
        alpha = 0.01
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        max_iterations = 300

        # initialize parameters
        A, mu, sigma = A_init, mu_init, sigma_init
        m_A = v_A = m_mu = v_mu = m_sigma = v_sigma = 0

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        for i in range(1, max_iterations + 1):
            grad_A = grad_mu = grad_sigma = 0.0

            # compute gradients
            exp_term = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            y_pred = A * exp_term
            error = y - y_pred

            grad_A = -2 * np.sum(error * exp_term)
            grad_mu = -2 * np.sum(error * A * exp_term * (x - mu) / (sigma ** 2))
            grad_sigma = -2 * np.sum(error * A * exp_term * ((x - mu) ** 2) / (sigma ** 3))

            # update A
            m_A = beta1 * m_A + (1 - beta1) * grad_A
            v_A = beta2 * v_A + (1 - beta2) * grad_A ** 2
            m_A_hat = m_A / (1 - beta1 ** i)
            v_A_hat = v_A / (1 - beta2 ** i)
            A -= alpha * m_A_hat / (np.sqrt(v_A_hat) + epsilon)

            # update mu
            m_mu = beta1 * m_mu + (1 - beta1) * grad_mu
            v_mu = beta2 * v_mu + (1 - beta2) * grad_mu ** 2
            m_mu_hat = m_mu / (1 - beta1 ** i)
            v_mu_hat = v_mu / (1 - beta2 ** i)
            mu -= alpha * m_mu_hat / (np.sqrt(v_mu_hat) + epsilon)

            # update sigma
            m_sigma = beta1 * m_sigma + (1 - beta1) * grad_sigma
            v_sigma = beta2 * v_sigma + (1 - beta2) * grad_sigma ** 2
            m_sigma_hat = m_sigma / (1 - beta1 ** i)
            v_sigma_hat = v_sigma / (1 - beta2 ** i)
            sigma -= alpha * m_sigma_hat / (np.sqrt(v_sigma_hat) + epsilon)

            # limit sigma range
            sigma = np.clip(sigma, 2.0, 10.0)

        return A, mu, sigma
    
    def _get_r2(self, x, y, A, mu, sigma):
        # predicted values from the Gaussian model
        y_pred = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        # residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # r-squared
        r_squared = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)

        return r_squared

    def _show_tile_debug_plot(self, plotter, region_rt, region_mz, peak_tile, fit_surface=None, title="Peak Tile Debug"):
        """
        Shows the peak tile as a 3D surface, with optional Gaussian fit overlay.
        
        Args:
            region_rt (1D np.array): Retention time values.
            region_mz (1D np.array): m/z values.
            peak_tile (2D np.array): Intensity matrix.
            fit_surface (2D np.array): Optional Gaussian fit surface with same shape as peak_tile.
            title (str): Plot title.
        """

        # Create meshgrid for RT and m/z axes
        MZ_grid, RT_grid = np.meshgrid(region_mz, region_rt)  # note: x = mz, y = rt

        fig = plotter.Figure()

        # Add peak tile as a 3D surface
        fig.add_trace(
            plotter.Surface(
                z=peak_tile,
                x=MZ_grid,
                y=RT_grid,
                colorscale="Viridis",
                colorbar=dict(title="Intensity"),
                name="Raw Tile",
                showscale=True,
                opacity=1.0,
                hovertemplate="RT: %{y:.3f}<br>m/z: %{x:.3f}<br>Intensity: %{z:.2f}<extra></extra>",
            )
        )

        # Add fit surface if available
        if fit_surface is not None:
            fig.add_trace(
                plotter.Surface(
                    z=fit_surface,
                    x=MZ_grid,
                    y=RT_grid,
                    surfacecolor=np.zeros_like(fit_surface),
                    colorscale=[[0, 'red'], [1, 'red']],
                    opacity=1.0,
                    name="Gaussian Fit",
                    showscale=False,
                    hovertemplate="RT: %{y:.3f}<br>m/z: %{x:.3f}<br>Fit Intensity: %{z:.2f}<extra></extra>",
                        contours={
                            "z": {
                                "show": True,
                                "usecolormap": False,
                                "color": "red",
                                "project_z": True,
                                }
                        }
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="m/z",
                yaxis_title="RT (min)",
                zaxis_title="Intensity",
            ),
            autosize=True,
            height=650,
        )

        return fig

    def run(self, analyses: MassSpecAnalyses) -> MassSpecAnalyses:
        """
        Extracts features from MS data.
        Args:
            analyses (MassSpecAnalyses): The MassSpecAnalyses instance to process.
        Returns:
            MassSpecAnalyses: The processed MassSpecAnalyses instance with features extracted.
        """

        data = analyses.data
        if len(data) == 0:
            print("No data to process.")
            return analyses
        
        plotter = analyses.plotter

        to_remove = None

        features_template = {
            "peak_height" : None,  
            "peak_rt" : None,
            "peak_mz" : None,
            "peak_area_2d" : None,
            "peak_volume" : None,
            "peak_fit_2d" : None,
            "peak_v/a_error" : None,
            "peak_s/n_2d" : None,
            "peak_mean" : None,
            "peak_fwhm_rt" : None,
            "peak_s/n" : None,
            "peak_fit_quality_rt" : None
        }

        if self.parameters["exclude"] is not None: 
            if isinstance(self.parameters["exclude"], list):
                if all(isinstance(item, str) for item in self.parameters["exclude"]):
                    to_remove = self.parameters["exclude"]
                else:
                    raise TypeError("Invalid inputs. Input only strings like 'Flush' that must be removed")
            elif isinstance(self.parameters["exclude"], str):
                to_remove = [self.parameters["exclude"]]

        entry = self.parameters["data"].lower()
        
        target_rt = self.parameters["target_rt"]
        target_mz = self.parameters["target_mz"]

        rt_window_size = self.parameters["rt_window_size"] 
        mz_window_size = self.parameters["mz_window_size"]

        for i, msd in enumerate(data): # by default for sim data.  
            feat = features_template.copy()

            chroma = msd[entry] # choose sim or tic

            if chroma is None: # if tic selected but no data available
                msd["features"] = feat
                data[i] = msd
                print(f"WARNING: This sample index {i} {msd["name"]} does not have TIC data. Skipping this iteration...")
                continue
            if to_remove:
                """
                EXTEND FOR OTHER STRINGS AS NEEDED
                """ 
                for string in to_remove:
                    pattern = re.compile(f"^{re.escape(string)}$", flags=re.IGNORECASE)
                if pattern.match(msd["sample"]):
                    print(f"{msd["sample"]} found at index {i}, Skipping extraction...")
                    msd["features"] = feat
                    continue

            rt = chroma["rt"]
            mz = chroma["mz"]
            intensity = chroma["intensity"]

            if self.parameters["smooth"]:
                if isinstance(self.parameters["smooth"], int) and self.parameters["smooth"] in [3, 5, 7]: # small choice of kernel_size to avoid over-smoothing
                    kernel_size = self.parameters["smooth"]
                else:
                    kernel_size = 3 # bind it to a 3x3 kernel 
                
                if entry == "sim": # choose anisotropic kernel creation when rts >> mzs (466, 19)
                    mode = 1
                else:
                    mode = 0
                intensity = self._apply_gaussian_filter(matrix=intensity, kernel_size=kernel_size, sigma=1.0, mode=mode) # higher sigma causes higher blur

            if target_rt is not None: 
                rt_index = np.argmin(np.abs(rt - target_rt)) # position of exact match or closest rt to given rt
            else:
                rt_index = 0
            
            if target_mz is not None:
                mz_index = np.argmin(np.abs(mz - target_mz))
            else:
                mz_index = 0

            # prep rt window
            rt_in_seconds = np.array(rt) * 60

            rt_lower_bound = rt_in_seconds[rt_index] - rt_window_size # adjust window by adding/subtracting rt_window seconds
            rt_start = max(0, np.argmin(np.abs(rt_in_seconds - rt_lower_bound))) # find position of true rt that is rt_window lower/higher than current target

            rt_upper_bound = rt_in_seconds[rt_index] + rt_window_size
            rt_end = min(len(rt), np.argmin(np.abs(rt_in_seconds - rt_upper_bound)))
            
            # prep mz window
            mz_mask = (mz >= mz[mz_index] - mz_window_size) & (mz <= mz[mz_index] + mz_window_size)
            if not np.any(mz_mask):
                mz_mask[mz_index] = True

            # when only whole number mz's ensure window is atleast 3 columns wide for 2D ops
            if np.sum(mz_mask) < 3:
                mz_start = max(0, mz_index - 1)
                mz_end = min(len(mz), mz_index + 2)
                mz_mask = np.zeros_like(mz, dtype=bool)
                mz_mask[mz_start:mz_end] = True
            
            rt_window = rt[rt_start : rt_end]
            mz_window = mz[mz_mask]

            intensity_tile = intensity[rt_start : rt_end, mz_mask]
            # target_label = f"_rt[{rt_window[0]}:{rt_window[-1]}]_mz[{mz_window[0]}:{mz_window[-1]}]" # save this for when the features should be named according to the target picked

            ## Peak Height
            peak = np.max(intensity_tile) # amplitude/height of peak
            feat[f"peak_height"] = peak # peak amplitude
            peak_position = np.argmax(intensity_tile) # index position of peak in 2D array
            peak_rt, peak_mz = np.unravel_index(peak_position, intensity_tile.shape)

            # if peak is at the edge of the tile, fitting will be poor
            if peak_rt < 1 or peak_rt > len(rt_window) - 2 or peak_mz < 1 or peak_mz > len(mz_window) - 2:
                print(f"Sample index {i} Warning: Peak near window edge at rt:{rt[rt_start + peak_rt]}, mz:{mz[mz_mask[0] + peak_mz]}") # debug this, mz value is still off in diag prints
            
            ## RT, MZ position at Peak
            feat[f"peak_rt"] = rt_window[peak_rt]
            peak_mz_value = mz_window[peak_mz]
            feat[f"peak_mz"] = peak_mz_value
            closest_mz_idx = np.argmin(np.abs(mz - peak_mz_value))

            ## 2D area under Peak. Will show anomalous value if peak is noisy
            peak_area_mz = np.trapz(intensity_tile, x=mz_window, axis=1)
            peak_area = np.trapz(peak_area_mz, x=rt_window)
            feat[f"peak_area_2d"] = peak_area

            rt_grid, mz_grid = np.meshgrid(rt_window, mz_window, indexing = 'ij')
            coords = (rt_grid.ravel(), mz_grid.ravel()) # coordinate grids for peak fitting
            tile = intensity_tile.ravel()
            assert intensity_tile.shape == rt_grid.shape == mz_grid.shape, "Shape mismatch between tile and coordinate grid"

            # initial conditions for 2D peak fitting
            A0 = peak
            mux0 = rt_window[np.argmax(np.sum(intensity_tile, axis=1))] # where it's brightest in RT
            muy0 = mz_window[np.argmax(np.sum(intensity_tile, axis=0))] # brightest in m/z
            #sigx0 = (rt_window[-1] - rt_window[0]) / 6 
            #sigy0 = (mz_window[-1] - mz_window[0]) / 6 # better fit than np.std()
            sigx0 = np.sqrt(np.average((rt_window - mux0)**2, weights=np.sum(intensity_tile, axis=1)))
            sigy0 = np.sqrt(np.average((mz_window - muy0)**2, weights=np.sum(intensity_tile, axis=0)))

            init_params = [A0, mux0, muy0, sigx0, sigy0]

            # constraints on params to avoid noisy fits
            bounds_lower = [0, rt_window[0], mz_window[0], 0.01, 0.01]  # A, mux, muy, sigx, sigy
            bounds_upper = [np.max(tile)*2, rt_window[-1], mz_window[-1], 
                            (rt_window[-1] - rt_window[0]), 
                            (mz_window[-1] - mz_window[0])]

            try:# try least_squares instead of curve_fit. More robust to noise and custom loss functions
                optimized_params, params_covariance = curve_fit(self._gaussian_2d, coords, tile, p0=init_params, bounds=(bounds_lower, bounds_upper))
                fitted = self._gaussian_2d(coords, *optimized_params).reshape(intensity_tile.shape)
                A, mux, muy, sigx, sigy = optimized_params
                #print(f"Fitted A={A:.2f}, mux={mux:.2f}, muy={muy:.2f}, sigx={sigx:.2f}, sigy={sigy:.2f}")

                good_fit_plot = self._show_tile_debug_plot(
                    plotter=plotter,
                    region_rt=rt_window, 
                    region_mz=mz_window, 
                    peak_tile=intensity_tile,
                    fit_surface=fitted, 
                    title=f"Index {i}, {msd["name"]}: Fit successful.")
                
                msd["fit_plot"] = good_fit_plot

                ## Peak volume. More robust estimate of peak ion count. Can be combined with area to identify S/N ratio in terms of effective peak area/volume
                volume = A * 2 * np.pi * sigx * sigy
                feat[f"peak_volume"] = volume

                ## Peak 2D gaussian fit
                residuals = intensity_tile - fitted
                r2_2d = np.sqrt(np.mean(residuals**2))
                feat[f"peak_fit_2d"] = r2_2d

                ## Area vs Noise error. A clean peak should show minimum deviation between the 2D area and peak volume values
                feat[f"peak_v/a_error"] = volume/peak_area

                ## S/N 2D
                noise_std_2d = np.std(residuals)
                if noise_std_2d == 0:
                    noise_std_2d = 1e-4
                signal_2d = A
                snr_2d = signal_2d / noise_std_2d
                feat[f"peak_s/n_2d"] = snr_2d  

            except RuntimeError as e:
                fallback_fit = self._gaussian_2d(coords, *init_params).reshape(intensity_tile.shape)
                print(f"[WARNING] Gaussian fit failed for analysis index {i} {msd["name"]} tile (rt[{rt_window[0]}:{rt_window[-1]}], mz[{mz_window[0]}:{mz_window[-1]}]: {e}.")
                print("\n")
                print("Returning debug_figure with initial fit estimates.")

                # plot failed fit for debugging
                debug_plot = self._show_tile_debug_plot(
                                        plotter=plotter,
                                        region_rt=rt_window, 
                                        region_mz=mz_window, 
                                        peak_tile=intensity_tile,
                                        fit_surface=fallback_fit, 
                                        title="Debug: Failed fit")
                debug_plot.show()
                input("Inspect tile. Press Enter to continue...")
                msd["fit_plot"] = debug_plot
                continue

            ## Peak FWHM along RT
            target_vector = intensity[rt_start : rt_end, closest_mz_idx]
            peak_pos_1d = np.argmax(target_vector)
            peak_half_max = peak/2 # autocast to float

            left_indices = np.where(target_vector[: peak_pos_1d] < peak_half_max)[0] # backward from peak
            left_cross = left_indices[-1] if len(left_indices) > 0 else 0

            right_indices = np.where(target_vector[peak_pos_1d :] < peak_half_max)[0] # search forward from peak
            right_cross = peak_pos_1d + (right_indices[0] if len(right_indices) > 0 else len(target_vector) - peak_pos_1d - 1)

            fwhm_rt = rt[right_cross] - rt[left_cross] # full width at half max by rt entries           
            feat[f"peak_fwhm_rt"] = fwhm_rt

            ## 1D gaussian fitting error
            sigma_1d = fwhm_rt/2.355
            A, mu, sigma = self._fit_gaussian(rt_window, target_vector, A_init=peak, mu_init=rt_window[peak_rt], sigma_init=sigma_1d)
            r2 = self._get_r2(rt_window, target_vector, A, mu, sigma)
            feat[f"peak_fit_quality_rt"] = r2

            ## S/N 
            noise_region = np.delete(target_vector, peak_pos_1d) # remove the peak
            noise_std = np.std(noise_region) if np.std(noise_region) != 0 else 1e-4
            snr = peak/noise_std
            feat[f"peak_s/n"] = snr

            ## Mean
            feat[f"peak_mean"] = np.mean(intensity_tile) # average around peak     

            msd["features"] = feat

            data[i] = msd
        
        analyses.data = data
        return analyses