"""
This module contains analyses child classes for a device.
"""

import os
import datetime
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.StreamPort.core import Analyses
from src.StreamPort.utils import get_file_encoding


def _read_pressure_curve_angi(fl: str, pc_template: dict) -> dict:
    """
    Read pressure curve data from a file.

    Args:
        file (str): Path to the pressure curve data file.

    Returns:
        dict: Dictionary containing the pressure curve data.
    """
    datetime_format = "%H:%M:%S %m/%d/%y"
    datetime_pattern = re.compile(r"(\d{2}:\d{2}:\d{2} \d{1,2}/\d{1,2}/\d{2})")

    fl_files = os.listdir(fl)
    fl_files = [os.path.join(fl, f) for f in fl_files]

    pressure_curve = [f for f in fl_files if "Pressure" in os.path.basename(f)]

    if len(pressure_curve) == 0:
        raise ValueError(f"No pressure curve files found in directory: {fl}")

    sample_xml = [f for f in fl_files if os.path.basename(f) == "SAMPLE.XML"]

    if len(sample_xml) == 0:
        raise ValueError(f"No SAMPLE.XML file found in directory: {fl}")

    log_file = [f for f in fl_files if os.path.basename(f) == "RUN.LOG"]

    if len(log_file) == 0:
        raise ValueError(f"No RUN.LOG file found in directory: {fl}")

    pressure_curve = pressure_curve[0]
    sample_xml = sample_xml[0]
    log_file = log_file[0]

    pc_fl = pc_template.copy()
    pc_fl["name"] = os.path.splitext(os.path.basename(fl))[0]
    pc_fl["path"] = fl
    pc_fl["batch"] = os.path.basename(os.path.dirname(fl))

    def align_data(file_path):
        decimal = "."
        with open(file_path, "r", encoding=get_file_encoding(file_path)) as file:
            first_line = file.readline()
            if "," in first_line:
                decimal = ","

        return decimal

    decimal = align_data(pressure_curve)

    df = pd.read_csv(
        pressure_curve,
        sep=";",
        decimal=decimal,
        header=None,
        names=["time", "pressure"],
    )

    for col in df.select_dtypes(include=["object"]):
        # Iterate over object (string) columns
        # Check if the column looks like it contains numeric data
        # (i.e., it has commas that need to be replaced)
        if df[col].str.contains(",").any():
            df[col] = df[col].str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # curve_runtime = (
    #     int(pressure_file["Time"].max())
    #     - int(pressure_file["Time"].min())
    # ) * 60  # convert to seconds

    # df = pd.read_csv(pressure_curve, header=None, sep=";")
    if df.shape[1] < 2:
        raise ValueError(f"File {pressure_curve} does not have at least two columns.")
    pc_fl["time_var"] = df.iloc[:, 0].to_numpy()
    pc_fl["pressure_var"] = df.iloc[:, 1].to_numpy()

    tree = ET.parse(sample_xml)
    root = tree.getroot()

    sample_name = root.find(".//Name")
    if sample_name is not None:
        pc_fl["sample"] = sample_name.text
    else:
        raise ValueError("No sample name found in SAMPLE.XML file.")

    method_name = root.find(".//ACQMethodPath")
    if method_name is not None:
        pc_fl["method"] = os.path.basename(method_name.text)
    else:
        raise ValueError("No method name found in SAMPLE.XML file.")

    with open(log_file, encoding=get_file_encoding(log_file)) as f:
        for line in f:
            if "Method started:" in line and pc_fl["idle_time"] is None:
                timestamp = datetime_pattern.search(line).group()
                timestamp = datetime.datetime.strptime(timestamp, datetime_format)
                pc_fl["timestamp"] = timestamp

            if "Run" in line and pc_fl["start_time"] is None:
                start_time = datetime_pattern.search(line).group()
                start_time = datetime.datetime.strptime(start_time, datetime_format)
                pc_fl["start_time"] = start_time

                match_key = re.match(r"(\S+)", line)
                if match_key:
                    pc_fl["detector"] = match_key.group(1)

            if "Postrun" in line and pc_fl["end_time"] is None:
                end_time = datetime_pattern.search(line).group()
                end_time = datetime.datetime.strptime(end_time, datetime_format)
                pc_fl["end_time"] = end_time

                match_key = re.match(r"(\S+)", line)
                if match_key:
                    pc_fl["pump"] = match_key.group(1)

    pc_fl["runtime"] = (pc_fl["end_time"] - pc_fl["start_time"]).total_seconds()
    return pc_fl


class PressureCurves(Analyses):
    """
    Class for analyzing pressure curves.

    Args:
        files (list): List of paths to pressure curve data files. Possible formats are .D.

    Attributes:
        data (list): List of dictionaries containing pressure curve data. Each dictionary contains the following
            keys:
            - index: Index of the pressure curve.
            - name: Name of the pressure curve.
            - path: Path to the pressure curve data file.
            - batch: Batch name.
            - batch_position: Position of the batch in the analysis.
            - idle_time: Idle time of the pressure curve.
            - sample: Sample name.
            - method: Method name.
            - timestamp: Timestamp of the pressure curve.
            - detector: Detector name.
            - pump: Pump name.
            - start_time: Start time of the pressure curve.
            - end_time: End time of the pressure curve.
            - runtime: Runtime of the pressure curve in seconds.
            - time_var: Time variable of the pressure curve.
            - pressure_var: Pressure variable of the pressure curve.

    Methods:
        plot_pressure_curves: Plots the pressure curves for given indices.
        plot_batches: Plots the batches of the pressure curves over the timestamps.
        plot_methods: Plots the methods of the pressure curves over the timestamps.
        plot_features_raw: Plots calculated raw data from features of the pressure curves over the time variable.
        plot_features: Plots calculated features of the pressure curves.
    """

    def __init__(self, files: list = None):
        super().__init__(data_type="PressureCurves", formats=[".D"])

        self.data = []

        pc_template = {
            "index": None,
            "name": None,
            "path": None,
            "batch": None,
            "batch_position": None,
            "idle_time": None,
            "sample": None,
            "method": None,
            "timestamp": None,
            "detector": None,
            "pump": None,
            "start_time": None,
            "end_time": None,
            "runtime": None,
            "time_var": None,
            "pressure_var": None,
        }

        if files is None:
            return

        if len(files) == 0:
            raise ValueError("No data provided for PressureCurves analysis.")

        if not isinstance(files, list):
            if isinstance(files, str):
                files = [files]
            else:
                raise TypeError("Files should be a list of file paths.")

        for fl in files:
            if not os.path.exists(fl):
                raise FileNotFoundError(f"File not found: {fl}")

            fl_ext = os.path.splitext(fl)[1]

            if fl_ext not in self.formats:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            if fl_ext == ".D":
                pc_fl = _read_pressure_curve_angi(fl, pc_template)
            else:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            self.data.append(pc_fl)
            self.data.sort(
                key=lambda x: (
                    x["timestamp"] if x["timestamp"] else datetime.datetime.min
                )
            )

            for i, item in enumerate(self.data):
                item["index"] = i
                self.data[i] = item

    def __str__(self):
        """
        Return a string representation of the PressureCurves object.
        """
        str_data = ""
        if len(self.data) > 0:
            for i, item in enumerate(self.data):
                if isinstance(item, dict):
                    str_data += f"    {i + 1}. {item["name"]} ({item["path"]})\n"
        else:
            str_data += "  No pressure curves data available."

        return (
            f"\n{type(self).__name__} \n"
            f"  data: {len(self.data)} \n"
            f"{str_data} \n"
        )

    def get_methods(self):
        """
        Get the methods of the pressure curves.

        Returns:
            list: List of unique methods used in the pressure curves.
        """
        methods = [item["method"] for item in self.data if "method" in item]
        return list(set(methods))

    def get_batches(self):
        """
        Get the batches of the pressure curves.

        Returns:
            list: List of unique batches used in the pressure curves.
        """
        batches = [item["batch"] for item in self.data if "batch" in item]
        return list(set(batches))

    def plot_pressure_curves(self, indices: list = None):
        """
        Plot the pressure curves for given indices.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        else:
            if not isinstance(indices, list):
                raise TypeError("Indices should be a list of integers.")
            if len(indices) == 0:
                raise ValueError("No indices provided for plotting.")

        fig = go.Figure()
        for i in indices:
            pc = self.data[i]
            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc["pressure_var"],
                    mode="lines",
                    name=f"{pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Pressure (bar)",
            template="simple_white",
        )

        return fig

    def plot_batches(self):
        """
        Plot the batches of the pressure curves over the timestamps.
        """

        timestamp_var = np.array([item["start_time"] for item in self.data])
        batch_position_var = np.array([item["batch_position"] for item in self.data])
        batch_name = [item.get("batch", "") for item in self.data]
        unique_batches = list(set(batch_name))

        fig = go.Figure()
        for batch in unique_batches:
            indices = [i for i, b in enumerate(batch_name) if b == batch]

            str_text = []
            for i in indices:
                str_text.append("")
                str_text[i] += f"{self.data[i]['index']}. ({self.data[i]['name']})<br>"
                str_text[i] += f"Batch: {batch}<br>"

            fig.add_trace(
                go.Scatter(
                    x=timestamp_var[indices],
                    y=batch_position_var[indices],
                    mode="markers",
                    name=batch,
                    text=str_text,
                    hovertemplate="%{text}<br>Timestamp: %{x}<br>Batch Position: %{y}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Batch Position",
            template="simple_white",
            legend_title="Batches",
        )

        return fig

    def plot_methods(self):
        """
        Plot the methods of the pressure curves over the timestamps.
        """

        timestamp_var = np.array([item["timestamp"] for item in self.data])
        batch_position_var = np.array([item["batch_position"] for item in self.data])
        method_var = [item.get("method", "") for item in self.data]

        unique_methods = list(set(method_var))

        fig = go.Figure()
        for method in unique_methods:
            indices = [i for i, m in enumerate(method_var) if m == method]

            str_text = []
            for i in indices:
                str_text.append("")
                str_text[i] += f"{self.data[i]['index']}. ({self.data[i]['name']})<br>"
                str_text[i] += f"Method: {method}<br>"

            fig.add_trace(
                go.Scatter(
                    x=timestamp_var[indices],
                    y=batch_position_var[indices],
                    mode="markers",
                    name=method,
                    text=str_text,
                    hovertemplate="%{text}<br>Timestamp: %{x}<br>Batch Position: %{y}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Batch Position",
            template="simple_white",
            legend_title="Methods",
        )

        return fig

    def plot_features_raw(self, indices: list = None):
        """
        Plot calculated raw data from features of the pressure curves over the time variable.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        else:
            if not isinstance(indices, list):
                raise TypeError("Indices should be a list of integers.")
            if len(indices) == 0:
                raise ValueError("No indices provided for plotting.")

        fig = go.Figure()
        for i in indices:
            pc = self.data[i]
            pc_feat = pc["features_raw"]
            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["trend"],
                    mode="lines",
                    name=f"trend {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["seasonal"],
                    mode="lines",
                    name=f"seasonal {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["residual"],
                    mode="lines",
                    name=f"residual {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["seasonal_fft"],
                    mode="lines",
                    name=f"ft_seasonal {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["residual_fft"],
                    mode="lines",
                    name=f"ft_residual {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["freq_bins"],
                    mode="lines",
                    name=f"freq_bins {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc_feat["freq_bins_indices"],
                    mode="lines",
                    name=f"freq_bins_indices {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="U.A.",
            template="simple_white",
        )

        return fig

    def plot_features(self, indices: list = None):
        """
        Plot calculated features of the pressure curves.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        else:
            if not isinstance(indices, list):
                raise TypeError("Indices should be a list of integers.")
            if len(indices) == 0:
                raise ValueError("No indices provided for plotting.")

        fig = go.Figure()
        for i in indices:
            pc = self.data[i]
            pc_feat = pc["features"]

            fig.add_trace(
                go.Scatter(
                    x=list(pc_feat.keys()),
                    y=list(pc_feat.values()),
                    mode="markers+lines",
                    name=f"{pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="U.A.",
            template="simple_white",
        )

        return fig
