"""
This module contains analyses child classes for a device.
"""

import os
import datetime
import re
import xml.etree.ElementTree as ET
import pandas as pd
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
        if df[col].str.contains(",").any():
            df[col] = df[col].str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
            if "Method started:" in line and pc_fl["timestamp"] is None:
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

    pc_fl["runtime"] = (max(pc_fl["time_var"]) - min(pc_fl["time_var"])) * 60
    return pc_fl


class PressureCurvesAnalyses(Analyses):
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
        super().__init__(data_type="PressureCurvesAnalyses", formats=[".D"])

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
            raise ValueError("No data provided for PressureCurvesAnalyses analysis.")

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

        self.data = sorted(self.data, key=lambda x: x["start_time"])

        for i, pc in enumerate(self.data):
            pc["index"] = i

            if i == 0:
                pc["idle_time"] = 0
                pc["batch_position"] = 1
                continue

            pc["idle_time"] = (
                pc["start_time"] - self.data[i - 1]["end_time"]
            ).total_seconds()

            if (
                pc["method"] == self.data[i - 1]["method"]
                and pc["batch"] == self.data[i - 1]["batch"]
            ):
                pc["batch_position"] = self.data[i - 1]["batch_position"] + 1
            else:
                pc["batch_position"] = 1

            self.data[i] = pc

    def __str__(self):
        """
        Return a string representation of the PressureCurvesAnalyses object.
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

    def get_methods(self) -> list[str]:
        """
        Get the methods of the pressure curves.

        Returns:
            list: List of unique methods used in the pressure curves.
        """
        methods = [item["method"] for item in self.data if "method" in item]
        return list(set(methods))

    def get_batches(self) -> list[str]:
        """
        Get the batches of the pressure curves.

        Returns:
            list: List of unique batches used in the pressure curves.
        """
        batches = [item["batch"] for item in self.data if "batch" in item]
        return list(set(batches))

    def get_method_indices(self, method: str) -> list[int]:
        """
        Get the indices of the pressure curves for a given method.

        Args:
            method (str): Method name to filter the pressure curves.

        Returns:
            list: List of indices of the pressure curves for the given method.
        """
        if not isinstance(method, str):
            raise TypeError("Method should be a string.")

        indices = [i for i, item in enumerate(self.data) if item["method"] == method]
        return indices

    def get_batch_indices(self, batch: str) -> list[int]:
        """
        Get the indices of the pressure curves for a given batch.

        Args:
            batch (str): Batch name to filter the pressure curves.

        Returns:
            list: List of indices of the pressure curves for the given batch.
        """
        if not isinstance(batch, str):
            raise TypeError("Batch should be a string.")

        indices = [i for i, item in enumerate(self.data) if item["batch"] == batch]
        return indices

    def get_metadata(self, indices: list = None) -> pd.DataFrame:
        """
        Get a DataFrame of the metadata of the pressure curves.

        Args:
            indices (list): List of indices of the pressure curves to include in the DataFrame. If None, all curves are included.

        Returns:
            pd.DataFrame: DataFrame containing the metadata of the pressure curves.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = [indices]
        if not isinstance(indices, list):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for DataFrame creation.")

        metadata = []
        for i in indices:
            pc = self.data[i].copy()
            for key in ["time_var", "pressure_var", "features", "features_raw"]:
                pc.pop(key, None)
            metadata.append(pc)

        return pd.DataFrame(metadata)

    def get_features(self, indices: list = None) -> pd.DataFrame:
        """
        Get a DataFrame of the features of the pressure curves.

        Args:
            indices (list): List of indices of the pressure curves to include in the DataFrame. If None, all curves are included.

        Returns:
            pd.DataFrame: DataFrame containing the features of the pressure curves.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = [indices]
        if not isinstance(indices, list):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for DataFrame creation.")

        features = []
        for i in indices:
            pc = self.data[i]
            features.append(pc["features"])

        return pd.DataFrame(features)

    def plot_pressure_curves(self, indices: list = None) -> go.Figure:
        """
        Plot the pressure curves for given indices.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = list(indices)
        elif isinstance(indices, tuple):
            indices = list(indices)
        if not isinstance(indices, list):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for DataFrame creation.")

        fig = go.Figure()
        for i in indices:
            pc = self.data[i]
            mt = self.get_metadata([i]).to_dict(orient="records")[0]
            text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in mt.items())
            fig.add_trace(
                go.Scatter(
                    x=pc["time_var"],
                    y=pc["pressure_var"],
                    mode="lines",
                    name=f"{pc['name']} ({pc['sample']})",
                    text=text,
                    hovertemplate=f"{text}<br><b>Time: </b>%{{x}}<br><b>Pressure: </b>%{{y}}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Pressure (bar)",
            template="simple_white",
        )

        return fig

    def plot_batches(self, indices: list = None) -> go.Figure:
        """
        Plot the batches of the pressure curves over the timestamps.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        df = self.get_metadata(indices)
        df = df.sort_values("timestamp")
        batches_in_order = df["batch"].drop_duplicates().tolist()
        fig = go.Figure()
        for batch in batches_in_order:
            mask = df["batch"] == batch
            df_batch = df[mask]
            text = [
                "<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items())
                for row in df_batch.to_dict(orient="records")
            ]
            fig.add_trace(
                go.Scatter(
                    x=df_batch["timestamp"],
                    y=df_batch["batch_position"],
                    mode="markers",
                    name=batch,
                    legendgroup=batch,
                    text=text,
                    hovertemplate=text,
                )
            )
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Batch Position",
            template="simple_white",
            legend_title="Batches",
        )

        return fig

    def plot_methods(self, indices: list = None) -> go.Figure:
        """
        Plot the methods of the pressure curves over the timestamps.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        df = self.get_metadata(indices)
        df = df.sort_values("timestamp")
        methods_in_order = df["method"].drop_duplicates().tolist()

        fig = go.Figure()
        for method in methods_in_order:
            mask = df["method"] == method
            df_method = df[mask]
            text = [
                "<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items())
                for row in df_method.to_dict(orient="records")
            ]
            fig.add_trace(
                go.Scatter(
                    x=df_method["timestamp"],
                    y=df_method["batch_position"],
                    mode="markers",
                    name=method,
                    legendgroup=method,
                    text=text,
                    hovertemplate=text,
                )
            )

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Batch Position",
            template="simple_white",
            legend_title="Methods",
        )

        return fig

    def plot_features_raw(self, indices: list = None) -> go.Figure:
        """
        Plot calculated raw data from features of the pressure curves over the time variable.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """

        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = [indices]
        if not isinstance(indices, list):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for DataFrame creation.")

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

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["pressure_baseline_corrected"],
            #         mode="lines",
            #         name=f"pressure_baseline_corrected {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["seasonal_fft"],
            #         mode="lines",
            #         name=f"ft_seasonal {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["residual_fft"],
            #         mode="lines",
            #         name=f"ft_residual {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["freq_bins"],
            #         mode="lines",
            #         name=f"freq_bins {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["freq_bins_indices"],
            #         mode="lines",
            #         name=f"freq_bins_indices {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="U.A.",
            template="simple_white",
        )

        return fig

    def plot_features(self, indices: list = None, normalize: bool = True) -> go.Figure:
        """
        Plot calculated features of the pressure curves.

        Args:
            indices (list): List of indices of the pressure curves to plot. If None, all curves are plotted.
        """
        mt = self.get_metadata(indices).to_dict(orient="records")

        ft = self.get_features(indices)
        if normalize:
            # normalize each column of the DataFrame
            for col in ft.columns:
                if ft[col].dtype in [int, float]:
                    ft[col] = ft[col] / ft[col].max()
                    #ft[col] = (ft[col] - ft[col].min()) / (ft[col].max() - ft[col].min())

        ft = ft.to_dict(orient="records")

        text = ["<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items()) for row in mt]

        fig = go.Figure()
        for i, fti in enumerate(ft):
            fti = ft[i]
            mti = mt[i]

            fig.add_trace(
                go.Scatter(
                    x=list(fti.keys()),
                    y=list(fti.values()),
                    mode="markers+lines",
                    name=f"{mti['name']} ({mti['sample']})",
                    text=text[i],
                    hovertemplate=(
                        f"<b>{mti['name']} ({mti['sample']})</b><br>"
                        + "%{{x}}<br>"
                        + "%{{y}}<extra></extra>"
                        if text is None
                        else f"{text[i]}<br><b>x: </b>%{{x}}<br><b>y: </b>%{{y}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="U.A.",
            template="simple_white",
        )

        return fig
