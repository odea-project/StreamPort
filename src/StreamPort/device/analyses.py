"""
This module contains analyses child classes for a device.
"""

import os
import datetime
import re
import xml.etree.ElementTree as ET
import rainbow as rb
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
    
    #handle error_lc
    if "D2F" in pc_fl["name"]:
        pc_fl["method"] = "error_lc"
    #/handle error_lc done

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

            filename = os.path.basename(fl)
            batch = os.path.basename(os.path.dirname(fl))
            run = os.path.join(batch, filename)

            fl_ext = os.path.splitext(fl)[1]

            if fl_ext not in self.formats:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            if fl_ext == ".D":
                try:
                    pc_fl = _read_pressure_curve_angi(fl, pc_template)
                except ValueError:
                    print("No data for this run: ", run)#error_lc. Corrupted .D files cannot be read with SignalExtraction
                    continue
            else:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            self.data.append(pc_fl)

        self.data = sorted(self.data, key=lambda x: x["start_time"])

        #remove StandBy samples in case of error-lc data
        self.data = [pc for pc in self.data if pc["sample"].lower() != "standby"]

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
                pc["method"] == self.data[i - 1]["method"]# error_lc batch position assignment broke here. Fixed.
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
            indices = [indices]
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
            xaxis_title="Time (min)",
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
        max_trend = None
        for i in indices:
            pc = self.data[i]
            pc_feat = pc["features_raw"]
            x=pc["time_var"]

            if max_trend is None:
                max_trend = max(pc_feat["trend"])
            elif max_trend < max(pc_feat["trend"]):
                max_trend = max(pc_feat["trend"])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pc_feat["trend"],
                    mode="lines",
                    name=f"trend {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["seasonal"],
            #         mode="lines",
            #         name=f"seasonal {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            # fig.add_trace(
            #     go.Scatter(
            #         x=pc["time_var"],
            #         y=pc_feat["residual"],
            #         mode="lines",
            #         name=f"residual {pc['name']} ({pc['sample']})",
            #         text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
            #     )
            # )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pc_feat["pressure_baseline_corrected"],
                    mode="lines",
                    name=f"pressure_baseline_corrected {pc['name']} ({pc['sample']})",
                    text=f"{pc['index']}. {pc['name']} ({pc['sample']})<br>Batch: {pc['batch']}<br>Method: {pc['method']}",
                )
            )

        fig.update_layout(
            xaxis_title="Time (min)",
            yaxis_title="U.A.",
            template="simple_white",
        )

        for i in range(len(pc_feat["bin_edges"])):
            x0 = (pc_feat["bin_edges"][i][0]).round(3)
            x1 = (pc_feat["bin_edges"][i][1]).round(3) 
            y1 = max_trend
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=0, y1=y1,  
                fillcolor="rgba(200,200,255,0.4)", 
                line=dict(width=1),
                layer="below"
            )

            fig.add_annotation(
                x=(x0 + x1) / 2,  
                y=y1 + 2.0, 
                text =f"<br>Bin {i+1}<br>",      
                hovertext=f"<br>Entries between time: {x0} and {x1}",
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center",  
                opacity=0.8
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
            # normalize each column of the DataFrame while handling NaN values
            ft = ft.fillna(0)  # Replace NaN with 0 
            for col in ft.columns:
                if ft[col].dtype in [int, float] and ft[col].max() != 0:
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


def _read_ms_data_rainbow(fl: str, ms_template: dict) -> dict:
    """
    Reads MS data from a file using the Rainbow API

    Args:
        fl (str): The .D file to read data from
        ms_template (dict): The template containing ordered MS data entries 

    Returns:
        ms_data (dict): The extracted and grouped MS data for an individual run.
    """
    datetime_format = "%H:%M:%S %m/%d/%y"
    datetime_pattern = re.compile(r"(\d{2}:\d{2}:\d{2} \d{1,2}/\d{1,2}/\d{2})")
    ms_data = ms_template.copy()

    datadir = rb.read(fl) # create .D directory object

    metadata = datadir.metadata # run metadata including vendor
    ms_data["vial_position"] = metadata.get("vialpos")

    datafiles = datadir.datafiles
    ms_files = [file for file in datafiles if file.detector == "MS"] # compatible types UV, CH, MS

    ms1, ms2 = None, None
    for file in ms_files:
        if file.name == "MSD1.MS":
            ms1 = file
        if file.name == "MSD2.MS":
            ms2 = file
    
    if ms1 is None:
        print(f"No MS1 data found in directory: {fl}")
        ms_data["ms1"] = None
    else:
        ms_data["ms1"] = {
                            "rt" : ms1.xlabels, # 1-D retention time (minutes)
                            "mz" : ms1.ylabels, # 1-D mz/wavelength
                            "intensity" : ms1.data # 2-D intensity # all np.ndarrays
                        }

    if ms2 is None:
        print(f"No MS2 data found in directory: {fl}")
        ms_data["ms2"] = None 
    else:               
        ms_data["ms2"] = {
                            "rt" : ms2.xlabels, # 1-D retention time (minutes)
                            "mz" : ms2.ylabels, # 1-D mz/wavelength
                            "intensity" : ms2.data # 2-D intensity # all np.ndarrays
                        }
        
    if not(ms1 is None or ms2 is None) and ms1.metadata != ms2.metadata:
        raise ValueError("The metadata pairs for this run do not match.")
    file_metadata = ms1.metadata # MS file metadata including date and method
    
    # add date string, unit, e.g. as metadata. Timestamp will still be used for operations
    ms_data.update(file_metadata)

    files_list = os.listdir(fl) # read other files not read by rainbow
    files_list = [os.path.join(fl, f) for f in files_list]

    sample_xml = [file for file in files_list if "SAMPLE.XML" in file]
    sample_xml = sample_xml[0]

    if sample_xml is None:
        raise ValueError(f"No SAMPLE.XML file found in directory: {fl}")
    
    log_file = [file for file in files_list if "RUN.LOG" in file] # run logs for metadata
    log_file = log_file[0]

    if log_file is None:
       raise ValueError(f"No RUN.LOG file found in directory: {fl}")

    tree = ET.parse(sample_xml)
    root = tree.getroot()
    sample_name = root.find(".//Name")
    if sample_name is not None:
        sample = sample_name.text
    else:
        raise ValueError("No sample name found in SAMPLE.XML file.")    
    ms_data["sample"] = sample

    ms_data["name"] = os.path.splitext(os.path.basename(fl))[0]
    ms_data["path"] = fl
    ms_data["batch"] = os.path.basename(os.path.dirname(fl))
    ms_data["detector"] = ms1.detector

    #handle error_lc ### Not a permanent fix. Discuss and correct here and in pressure_curves!!!
    if "D2F" in ms_data["name"]:
        ms_data["method"] = "error_lc"
    #/handle error_lc done

    with open(log_file, encoding=get_file_encoding(log_file)) as f:
        for line in f:
            if "Method started:" in line and ms_data["timestamp"] is None:
                timestamp = datetime_pattern.search(line).group()
                timestamp = datetime.datetime.strptime(timestamp, datetime_format)
                ms_data["timestamp"] = timestamp

            if "Run" in line and ms_data["start_time"] is None:
                start_time = datetime_pattern.search(line).group()
                start_time = datetime.datetime.strptime(start_time, datetime_format)
                ms_data["start_time"] = start_time

            if "Postrun" in line and ms_data["end_time"] is None:
                end_time = datetime_pattern.search(line).group()
                end_time = datetime.datetime.strptime(end_time, datetime_format)
                ms_data["end_time"] = end_time

                match_key = re.match(r"(\S+)", line)
                if match_key:
                    ms_data["pump"] = match_key.group(1)

    ms_data["runtime"] = (max(ms1.xlabels) - min(ms1.xlabels)) * 60

    return ms_data


class MassSpecAnalyses(Analyses):
    """
    Class for analyzing Mass Spectrometry data.

    Args:
        files (list): List of paths to MS data files. Possible formats are .D.

    Attributes:
        data (list): List of dictionaries containing pressure curve data. Each dictionary contains the following
            keys:
            - index: Index of the run.
            - name: Name of the run.
            - path: Path to the ms data file.
            - batch: Batch name.
            - batch_position: Position of the batch in the analysis.
            - idle_time: Idle time of the (instrument before the) run.
            - sample: Sample name, whether the run was a Flush, Sample or Blank.
            - method: Method name.
            - timestamp: Timestamp of the run.
            - unit: Unit.
            - instrument: Type of instrument (device)
            - signal: Signal.
            - detector: Detector name.
            - pump: Pump name.
            - vial_position: Vial Position entry of the run.
            - start_time: Start time of the run.
            - end_time: End time of the run.
            - runtime: Runtime.
            - ms1: MS1 data dict containing the corresponding rt, mz and ansorbances.
            - ms2: MS2 data dict.

    Methods:
        plot_pressure_curves: Plots the pressure curves for given indices.
        plot_batches: Plots the batches of the pressure curves over the timestamps.
        plot_methods: Plots the methods of the pressure curves over the timestamps.
        plot_features_raw: Plots calculated raw data from features of the pressure curves over the time variable.
        plot_features: Plots calculated features of the pressure curves.
    """

    def __init__(self, files: list = None):
        super().__init__(data_type="MassSpecAnalyses", formats=[".D"])

        self.data = []

        ms_template = {
            "index": None,
            "name": None,
            "path": None,
            "batch": None,
            "batch_position": None,
            "idle_time": None,
            "sample": None,
            "method": None,
            "timestamp": None,
            "unit": None,
            "instrument": None,
            "signal": None,
            "detector": None,
            "pump": None,
            "vial_position": None, 
            "start_time": None,
            "end_time": None,
            "runtime": None,
            "ms1": None,
            "ms2": None
        }
        ms_data = ms_template.copy()

        if files is None:
            return

        if len(files) == 0:
            raise ValueError("No data provided for MassSpecAnalyses analysis.")

        if not isinstance(files, list):
            if isinstance(files, str):
                files = [files]
            else:
                raise TypeError("Files should be a list of file paths.")

        for fl in files:
            if not os.path.exists(fl):
                raise FileNotFoundError(f"File not found: {fl}")

            filename = os.path.basename(fl)
            batch = os.path.basename(os.path.dirname(fl))
            run = os.path.join(batch, filename)

            fl_ext = os.path.splitext(fl)[1]

            if fl_ext not in self.formats:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            if fl_ext == ".D":
                try:
                    ms_data = _read_ms_data_rainbow(fl, ms_template)
                except ValueError:
                    print("No data for this run: ", run) # Corrupted .D files cannot be read with SignalExtraction and will not yield data
                    continue
            else:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            self.data.append(ms_data)

        self.data = sorted(self.data, key=lambda x: x["start_time"])

        #remove StandBy samples in case of error-lc data
        self.data = [msd for msd in self.data if msd["sample"].lower() != "standby"]

        for i, msd in enumerate(self.data):
            msd["index"] = i

            if i == 0:
                msd["idle_time"] = 0
                msd["batch_position"] = 1
                continue

            msd["idle_time"] = (
                msd["start_time"] - self.data[i - 1]["end_time"]
            ).total_seconds()

            if (
                msd["method"] == self.data[i - 1]["method"]# error_lc batch position assignment broke here. Patched (?) by setting all error_lc methods to same value.
                and msd["batch"] == self.data[i - 1]["batch"]
            ):
                msd["batch_position"] = self.data[i - 1]["batch_position"] + 1
            else:
                msd["batch_position"] = 1

            self.data[i] = msd

    def __str__(self):
        """
        Return a string representation of the MassSpecAnalyses object.
        """
        str_data = ""
        if len(self.data) > 0:
            for i, item in enumerate(self.data):
                if isinstance(item, dict):
                    str_data += f"    {i + 1}. {item["name"]} ({item["path"]})\n"
        else:
            str_data += "  No MS data available."

        return (
            f"\n{type(self).__name__} \n"
            f"  data: {len(self.data)} \n"
            f"{str_data} \n"
        )

    def get_methods(self) -> list[str]:
        """
        Get the methods of the ms data.

        Returns:
            list: List of unique methods used in the data.
        """
        methods = [item["method"] for item in self.data if "method" in item]
        return list(set(methods))

    def get_batches(self) -> list[str]:
        """
        Get the batches of the ms data.

        Returns:
            list: List of unique batches used in the data.
        """
        batches = [item["batch"] for item in self.data if "batch" in item]
        return list(set(batches))

    def get_method_indices(self, method: str) -> list[int]:
        """
        Get the indices of the ms data for a given method.

        Args:
            method (str): Method name to filter the data.

        Returns:
            list: List of indices of the ms data for the given method.
        """
        if not isinstance(method, str):
            raise TypeError("Method should be a string.")

        indices = [i for i, item in enumerate(self.data) if item["method"] == method]
        return indices

    def get_batch_indices(self, batch: str) -> list[int]:
        """
        Get the indices of the ms_data for a given batch.

        Args:
            batch (str): Batch name to filter the data.

        Returns:
            list: List of indices of the ms data for the given batch.
        """
        if not isinstance(batch, str):
            raise TypeError("Batch should be a string.")

        indices = [i for i, item in enumerate(self.data) if item["batch"] == batch]
        return indices

    def get_metadata(self, indices: list = None) -> pd.DataFrame:
        """
        Get a DataFrame of the metadata of the MS data.

        Args:
            indices (list): List of indices of the data to include in the DataFrame. If None, all curves are included.

        Returns:
            pd.DataFrame: DataFrame containing the metadata of the ms chromatograms.
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
            for key in ["ms1", "ms2", "features", "features_raw"]:
                pc.pop(key, None)
            metadata.append(pc)

        return pd.DataFrame(metadata)
    
    def plot_data(self, indices: int = None, data: str = None, mz: int = None, dim: int = None) -> go.Figure:
        """
        Plot the chromatograms of the MS data. If mz label is not provided, the plot for the first mz entry is returned.

        Args:
            indices (int|list): Index identifiers to samples to plot. Defaults to None - all samples are plotted.
            data (str): Choice of whether "MS1" or "MS2" data should be retrieved. Defaults to "MS1".
            mz (int): In case of 2-D plot, fix the mz value for which the correspnding rt-intensity graph is created. Defaults to None - sets mz to mz[0]
            dim (int): Dimensions to plot. If 3, creates a 3D plot of the entire matrix and negates the "mz" argument. Defaults to None - 2D plot.  
        """
        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, tuple):
            indices = list(indices)
        if not isinstance(indices, list):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for DataFrame creation.")
        
        if data.lower() == "ms2":
            entry = "ms2"
        else:
            entry = "ms1"

        fig = go.Figure()
        for i in indices:            
            msd = self.data[i]
            if mz is not None:
                mz_index = msd[entry]["mz"].tolist().index(mz)
            else:
                mz_index = 0
            mt = self.get_metadata([i]).to_dict(orient="records")[0]
            text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in mt.items())
            
            if dim == 3:
                fig.add_trace(
                    go.Surface(
                        z=msd[entry]["intensity"], 
                        x=msd[entry]["rt"],
                        y=msd[entry]["mz"],
                        name=f"{msd['name']} ({msd['sample']})",
                        text=text,
                        hovertemplate=f"{text}<br><b>m/z: {str(mz)}<br><b>Time: </b>%{{x}}<br><b>Intensity: </b>%{{y}}<extra></extra>",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=msd[entry]["rt"],
                        y=msd[entry]["intensity"][:, mz_index],
                        mode="lines",
                        name=f"{msd['name']} ({msd['sample']})",
                        text=text,
                        hovertemplate=f"{text}<br><b>m/z: {str(mz)}<br><b>Time: </b>%{{x}}<br><b>Intensity: </b>%{{y}}<extra></extra>",
                    )
                )

        fig.update_layout(
            title=f"{entry.capitalize()} Traces",
            xaxis_title="Retention time (min)",
            yaxis_title="Intensity",
            template="simple_white",
        )

        return fig
    
    def plot_batches(self, indices: list = None) -> go.Figure:
        """
        Plot the batches of the MS data over the timestamps.

        Args:
            indices (list): List of indices of the batches to plot. If None, all batches are plotted.
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


def _read_actuals_angi(fl: str) -> dict:
    """
    Reads actuals from a file.
    """
    return_object = {
        "actuals" : None,
        "ac_types" : None
    }
    actuals = None
    ac_types = None

    actuals_file = pd.read_csv(
        fl,
        low_memory=False,
        sep=";",
        decimal="."
    )
    #INCOMPLETE
    actuals = actuals_file.dropna(axis=1, how="all")

    ac_type = actuals["ModuleId"][0]
    ac_keys = actuals.columns.tolist()
    
    ac_types = {ac_type : ac_keys}

    return_object["actuals"] = actuals
    return_object["ac_types"] = ac_types

    return return_object


class ActualsAnalyses(Analyses):
    """
    Class for analyzing device Actuals to be used in combination with pressure signal readings
    
    Args:
        files (list): List of paths to actuals data files. Possible formats are .csv.

    Attributes:
        data (list): List of dictionaries containing actuals data. Each dictionary contains the following
            keys:
            - index: Index of the actuals file.
            - path: Path to the actuals data file.
            - batch: Batch name assigned using corresponding pressure curve.
            - batch_position: Position of the batch in the analysis.
            - sample: Sample name assigned using corresponding pressure curve.
            - method: Method name using corresponding pressure curve.
            - timestamp: Timestamp of the actuals data used to identify and connect to pressure data.
            - start_time: Start time of the actuals file.
            - end_time: End time of the actuals file.
            - module_id: Name of the module/sensor (THM, DAD, ...).
            - temperature_var: Temperature signal reading.
            - valve_position: Valve position entry.
            - error_state: Entry on any discovered errors.

    Methods:
    """
    #INCOMPLETE
    def __init__(self, files: list = None, types: dict = None):
        super().__init__(data_type = "ActualsAnalyses", formats = [".csv"])

        self.data = []
        self.actual_types = {}

        if files is None:
            return

        if len(files) == 0:
            raise ValueError("No data provided for ActualsAnalyses analysis.")
        
        if not isinstance(files, list):
            if isinstance(files, str):
                files = [files]
            else:
                raise TypeError("Files should be a list of file paths.")        

        for fl in files:
            if not os.path.exists(fl):
                raise FileNotFoundError(f"File not found: {fl}")

            filename = os.path.basename(fl)
            batch = os.path.basename(os.path.dirname(fl))
            run = os.path.join(batch, filename)

            fl_ext = os.path.splitext(fl)[1]

            if fl_ext not in self.formats:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            if fl_ext == ".csv":
                try:
                    ac_fl = _read_actuals_angi(fl)
                except ValueError:
                    print("No data for this run: ", run)
                    continue
            else:
                raise ValueError(
                    f"Unsupported file format: {fl_ext}. Supported formats are: {self.formats}"
                )

            self.data.append(ac_fl["actuals"])
            self.actual_types.update(ac_fl["ac_types"])

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
            str_data += "  No actuals data available."

        return (
            f"\n{type(self).__name__} \n"
            f"  data: {len(self.data)} \n"
            f"{str_data} \n"
        )
    
