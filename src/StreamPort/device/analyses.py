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

    try:
        ms1 = datadir.get_file("MSD1.MS") # compatible types UV, CH, MS
        ms2 = datadir.get_file("MSD2.MS") # MSD1 corresponds to SIM data, MSD2 to TIC
    except Exception: # possible that MSD2 does not exist
        ms2 = None
        pass

    if ms1 is None:
        ms_data["sim"] = None
        raise FileNotFoundError(f"No MS1 data found in directory: {fl}")
        #print(f"No MS1 data found in directory: {fl}")
    else:
        ms_data["sim"] = {
                            "rt" : ms1.xlabels, # 1-D retention time (minutes)
                            "mz" : ms1.ylabels, # 1-D mz/wavelength
                            "intensity" : ms1.data # 2-D intensity # all np.ndarrays
                        }

    if ms2 is None:
        print(f"No MS2 data found in directory: {fl}")
        ms_data["tic"] = None 
    else:               
        ms_data["tic"] = {
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
        raise FileNotFoundError(f"No SAMPLE.XML file found in directory: {fl}")
    
    log_file = [file for file in files_list if "RUN.LOG" in file] # run logs for metadata
    log_file = log_file[0]

    if log_file is None:
       raise FileNotFoundError(f"No RUN.LOG file found in directory: {fl}")
    
    acq_file = [file for file in files_list if "acq.txt" in file] # acquisition files for compound details
    acq_file = acq_file[0]

    if acq_file is None:
        raise FileNotFoundError(f"No acq.txt file found in directory: {fl}")

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
    ms_data["runtime"] = (max(ms1.xlabels) - min(ms1.xlabels)) * 60

    # handle error_lc ### Not a permanent fix. Discuss and correct here and in pressure_curves!!!
    if "D2F" in ms_data["name"]:
        ms_data["method"] = "error_lc"
    # /handle error_lc done

    # read run logs 
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

    # read acquisition logs
    with open(acq_file, encoding=get_file_encoding(acq_file)) as f:
        content = f.readlines()

    sim_found = False
    table_head_found = False

    for idx, line in enumerate(content):
        if sim_found == False:
            if "Sim Parameters" in line:
                sim_found = True    
            continue
        
        if not table_head_found:
            if "Time" in line and "|" in line: # beginning of the table
                table_head_found = True
                table_pipes_at = [i for i, c in enumerate(line) if c == "|"] # pipes separate columns
                break   

    # idx is now at table head
    
    sim_ready = False
    comp_ready = False
    while not(sim_ready and comp_ready):
        line = content[idx]
        if sim_ready == False:
            sim_head = line.find("SIM")
            if sim_head != -1:
                sim_ready = True
        if comp_ready == False:
            comp_head = line.find("Compound")
            if comp_head != -1:
                comp_ready = True
        idx += 1

    # idx is now at dashed line

    while True:
        line = content[idx]
        if re.match(r'^[-|\s]+$', line.strip()):    
            dashes_pipes_at = [i for i, c in enumerate(line) if c == "|"]
            break
        idx += 1
        
    if table_pipes_at != dashes_pipes_at:
        print("Table pipes: ", table_pipes_at)
        print("\n")
        print("Dash pipes: ", dashes_pipes_at)       
        raise ValueError("Parsing failed")

    for i in range(len(table_pipes_at) - 1):
        if table_pipes_at[i] < sim_head < table_pipes_at[i + 1]:
            sim_slice = (table_pipes_at[i], table_pipes_at[i + 1])
        if table_pipes_at[i] < comp_head < table_pipes_at[i + 1]:
            comp_slice = (table_pipes_at[i], table_pipes_at[i + 1])
            istd_start = table_pipes_at[i + 1]

    idx += 1

    # idx is now at beginning of table entries 
    mzs = []
    compounds = []
    standards = []
    for i in range(len(ms1.ylabels)): # sim ion entries are the same as mzs corresponding to compounds
        entry = content[idx]
        sim = int(float(entry[sim_slice[0] : sim_slice[1]].strip()))
        mzs.append(sim)

        compound = entry[comp_slice[0] : comp_slice[1]].strip()
        compounds.append(compound)

        standard = int(entry[istd_start : ].strip())
        standards.append(standard)

        idx += 1
    
    if mzs != ms1.ylabels.tolist():
        raise ValueError("SIM m/z values from file do not match")

    ms_data["sim"].update({
                            "compound" : compounds,
                            "int_standard" : standards
                        })

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
            - instrument: Type of instrument (device).
            - signal: Signal.
            - date: Stringified date entry of MS run. Timestamp is still used for analysis.
            - detector: Detector name.
            - pump: Pump name.
            - vial_position: Vial Position entry of the run.
            - start_time: Start time of the run.
            - end_time: End time of the run.
            - runtime: Runtime.
            - sim: Selected Ion Monitoring data dict containing the corresponding rt, mz and ansorbances.
            - tic: Total Ion Chromatogram data dict.
        plotter (plotly.graph_objects): An instance of the Plotly graph objects class.
    
    Methods:
        plot_chromatogram: Plots EIC/TIC for the MS data.
        plot_3d: Creates a 3D Surface plot/2D Heatmap for the MS data.
        plot_bpc: Plots the Base Peak Intensity across rts.
        plot_ms: Plots the Mass Spectrum for a fixed rt across mzs.
        plot_gaussian_fit: Plots the gaussian fit created for the target window during feature extraction.
        plot_features: Plots the spread of extracted feature values.
    """

    def __init__(self, files: list = None):
        super().__init__(data_type="MassSpecAnalyses", formats=[".D"])

        self.data = []
        self.plotter = go

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
            "date": None,
            "detector": None,
            "pump": None,
            "vial_position": None, 
            "start_time": None,
            "end_time": None,
            "runtime": None,
            "sim": None,
            "tic": None
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

    def get_features(self, indices: list = None, filter_none: bool = True) -> pd.DataFrame:
        """
        Get a DataFrame of the features of the pressure curves.

        Args:
            indices (list): List of indices of the pressure curves to include in the DataFrame. If None, all curves are included.
            filter (bool): Filter out samples with None values in the feature matrix. 

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
            msd = self.data[i]
            features_dict = msd["features"]
            if filter_none and any(value is None for value in features_dict.values()):
                print(f"None type features found for sample {i}. Skipping...")
                continue
            features.append(features_dict)

        return pd.DataFrame(features)

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
            msd = self.data[i].copy()
            for key in ["sim", "tic", "features", "fit_plot"]:
                msd.pop(key, None)
            metadata.append(msd)

        return pd.DataFrame(metadata)
    
    def plot_batches(self, indices: list = None) -> go.Figure:
        """
        Plot the batches of the MS data over the timestamps.

        Args:
            indices (list): List of indices of the batches to plot. If None, all batches are plotted.
        """

        df = self.get_metadata(indices)
        df = df.sort_values("timestamp")
        batches_in_order = df["batch"].drop_duplicates().tolist()
        fig = self.plotter.Figure()
        for batch in batches_in_order:
            mask = df["batch"] == batch
            df_batch = df[mask]
            text = [
                "<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items())
                for row in df_batch.to_dict(orient="records")
            ]
            fig.add_trace(
                self.plotter.Scatter(
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
    
    def plot_chromatogram(self, indices=None, data: str = "SIM", mz: int = None, metric: str = "sum") -> go.Figure:
        """
        Plot TIC or Extracted Ion Chromatogram (XIC/SIM).

        Args:
            indices (int | list): Sample indices to plot. Defaults to all.
            data (str): "SIM" for extracted ion chromatogram, "TIC" for total ion chromatogram. Defaults to "SIM".
            mz (int): If data="SIM", the m/z value to extract. Defaults to the first m/z in the data mz[0].
            metric (str): If data="TIC", choose whether to plot "sum" or "mean" (MIC) over all intensities. Defaults to "sum"
        
        Returns:
            go.Figure: Plotly figure with chromatograms.
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
            raise ValueError("No indices provided for plotting.")

        data_key = "tic" if data.lower() == "tic" else "sim"

        fig = self.plotter.Figure()
        for i in indices:
            msd = self.data[i]
            if msd[data_key] is None:
                raise ValueError(f"No {data_key.upper()} data available for sample index {i} at {msd['path']}")

            dataset = msd[data_key]

            if data_key == "sim":  # Extracted Ion Chromatogram
                if mz is not None and mz in dataset["mz"]:
                    mz_index = dataset["mz"].tolist().index(mz)
                else:
                    mz_index = 0
                y = dataset["intensity"][:, mz_index]
            else:  # TIC data
                y = dataset["intensity"].sum(axis=1) if metric == "sum" else dataset["intensity"].mean(axis=1)

            metadata = self.get_metadata([i]).to_dict(orient="records")[0]
            text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in metadata.items())

            fig.add_trace(
                self.plotter.Scatter(
                    x=dataset["rt"],
                    y=y,
                    mode="lines",
                    name=f"{msd['name']} ({msd['sample']})",
                    text=text,
                    hovertemplate=f"{text}<br><b>Time: </b>%{{x}}<br><b>Intensity: </b>%{{y}}<extra></extra>",
                )
            )

        title = f"{data_key.upper()} Chromatograms "
        fig.update_layout(
            title=title + f"{metric}" if data=="tic" else title,
            xaxis_title="Retention Time (min)",
            yaxis_title="Intensity",
            template="simple_white",
        )
        return fig

    def plot_3d(self, indices: int = None, data: str = "sim", mz_range: tuple = None, rt_range: tuple = None, 
                plot_type: str = "surface", downsample: int = 5, mz: int = None, rt: float = None) -> go.Figure:
        """
        Plot a 3D visualization of the MS intensity matrix.

        Args:
            indices (int | list[int]): Index of sample to plot. Only one sample allowed. Defaults to None (first sample).
            data (str): "sim" or "tic" data to plot. Defaults to "sim".
            mz_range (tuple(float, float), optional): Range of m/z values to plot (min_mz, max_mz). Defaults to None (full range).
            rt_range (tuple(float, float), optional): Range of retention times to plot (min_rt, max_rt). Defaults to None (full range).
            plot_type (str): "surface" (default) for 3D surface plot or "heatmap" for 2D heatmap plot.
            downsample (int): Factor by which to downsample data along mz and rt axes. Defaults to 5.
                Increasing this value reduces plot resolution but improves performance and stability.
            mz (int, optional): Specific m/z value to highlight with marker (only for surface plot). Defaults to None.
            rt (float, optional): Specific retention time to highlight with marker (only for surface plot). Defaults to None.

        Returns:
            go.Figure: Plotly figure object.

        Notes:
            - Downsampling reduces the number of points plotted, trading off surface detail for performance.
            Higher downsample factor values make the plot faster and less memory intensive but coarser.
            - Restricting mz_range and rt_range to smaller windows can focus on regions of interest and reduce data size,
            allowing finer detail at manageable performance cost.
            - Plotting large LC-MS datasets at full resolution in 3D surface plots can crash or freeze the notebook/browser.
            Use downsampling and zooming options to avoid this.
        """
        if indices is None:
            indices = 0
        if isinstance(indices, (list, tuple)):
            if len(indices) != 1:
                raise ValueError("plot_3d only supports plotting one sample at a time.")
            indices = indices[0]
        if not isinstance(indices, int):
            raise TypeError("Index must be an integer.")

        if data.lower() == "tic":
            entry = "tic"
        else:
            entry = "sim"

        msd = self.data[indices]
        if msd[entry] is None:
            raise ValueError(f"No {entry.upper()} data available for sample index {indices}")

        intensity = msd[entry]["intensity"]  
        mz_values = msd[entry]["mz"]         
        rt_values = msd[entry]["rt"]         

        # filter mz indices based on mz_range
        mz_indices = [i for i, mzv in enumerate(mz_values) if (mz_range is None or (mz_range[0] <= mzv <= mz_range[1]))]
        # filter rt indices based on rt_range
        rt_indices = [i for i, rtv in enumerate(rt_values) if (rt_range is None or (rt_range[0] <= rtv <= rt_range[1]))]

        # filter mz and rt values
        mz_filtered = [mz_values[i] for i in mz_indices]
        rt_filtered = [rt_values[i] for i in rt_indices]

        # filter intensity matrix with nested list comprehension (simulate np.ix_)
        intensity_filtered = [
            [intensity[rt_i][mz_i] for mz_i in mz_indices]
            for rt_i in rt_indices
        ]

        # downsample indices
        mz_ds = mz_filtered[::downsample]
        rt_ds = rt_filtered[::downsample]
        intensity_ds = [
            row[::downsample]
            for row in intensity_filtered[::downsample]
        ]

        fig = self.plotter.Figure()
        metadata = self.get_metadata([indices]).to_dict(orient="records")[0]
        text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in metadata.items())

        if plot_type.lower() == "surface":
            fig.add_trace(self.plotter.Surface(
                z=intensity_ds,
                x=mz_ds,
                y=rt_ds,
                name=f"{msd['name']} ({msd['sample']})",
                text=text,
                hovertemplate=f"{text}<br><b>Intensity:</b> %{{z}}<br><b>RT:</b> %{{y}}<br><b>m/z:</b> %{{x}}<extra></extra>"
            ))

            # highlight specific mz and rt if given
            if mz is not None and rt is not None:
                # find closest mz index
                mz_idx = min(range(len(mz_ds)), key=lambda i: abs(mz_ds[i] - mz))
                # find closest rt index
                rt_idx = min(range(len(rt_ds)), key=lambda i: abs(rt_ds[i] - rt))
                intensity_val = float(intensity_ds[rt_idx][mz_idx])

                fig.add_trace(self.plotter.Scatter3d(
                    x=[mz_ds[mz_idx]],
                    y=[rt_ds[rt_idx]],
                    z=[intensity_val],
                    mode='markers',
                    marker=dict(size=6, color='red'),
                    name="Selected Point",
                    hovertemplate=f"{text}<br><b>Intensity:</b> {intensity_val}<br><b>RT:</b> {rt_ds[rt_idx]}<br><b>m/z:</b> {mz_ds[mz_idx]}<extra></extra>"
                ))

            fig.update_layout(
                title=f"3D Surface Plot ({entry.upper()})",
                scene=dict(
                    xaxis_title="m/z",
                    yaxis_title="RT (min)",
                    zaxis_title="Intensity"
                ),
                width=1000,
                height=700,
                template="simple_white"
            )

        elif plot_type.lower() == "heatmap":
            fig.add_trace(self.plotter.Heatmap(
                z=intensity_ds,
                x=mz_ds,
                y=rt_ds,
                colorscale="Viridis",
                colorbar=dict(title="Intensity"),
                hoverongaps=False
            ))
            fig.update_layout(
                title=f"2D Heatmap Plot ({entry.upper()})",
                xaxis_title="m/z",
                yaxis_title="RT (min)",
                width=1000,
                height=700,
                template="simple_white"
            )

        else:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Choose 'surface' or 'heatmap'.")

        return fig

    def plot_bpc(self, indices=None) -> go.Figure:
        """
        Plot Base Peak Chromatogram (BPC) for given samples by picking the highest peak across all mzs for each rt.

        Args:
            indices (int | list): Sample indices to plot. Defaults to all.

        Returns:
            go.Figure: BPC intensity plot over retention time.
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
            raise ValueError("No indices provided for plotting.")

        fig = self.plotter.Figure()

        for i in indices:
            msd = self.data[i]
            # use SIM data (intensity matrix) to find base peak
            dataset = msd.get("sim", None)
            if dataset is None:
                raise ValueError(f"No SIM data available for sample index {i} at {msd['path']}")

            intensity_matrix = dataset["intensity"]  # shape: (rt, mz)
            base_peak_intensity = intensity_matrix.max(axis=1)  # max intensity per retention time

            metadata = self.get_metadata([i]).to_dict(orient="records")[0]
            text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in metadata.items())

            fig.add_trace(
                self.plotter.Scatter(
                    x=dataset["rt"],
                    y=base_peak_intensity,
                    mode="lines",
                    name=f"{msd['name']} ({msd['sample']})",
                    text=text,
                    hovertemplate=f"{text}<br><b>Retention Time: %{{x}}<br><b>Base Peak Intensity: %{{y}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Base Peak Chromatogram (BPC)",
            xaxis_title="Retention Time (min)",
            yaxis_title="Intensity",
            template="simple_white",
        )
        return fig

    def plot_ms(self, indices=None, data: str = "SIM", rt: float = None) -> go.Figure:
        """
        Plot Mass Spectrum (intensity vs m/z) at a fixed retention time.

        Args:
            indices (int | list): Sample indices to plot. Defaults to all.
            data (str): "SIM" or "TIC". Defaults to "SIM".
            rt (float): Retention time at which to extract the mass spectrum. If None, uses the first RT.

        Returns:
            go.Figure: Mass spectrum plot.
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
            raise ValueError("No indices provided for plotting.")

        data_key = "tic" if data.lower() == "tic" else "sim"

        fig = self.plotter.Figure()

        for i in indices:
            msd = self.data[i]
            dataset = msd.get(data_key, None)
            if dataset is None:
                raise ValueError(f"No {data_key.upper()} data available for sample index {i} at {msd['path']}")

            rt_array = dataset["rt"]
            if rt is not None:
                rt_index = abs(rt_array - rt).argmin()
            else:
                rt_index = 0

            mz_array = dataset["mz"]

            intensity = dataset["intensity"][rt_index, :] if data_key == "sim" else dataset["intensity"]  # TIC might be 1D

            metadata = self.get_metadata([i]).to_dict(orient="records")[0]
            text = "<br>".join(f"<b>{k}</b>: {v}" for k, v in metadata.items())

            fig.add_trace(
                self.plotter.Scatter(
                    x=mz_array,
                    y=intensity,
                    mode="lines",
                    name=f"{msd['name']} ({msd['sample']})",
                    text=text,
                    hovertemplate=f"{text}<br><b>m/z: %{{x}}<br><b>Intensity: %{{y}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"{data_key.upper()} Mass Spectrum at RT={rt if rt else rt_array[0]:.2f}",
            xaxis_title="m/z",
            yaxis_title="Intensity",
            template="simple_white",
        )
        return fig

    def plot_gaussian_fit(self, indices: int | list = None) -> go.Figure | dict:
        """
        Plot the gaussian fit for the given indices created during feature extraction.

        Args:
            indices (int|list): Single integer or list of integer index values(s) of data to plot. Plotting for all indices is computationally inefficient. 
            See MassSpecMethodExtractFeaturesNative args for remaining details. 

        Returns:
            go.Figure | dict({index : go.Figure}): A plot object or a dict of such objects linked on the passed indices.
        """
        if indices is None:
            indices = list(range(len(self.data)))
        elif isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, tuple):
            indices = list(indices)
        if not isinstance(indices, list) or not isinstance(indices[0], int):
            raise TypeError("Indices should be a list of integers.")
        if len(indices) == 0:
            raise ValueError("No indices provided for plotting.")
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available.")

        figs = {}
        for i in indices:
            msd = self.data[i]
            plot = msd.get("fit_plot")
            if plot is None:
                print(f"No features have been extracted for sample {i}") 
            figs[i] = plot

        if len(list(figs)) == 1:
            return plot

        return figs
    
    def plot_features(self, indices: list = None, normalize: bool = True) -> go.Figure:
        """
        Plot calculated features of the MS chromatograms.

        Args:
            indices (list): List of indices of the data to plot. If None, features for all data are plotted.
        """
        ft = self.get_features(indices, filter_none = False).to_dict(orient = "records")
        mt = self.get_metadata(indices).to_dict(orient="records")

        valid = [(ft_row, mt_row) for ft_row, mt_row in zip(ft, mt)
            if all(value is not None for value in ft_row.values())]
        
        ft = pd.DataFrame([f for f, _ in valid])

        if normalize:
            # normalize each column of the DataFrame while handling NaN values
            ft = ft.fillna(0)  # Replace NaN with 0 
            for col in ft.columns:
                if ft[col].dtype in [int, float] and ft[col].max() != 0:
                    ft[col] = ft[col] / ft[col].max()
                    #ft[col] = (ft[col] - ft[col].min()) / (ft[col].max() - ft[col].min())
    
        ft = ft.to_dict(orient="records")

        text = ["<br>".join(f"<b>{k}: </b>{v}" for k, v in row.items()) for row in mt]

        fig = self.plotter.Figure()
        for i, fti in enumerate(ft):
            fti = ft[i]
            mti = mt[i]

            fig.add_trace(
                self.plotter.Scatter(
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
    
