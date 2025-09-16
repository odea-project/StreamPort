import streamlit as st
import time
import threading
from app_utils.functions import *


st.set_page_config(page_title="Anomaly Detection - Workflow", layout="wide")


## Imports ##
from src.StreamPort.core import Engine
from src.StreamPort.device.methods import PressureCurvesMethodExtractFeaturesNative, MassSpecMethodExtractFeaturesNative

# Initialize session state
if "workflows" not in st.session_state:
    st.session_state["workflows"] = {}

# Function to run workflow in a background thread
def run_workflow_thread(workflow_id, engine, ana_type, scaler_type, processor, date_threshold_min):
    try:
        wf = st.session_state["workflows"][workflow_id]
        wf["status"] = "Running"
        wf["progress"] = 0

        extract_features(engine, processor)
        st.write("Num. analyses after exclusion: ", len(engine.analyses.data))

        if ana_type == "Pressure Curves":
            train_indices, test_indices = select_train_set_pc(engine=engine, date_threshold_min=date_threshold_min)
        elif ana_type == "Mass Spec":
            train_indices, test_indices = select_train_set_ms(engine=engine, date_threshold_min=date_threshold_min)
        else:
            wf["status"] = "Failed: Invalid analysis type"
            return

        ml = scale_data(engine, train_indices, scaler_type)
        ml = create_iforest(ml)

        # currently follows the serial test pattern using known/available test cases. Update using sensor info for real-time testing
        for i, index in enumerate(test_indices):
            wf = st.session_state["workflows"][workflow_id]
            if wf.get("cancel"):
                wf["status"] = "Cancelled"
                st.session_state["workflows"][workflow_id] = wf
                return
            ml = test_sample(ml, engine, index)
            wf["progress"] = int(100 * (i + 1) / len(test_indices))
            time.sleep(0.05)  # Simulate processing time

        wf["ml"] = ml
        wf["status"] = "Completed"

    except Exception as e:
        wf["status"] = f"Failed: {str(e)}"


st.title("Anomaly Detection")
st.markdown("The results of the analyses can be accessed when they are available")

path = st.text_input("Input analyses path (plaintext, no quotes)")

collect = False
if st.button("Read files") or "files" in st.session_state:
    if "files" not in st.session_state:
        if not path:
            st.error("Invalid path entered")
        else:
            collect = True            
    else:
        if path and path != st.session_state["path"]:
            st.write("New path detected. Overwriting previous files...")
            collect = True
        else:
            files = st.session_state["files"]
            st.success(f"{len(files)} files re-used from session.")

    if collect:
        files = collect_data(path)
        st.session_state["path"] = path
        st.session_state["files"] = files
        st.success(f"{len(files)} files loaded from {path}.")

    # UI for analysis selection
    ana_type = st.selectbox("Select data to analyse", ["--", "Pressure Curves", "Mass Spec"])
    name = st.text_input("Name your workflow (optional)")

    if "analyses" not in st.session_state:
        analyses = create_analyses(files, ana_type)
        if name:
            engine = Engine(metadata={"name":name, "author":None, "path":None}, analyses=analyses)
        else:
            engine = Engine(analyses=analyses)
        if analyses is None:
            st.error("Unknown analysis type")
            st.stop()
        st.session_state["analyses"] = analyses
        st.session_state["engine"] = engine

    else:
        analyses = st.session_state["analyses"]
        engine = st.session_state["engine"]

    st.write("Number of analyses:", len(analyses.data))

    processor = None
    scaler_type = None
    threshold = "auto"
    params = {}
    
    #exclude (list | str): Choice of whether to exclude "Flush", "Blank" or other such runs from the analysis. This will set the features for the respective samples to None, without removing the samples.

    if ana_type == "Pressure Curves":
        params["period"] = st.slider("Period for seasonal decomposition of curves. Default is 10", 10, 30, step=10, value=10)
        params["window_size"] = st.slider("Window Size/Resolution for baseline correction. Default is 7", 5, 13, step=2, value=7)
        params["bins"] = st.slider("Number of Bins/Intervals for feature extraction. Default is 4", 2, 8, value=4)
        params["crop"] = st.slider("The number of elements to Crop from the beginning and end of the pressure vector to remove unwanted artifacts. Default is 2.", 1, 4, value=2)
        exclude = st.text_input("Exclude samples (comma-separated)", value="")
        params["exclude"] = [e.strip() for e in exclude.split(",") if e.strip()]
        processor = PressureCurvesMethodExtractFeaturesNative(**params)

    elif ana_type == "Mass Spec":
        data = st.selectbox("Data type: 'sim' or 'tic' based on user's choice. Defaults to SIM, and targets the closest mz if the mz input parameter is invalid.", ["--", "SIM", "TIC"])
        rt_options = analyses.data[0][data.lower()]["rt"]
        mz_options = analyses.data[0][data.lower()]["mz"]
        params["data"] = data
        params["rt"] = st.selectbox("RT: The retention time for the target to be analysed", rt_options)
        params["mz"] = st.selectbox("MZ: The M/Z for the target", mz_options)
        params["rt_window"] = st.slider("RT Window: The minimum distance (s) before and after target RT to be considered when finding peaks (default = 8s window)", 5, 30, value=8)
        params["mz_window"] = st.slider("MZ Window: Range of adjacent mz value to be considered for 2D tile/window creation. Defaults to 1.0 Da window", 0.1, 2.0, value=1.0)
        params["smooth"] = st.selectbox("Smoothing: User may provide a kernel size and choose whether the signal must be pre-treated using smoothing. Default is None(No smoothing)", ["No smoothing", 3, 5, 7])
        exclude = st.text_input("Exclude samples (comma-separated): Choice of whether to exclude 'Flush', 'Blank' or other such runs from the analysis. This will remove the samples once the applied workflow is run. Default = None(Do not exclude)", value="")
        params["exclude"] = [e.strip() for e in exclude.split(",") if e.strip()]
        processor = MassSpecMethodExtractFeaturesNative(**params)
    
    date_threshold_min = st.date_input("Enter cutoff date for Train set. Data from before the given date will be used to train the model.")
    scaler_type = st.selectbox("Choose Scaler Type", ["StandardScaler", "MinMaxScaler", "Normalizer", "MaxAbsScaler", "RobustScaler"])
    threshold = st.selectbox("Set Detection Threshold", ["Min", "Auto", "Custom"])
    if threshold == "Custom":
        threshold = st.number_input("Enter a float threshold value.")
        if not isinstance(threshold, float):
            threshold = float(threshold)

    if st.button("Run Workflow"):
        workflow_id = int(time.time() * 1000)
        st.session_state["workflows"][workflow_id] = {
            "name": engine.metadata["name"] if name else f"{ana_type} workflow {workflow_id}",
            "start_time": time.time(),
            "status": "Running",
            "progress": 0,
            "cancel": False,
            "ml": None,
            "engine": engine,
            "analyses": analyses,  # optional for later reference
            "type": ana_type,
            "scaler": scaler_type,
            "threshold": threshold,
            "processor": processor,
        }

        threading.Thread(
            target=run_workflow_thread,
            args=(workflow_id, engine, ana_type, scaler_type, processor, date_threshold_min),
            daemon=True
        ).start()
        st.success(f"Workflow {name} started in background.")