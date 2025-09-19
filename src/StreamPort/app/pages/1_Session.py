import streamlit as st
import time
import copy
from threading import Thread
from app_utils.functions import *
from app_utils.workflow_state import workflow_lock, shared_workflow_data

st.set_page_config(page_title="Anomaly Detection - Workflow", layout="wide")

## Imports ##
from src.StreamPort.core import Engine
from src.StreamPort.device.methods import PressureCurvesMethodExtractFeaturesNative, MassSpecMethodExtractFeaturesNative

# Function to run workflow in a background thread
def run_workflow_thread(workflow_id, engine, ana_type, method, scaler_type, processor, date_threshold_min):

    fail_flag = None    
    with workflow_lock:
        shared_workflow_data[workflow_id] = {
            "status": "Running",
            "progress": 0,
            "ml": None,
            "results": {}
        }
    
    try:
        fail_flag = "Feature Extraction"
        extract_features(engine, processor)

        if ana_type == "Pressure Curves":
            fail_flag = "Train Selection PC"
            train_indices, test_indices = select_train_set_pc(engine=engine, method=method, date_threshold_min=date_threshold_min)
        elif ana_type == "Mass Spec":
            fail_flag = "Train Selection MS"
            train_indices, test_indices = select_train_set_ms(engine=engine, date_threshold_min=date_threshold_min)
        else:
            raise ValueError("Invalid analysis type")

        fail_flag = "Data Scaling"
        ml = scale_data(engine, train_indices, scaler_type)
        fail_flag = "Iforest Creation"
        ml = create_iforest(ml)

        # currently follows the serial test pattern using known/available test cases. Update using sensor info for real-time testing
        for i, index in enumerate(test_indices[:len(test_indices)//2]):
            with workflow_lock:
                if shared_workflow_data[workflow_id].get("cancel"):
                    shared_workflow_data[workflow_id]["status"] = "Cancelled"
                    return
            fail_flag = "Testing"    
            ml = test_sample(ml, engine, index, n_tests=5)
            new_ml = copy.deepcopy(ml)
            time.sleep(0.05)  # Simulate processing time

            with workflow_lock:
                shared_workflow_data[workflow_id]["ml"] = new_ml
                shared_workflow_data[workflow_id]["progress"] = int(100 * (i + 1) // (len(test_indices) // 2))
                shared_workflow_data[workflow_id]["results"][index] = {
                    "ml": new_ml,
                    "engine": engine
                }

        with workflow_lock:
            shared_workflow_data[workflow_id]["status"] = "Completed"

    except Exception as e:
        with workflow_lock:
            shared_workflow_data[workflow_id]["status"] = f"Failed: {str(e)} at {fail_flag}"
        return

# Initialize session state
if "workflows" not in st.session_state:
    st.session_state["workflows"] = {}

if "path" not in st.session_state:
    st.session_state["last_path"] = None
if "last_ana_type" not in st.session_state:
    st.session_state["last_ana_type"] = None
if "start_clicked" not in st.session_state:
    st.session_state["start_clicked"] = False

st.title("Anomaly Detection")

name = st.text_input("Name your workflow (optional)", value = None)

path = st.text_input("Input analyses path (plaintext, no quotes). Files from the same path do not need to be reloaded on page refresh.")

collect = False
if st.button("Read files") or collect:
    if "files" not in st.session_state:
        if not path:
            st.error("Invalid path entered")
        else:
            st.session_state["path"] = path
            collect = True            
    else:
        if path and path != st.session_state["path"]:
            st.write("New path detected. Overwriting previous files...")
            st.session_state["start_clicked"] = False
            st.session_state["path"] = path
            collect = True
        else:
            files = st.session_state["files"]
            st.success(f"{len(files)} files re-used from session.")
            st.session_state["loaded"] = True

    if collect:
        path = st.session_state["path"]
        files = collect_data(path)
        st.session_state["files"] = files
        st.success(f"{len(files)} files loaded from {path}.")
        st.session_state["loaded"] = True

if st.session_state.get("loaded"):    
    # UI for analysis selection
    ana_type = st.selectbox("Select data to analyse", ["--", "Pressure Curves", "Mass Spec"])

    if ana_type == "--":
        st.info("Please select a valid analysis type to proceed.")
        st.stop()

    if ana_type != st.session_state.get("last_ana_type"):
        st.session_state["start_clicked"] = False
        st.session_state["last_ana_type"] = ana_type

    if st.button("Start"):
        
        if "analyses" not in st.session_state or not st.session_state["start_clicked"]:
            files = st.session_state["files"]
            with st.spinner(f"Finding {ana_type} data..."):
                analyses = create_analyses(files, ana_type)
            if name:
                metadata={"name":name, "author":None, "path":None}
            else:
                metadata=None
            engine = Engine(metadata=metadata, analyses=analyses)
            if analyses is None:
                st.error("Unknown analysis type")
                st.stop()
            st.session_state["analyses"] = analyses
            st.session_state["engine"] = engine

        else:
            analyses = st.session_state["analyses"]
            engine = st.session_state["engine"]

        st.write("Number of analyses:", len(analyses.data))
        st.session_state["start_clicked"] = True

    if st.session_state.get("start_clicked", False):
        processor = None
        scaler_type = None
        threshold = "auto"
        params = {}
        engine = st.session_state["engine"]
        analyses = st.session_state["analyses"]
        method = None

        if ana_type == "Pressure Curves":
            method = st.selectbox("Choose the method/substance to analyze. Default is SAA_411_Pac.M", engine.analyses.get_methods())

            params["period"] = st.slider("Period for seasonal decomposition of curves. Default is 10", 10, 30, step=10, value=10)
            params["window_size"] = st.slider("Window Size/Resolution for baseline correction. Default is 7", 5, 13, step=2, value=7)
            params["bins"] = st.slider("Number of Bins/Intervals for feature extraction. Default is 4", 2, 8, value=4)
            params["crop"] = st.slider("The number of elements to Crop from the beginning and end of the pressure vector to remove unwanted artifacts. Default is 2.", 1, 4, value=2)
            exclude = st.text_input("Exclude samples (comma-separated)'Flush', 'Blank' or other such runs from the analysis. This will remove the samples once the applied workflow is run. Default = None(Do not exclude)", value="")
            params["exclude"] = [e.strip() for e in exclude.split(",") if e.strip()]
            processor = PressureCurvesMethodExtractFeaturesNative(**params)

        elif ana_type == "Mass Spec":
            data = st.selectbox("Data type: 'sim' or 'tic' based on user's choice. Defaults to SIM, and targets the closest mz if the mz input parameter is invalid.", ["SIM", "TIC"])
            msd = engine.analyses.data[0]
            rt_options = msd[f"{data.lower()}"]["rt"]
            mz_options = msd[f"{data.lower()}"]["mz"]
            params["data"] = data
            params["rt"] = st.selectbox("RT: The retention time for the target to be analysed", rt_options)
            params["mz"] = st.selectbox("MZ: The M/Z for the target", mz_options)
            params["rt_window_size"] = st.slider("RT Window: The minimum distance (s) before and after target RT to be considered when finding peaks (default = 8s window)", 5, 30, value=8)
            params["mz_window_size"] = st.slider("MZ Window: Range of adjacent mz value to be considered for 2D tile/window creation. Defaults to 1.0 Da window", 0.1, 2.0, value=1.0)
            params["smooth"] = st.selectbox("Smoothing: User may provide a kernel size and choose whether the signal must be pre-treated using smoothing. Default is None(No smoothing)", ["No smoothing", 3, 5, 7])
            exclude = st.text_input("Exclude samples (comma-separated) 'Flush', 'Blank' or other such runs from the analysis. This will remove the samples once the applied workflow is run. Default = None(Do not exclude)", value="")
            params["exclude"] = [e.strip() for e in exclude.split(",") if e.strip()]
            processor = MassSpecMethodExtractFeaturesNative(**params)

        date_threshold_min = st.date_input("Enter cutoff date for Train set. Data from before the given date will be used to train the model.")
        scaler_type = st.selectbox("Choose Scaler Type", ["StandardScaler", "MinMaxScaler", "Normalizer", "MaxAbsScaler", "RobustScaler"])
        threshold = st.selectbox("Set Detection Threshold", ["Min", "Auto", "Custom"])
        if threshold == "Custom":
            threshold = st.number_input("Enter a float threshold value.")
            if not isinstance(threshold, float):
                threshold = float(threshold)
        else:
            threshold = threshold.lower()

        if st.button("Run Workflow"):
            workflow_id = int(time.time() * 1000)

            if "workflows" not in st.session_state:
                st.session_state["workflows"] = {}

            st.session_state["workflows"][workflow_id] = {
                "name": name if name else f"{ana_type} workflow {workflow_id}",
                "start_time": time.time(),
                "type": ana_type,
                "scaler": scaler_type,
                "threshold": threshold,
                "processor": processor,
            }

            Thread(
                target=run_workflow_thread,
                args=(workflow_id, engine, ana_type, method, scaler_type, processor, date_threshold_min),
                daemon=True
            ).start()
            st.success(f"Workflow {name if name else workflow_id} started in background.")
            st.session_state["loaded"] = False