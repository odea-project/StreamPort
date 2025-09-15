import streamlit as st
import time
import threading
import uuid
from app_utils.functions import *


st.set_page_config(page_title="Anomaly Detection - Workflow", layout="wide")


## Imports ##
from device.methods import PressureCurvesMethodExtractFeaturesNative, MassSpecMethodExtractFeaturesNative
from device.analyses import PressureCurvesAnalyses, MassSpecAnalyses

# Initialize session state
if "workflows" not in st.session_state:
    st.session_state["workflows"] = {}

# Function to run workflow in a background thread
def run_workflow_thread(workflow_id, analyses, ana_type, scaler_type, processor):
    try:
        wf = st.session_state["workflows"][workflow_id]
        wf["status"] = "Running"
        wf["progress"] = 0

        analyses = extract_features(analyses, processor)

        if ana_type == "Pressure Curves":
            train_indices, test_indices = select_train_set_pc(pc=analyses)
        elif ana_type == "Mass Spec":
            train_indices, test_indices = select_train_set_ms(analyses)
        else:
            wf["status"] = "Failed: Invalid analysis type"
            return

        ml = scale_data(analyses, train_indices, scaler_type)
        ml = create_iforest(ml)

        # currently follows the serial test pattern using known/available test cases. Update using sensor info for real-time testing
        for i, index in enumerate(test_indices):
            wf = st.session_state["workflows"][workflow_id]
            if wf.get("cancel"):
                wf["status"] = "Cancelled"
                st.session_state["workflows"][workflow_id] = wf
                return
            ml = test_sample(ml, analyses, index)
            wf["progress"] = int(100 * (i + 1) / len(test_indices))
            time.sleep(0.05)  # Simulate processing time

        wf["ml"] = ml
        wf["status"] = "Completed"

    except Exception as e:
        wf["status"] = f"Failed: {str(e)}"


st.title("Anomaly Detection")
st.markdown("The results of the analyses can be accessed when they are available")

path = st.text_input("Input analyses path (plaintext, no quotes)")

if st.button("Read files") or "files" in st.session_state:
    if "files" not in st.session_state:
        if not path:
            st.error("Invalid path entered")
        else:
            files = collect_data(path)
            st.session_state["files"] = files
            st.success(f"{len(files)} files loaded.")
    else:
        files = st.session_state["files"]
        st.success(f"{len(files)} files re-used from session.")

    # UI for analysis selection
    ana_type = st.selectbox("Select data to analyse", ["--", "Pressure Curves", "Mass Spec"])
    
    if "analyses" not in st.session_state:
        if ana_type == "Pressure Curves":
            analyses = PressureCurvesAnalyses(files=files)
        elif ana_type == "Mass Spec":
            analyses = MassSpecAnalyses(files=files)
        else:
            st.error("Unknown analysis type")
            st.stop()

    else:
        analyses = st.session_state["analyses"]

    st.write("Number of analyses:", len(analyses.data))

    use_default = st.checkbox("Use default processor settings", value=True)

    processor = None
    scaler_type = None
    threshold = "auto"
    params = {}

    if not use_default:

        if ana_type == "Pressure Curves":
            params["period"] = st.slider("Period", 10, 30, step=10, value=10)
            params["window_size"] = st.slider("Window Size", 5, 13, step=2, value=7)
            params["bins"] = st.slider("Bins", 2, 8, value=4)
            params["crop"] = st.slider("Crop", 1, 4, value=2)
            processor = PressureCurvesMethodExtractFeaturesNative(**params)

        elif ana_type == "Mass Spec":
            data = st.selectbox("Data type", ["--", "SIM", "TIC"])
            rt_options = analyses.data[0][data.lower()]["rt"]
            mz_options = analyses.data[0][data.lower()]["mz"]
            params["data"] = data
            params["rt"] = st.selectbox("RT", rt_options)
            params["mz"] = st.selectbox("MZ", mz_options)
            params["rt_window"] = st.slider("RT Window", 5, 30, value=8)
            params["mz_window"] = st.slider("MZ Window", 0.1, 2.0, value=1.0)
            params["smooth"] = st.selectbox("Smoothing", ["No smoothing", 3, 5, 7])
            exclude = st.text_input("Exclude samples (comma-separated)", value="")
            params["exclude"] = [e.strip() for e in exclude.split(",") if e.strip()]
            processor = MassSpecMethodExtractFeaturesNative(**params)
        
        scaler_type = st.selectbox("Choose Scaler Type", ["StandardScaler", "MinMaxScaler", "Normalizer", "MaxAbsScaler", "RobustScaler"])
        threshold = st.select_box("Set Detection Threshold", ["Min", "Auto", "Custom"])
        if threshold == "Custom":
            threshold = st.number_input("Enter a float threshold value.")
            if not isinstance(threshold, float):
                threshold = float(threshold)
    else:
        if ana_type == "Pressure Curves":
            processor = PressureCurvesMethodExtractFeaturesNative()
        elif ana_type == "Mass Spec":
            processor = MassSpecMethodExtractFeaturesNative()


    if st.button("Run Workflow"):
        workflow_id = str(uuid.uuid4())
        st.session_state["workflows"][workflow_id] = {
            "start_time": time.time(),
            "status": "Running",
            "progress": 0,
            "cancel": False,
            "ml": None,
            "analyses": analyses,  # optional for later reference
            "type": ana_type,
            "scaler": scaler_type,
            "threshold": threshold,
            "processor": processor,
        }

        threading.Thread(
            target=run_workflow_thread,
            args=(workflow_id, analyses, ana_type, scaler_type, processor),
            daemon=True
        ).start()
        st.success(f"Workflow {workflow_id[:8]} started in background.")