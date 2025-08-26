import streamlit as st


import os
import time

from device.analyses import PressureCurvesAnalyses ##this requires a change in import statements in device and ml
#from device.methods import PressureCurvesMethodExtractFeaturesNative

#from machine_learning.analyses import MachineLearningAnalyses
#from machine_learning.methods import MachineLearningScaleFeaturesScalerSklearn, MachineLearningMethodIsolationForestSklearn

st.title("StreamPort")

st.markdown(
    """ 

    This is a prototype of the :blue[StreamPort] anomaly detection package. 
    
    Enter the path to the analysis data and click start to begin.
    """
)

#C:/Users/Sandeep/Desktop/ExtractedSignals
#C:/Users/Sandeep/Desktop/Error-LC/Method-Data

def start(path):
    batches = os.listdir(path)
    batches = [os.path.join(path, file) for file in batches]

    error_lc_files = []
    for batch in batches:
        batch_files = os.listdir(batch)
        batch_files = [os.path.join(batch, file) for file in batch_files if ".D" in file]
        error_lc_files.extend(batch_files)

    analyses = PressureCurvesAnalyses(files=error_lc_files)
    return analyses

path = st.text_input("Input analyses path as plaintext - no quotes!")
if path is None or path == "":
    st.write("Invalid path entered")

if st.button("Start"):
    analyses = start(path)
    with st.spinner("processing..."):
        time.sleep(5)
    st.write("Data has been read successfully. Number of analyses: ", len(analyses.data))

    if st.button("Plot batches"):
        batch_plot = analyses.plot_batches()
        st.write(type(batch_plot))
        st.plotly_chart(batch_plot, use_container_width=True)
        


