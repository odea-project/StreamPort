import streamlit as st
import os
import asyncio
import time

## Known Paths ##
#C:/Users/Sandeep/Desktop/ExtractedSignals
#C:/Users/Sandeep/Desktop/Error-LC/Method-Data

## Functions ##
# Data Collection
def start(path):
    batches = os.listdir(path)
    batches = [os.path.join(path, file) for file in batches]

    files = []
    for batch in batches:
        batch_files = os.listdir(batch)
        batch_files = [os.path.join(batch, file) for file in batch_files if ".D" in file]
        files.extend(batch_files)
    return files


st.title("StreamPort")

st.markdown(
    """ 
    This is a prototype of the :blue[StreamPort] anomaly detection package. 
    
    Choose the workflow to begin
    """
)

# Process Selection
pipeline = st.selectbox("What would you like to do?", ["Anomaly Detection"])
st.session_state["pipeline"] = pipeline 

if pipeline == "Anomaly Detection":
    path = st.text_input("Input analyses path as plaintext without quotes.")
    if st.button("Read files") or "files" in st.session_state:
        if "files" not in st.session_state:
            if not path:
                st.error("Invalid path entered")
            else:
                files = start(path)
                st.session_state["files"] = files
                with st.spinner("processing..."):
                    time.sleep(2)
                st.success(f"Data has been read successfully. {len(files)} files were found.")
        else:
            files = st.session_state["files"]
            st.success(f"{len(files)} files loaded.")

        ana_type = st.selectbox("Select data to analyse", ["Pressure Curves", "Mass Spec", "Actuals"])
        if st.button("Load Analyses") or "analyses" in st.session_state:

            from device.methods import PressureCurvesMethodExtractFeaturesNative
            from device.methods import MassSpecMethodExtractFeaturesNative
            
            from machine_learning.analyses import MachineLearningAnalyses
            from machine_learning.methods import MachineLearningMethodIsolationForestSklearn
            from machine_learning.methods import MachineLearningScaleFeaturesScalerSklearn
            
            if "analyses" not in st.session_state:
                if ana_type == "Pressure Curves":
                    from device.analyses import PressureCurvesAnalyses
                    analyses = PressureCurvesAnalyses(files=files)

                elif ana_type == "Mass Spec":
                    from device.analyses import MassSpecAnalyses
                    analyses = MassSpecAnalyses(files=files)
                    
                # elif ana_type == "Actuals":
                #     from device.analyses import ActualsAnalyses
                #     analyses = ActualsAnalyses(files=files)

                else:
                    st.write("No Analyses type selected. Please specify the data to analyse.")
                    analyses = None
                st.session_state["analyses"] = analyses

            else:
                analyses = st.session_state["analyses"]
            st.write("Number of analyses: ", len(analyses.data))

            if st.button("Run with default settings"):
                if ana_type == "Pressure Curves":
                    processor = PressureCurvesMethodExtractFeaturesNative()
              
                elif ana_type == "Mass Spec":
                    processor = MassSpecMethodExtractFeaturesNative()

            elif st.button("Edit settings"):
                if ana_type == "Pressure Curves":
                    st.markdown(
                        """ 
                        # Feature Extraction
                        ### Period for time series decomposition decides the number of observations that make up a seasonal cycle. Default is 10. 
                        ### Window size chooses the number of observations to be averaged to smooth the curve. Defaults is 7.
                        ### Number of Bins splits the data into intervals and considers statistical features within those intervals. Default is 4. 
                        ### Crop specifies the number of leading and trailing observations of a signal that must be removed. Default is 2.
                        ### You may choose to use the default values, set automatically.
                        """
                    )
                    # Period (integers): 10, 20, 30 (default: 10)
                    period = st.slider("Period for time-series decomposition", min_value=10, max_value=30, step=10, value=10)

                    # Window Size: 5 to 13 (odd numbers), default: 7 (index=1 in range(5,14,2))
                    window_size = st.slider("Smoothing Window size", min_value=5, max_value=13, step=2, value=7)

                    # Bins: 2 to 8, default: 4 (index=2 in range(2,9))
                    bins = st.slider("Number of bins to aggregate data", min_value=2, max_value=8, step=1, value=4)

                    # Crop: 1 to 4, default: 2 (index=1 in range(1,5))
                    crop = st.slider("Number of entries to crop", min_value=1, max_value=4, step=1, value=2)
                    
                    processor = PressureCurvesMethodExtractFeaturesNative(
                        period=period, 
                        window_size=window_size, 
                        bins=bins,
                        crop=crop
                        )
                
                elif ana_type == "Mass Spec":
                    st.markdown(
                        """
                        # Feature Extraction
                        ### data (str): "sim" or "tic" based on user's choice. Defaults to sim, and targets the closest mz if the mz input parameter is invalid.
                        ### rt (float): The retention time for the target to be analysed.
                        ### mz (float): The mz for the target.
                        ### rt_window (int): The minimum distance by seconds before and after current one to be considered when finding peaks (default = 8s). Any peaks within this distance from each other will be disregarded.
                        ### mz_window (float): Range of adjacent mz value to be considered for 2D tile/window creation. Defaults to 1.0 Da (mz(s) within 1.0 to the current one, here, totalling 1 mz before and after the target).
                        ### smooth (int | bool): User may provide an integer kernel size to choose whether the signal must be pre-treated using smoothing. Default is False(No smoothing). Note: If an int n is passed, the window will be nxn over the 2D intensity array, where n is always automatically limited to the values 3, 5, and 7.
                        ### exclude (str): Choice of whether to exclude "Flush", "Blank" or other such runs from the analysis. This will set the features for the respective samples to None, without removing the samples.
                        """
                    )

                    data = st.selectbox("Data type", ["SIM", "TIC"])

                    rt_options = analyses.data[0][data.lower()]["rt"]
                    rt = st.selectbox("Retention time", rt_options)

                    mz_options = analyses.data[0][data.lower()]["mz"]
                    mz = st.selectbox("M/Z", mz_options)

                    rt_window = st.slider("RT window", min_value=5, max_value=30, value=8)
                    
                    mz_window = st.slider("M/Z window", min_value=0.1, max_value=2.0, value=1.0)

                    smooth = st.selectbox("Smoothing", ["No smoothing", 3, 5, 7])

                    exclude = st.text_input("Enter the data to remove. If more than one, separate them with commas: ", value =None)
                    to_exclude = [item.strip().lower() for item in exclude.split(",") if item.strip()]

                    from device.methods import MassSpecMethodExtractFeaturesNative
                    processor = MassSpecMethodExtractFeaturesNative(
                        data=data,
                        rt=rt,
                        mz=mz,
                        rt_window_size=rt_window,
                        mz_window_size=mz_window,
                        smooth=smooth,
                        exclude=to_exclude
                    )

            analyses = processor.run(analyses)






                plot_type = st.selectbox("Select plot type", ["Methods", "Batches"])
                if plot_type == "Methods":
                    plot = analyses.plot_methods()
                elif plot_type == "Batches":
                    plot = analyses.plot_batches()
                else:
                    plot = None
                st.session_state["plot"] = plot    
                st.plotly_chart(plot, use_container_width=True)

                method = st.selectbox("Select method", analyses.get_methods())
                method_indices = analyses.get_method_indices(method)
                
                plot_type = st.selectbox("Select plot type", ["Pressure Curves", "Features", "Features Raw"])
                if plot_type == "Methods":
                    plot = analyses.plot_methods()
                elif plot_type == "Batches":
                    plot = analyses.plot_batches()
                else:
                    plot = None

                st.session_state["plot"] = plot    
                st.plotly_chart(plot, use_container_width=True)

