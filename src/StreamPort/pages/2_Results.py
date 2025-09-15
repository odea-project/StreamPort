import streamlit as st

st.set_page_config(page_title="Anomaly Detection - Results", layout="wide")

workflow_ids = list(st.session_state["workflows"].keys())
copy = workflow_ids
for id in copy:
    if st.session_state["workflows"][id]["status"] != "Completed":
        workflow_ids.remove(id)

st.title("Results")
st.subheader("Plots: ")

st.markdown("Data: Renders and returns a plot of the scaled train data." )
st.markdown("Scores: Plots the anomaly scores of the data.")
st.markdown("Confidence_variation: Plots the variation in confidence about the detection threshold for a sample.")
st.markdown("Threshold_variation: Change in detection threshold over test runs and train_set size.")
st.markdown("Model_stability: Plots the stability in confidence over train_size and required n_tests.")
st.markdown("Train_time:Plots the training time of the model over training set size.")

data, plots = st.columns(2)

results = st.radio("Choose results", [f"{id[:8]} ({st.session_state['workflows'][id]['start_time']})" for id in workflow_ids])

for i in workflow_ids:
    if f"{id[:8]} ({st.session_state['workflows'][id]['start_time']})" in results:
        wf = st.session_state["workflows"][i]
        ml = wf.get("ml", None)
        with data:
            if ml is not None:
                ft = ml.data.get("variables", None)
                ft.describe()
        
        available_plots = ["plot_data", "plot_scores", "plot_confidence_variation", "plot_threshold_variation", "plot_train_time", "plot_model_stability"]
        with plots:
            fig1, fig2 = st.columns(2)
            plot_types = st.radio("Choose plot type", [name.replace("_", " ").replace("plot_", "").capitalize() for name in available_plots])
            threshold = wf.get("threshold")
            for j, type in enumerate(plot_types):
                if j%2 == 0:
                    with fig1:
                        if type == "Plot data":
                            plot = ml.plot_data()
                        elif type == "Plot scores":
                            plot = ml.plot_scores(threshold=threshold)
                        elif type == "Plot confidence variation":
                            plot = ml.plot_confidence_variation()
                        elif type == "Plot threshold variation":
                            plot = ml.plot_threshold_variation()
                        elif type == "Plot train time":
                            plot = ml.plot_train_time()
                        elif type == "Plot model stability":
                            plot = ml.plot_model_stability()
                else:
                    with fig2:
                        if type == "Plot data":
                            plot = ml.plot_data()
                        elif type == "Plot scores":
                            plot = ml.plot_scores(threshold=threshold)
                        elif type == "Plot confidence variation":
                            plot = ml.plot_confidence_variation()
                        elif type == "Plot threshold variation":
                            plot = ml.plot_threshold_variation()
                        elif type == "Plot train time":
                            plot = ml.plot_train_time()
                        elif type == "Plot model stability":
                            plot = ml.plot_model_stability()
