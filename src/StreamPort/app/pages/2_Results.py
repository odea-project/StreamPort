import streamlit as st
import time

st.set_page_config(page_title="Anomaly Detection - Results", layout="wide")

if "workflows" not in st.session_state or not st.session_state["workflows"]:
    st.warning("No workflows running or completed yet.")
    st.stop()

st.title("Results")

available_statuses = ["Running", "Completed", "Cancelled", "Failed"]

selected_statuses = st.multiselect(
    "Filter by status", available_statuses, default=["Completed", "Running"]
)

if not selected_statuses:
    selected_statuses = available_statuses  # Show all if none selected

# Filter workflows before building tabs
filtered_workflows = {
    wid: wf for wid, wf in st.session_state["workflows"].items()
    if any(s in wf["status"] for s in selected_statuses)
}

if not filtered_workflows:
    st.info("No workflows match the selected filter.")
    st.stop()

# Create a tab for each workflow
workflow_tabs = []
workflow_keys = []
for wid, wf in filtered_workflows.items():
    label = f"{wf['name']} ({wid})"
    workflow_tabs.append(label)
    workflow_keys.append(wid)

tabs = st.tabs(workflow_tabs)

for idx, tab in enumerate(tabs):
    wid = workflow_keys[idx]
    wf = st.session_state["workflows"][wid]
    ml = wf.get("ml", None)
    results = st.session_state.get("results", {}).get(wid, {})

    with tab:
        st.subheader(f"Workflow {wid} — {wf['status']}")
        st.markdown(f"**Type:** {wf['type']}")
        st.markdown(f"**Start Time:** {wf.get('start_time', 'N/A')}")

        # Progress bar if not completed
        if wf["status"] != "Completed":
            st.progress(wf["progress"])
            if st.button("Refresh results"):
                st.info("Workflow is running... refreshing for updates.")
                time.sleep(3)
                st.rerun()

        if wf["status"] == "Running":
            if st.button("Cancel Workflow", key=f"cancel_{wid}"):
                wf["cancel"] = True
                st.session_state["workflows"][wid] = wf
                st.warning(f"Cancel requested for workflow {wid}")
        
        if results:
            st.markdown("### Per-sample Test Results")
            sample_indices = sorted(results.keys())
            selected_indices = st.radio(
                "Select test sample index",
                sample_indices,
                key=f"index_select_{wid}"
            )

            for idx in selected_indices:
                sample_result = results[idx]
                sample_ml = sample_result.get("ml", None)
                
                if sample_ml and sample_ml.data:
                    sample_ft = sample_ml.data.get("variables", None)
                    if sample_ft is not None:
                        st.markdown("#### Sample Features")
                        st.dataframe(sample_ft.loc[[idx]].T)
                    
                    st.markdown("#### Sample Score Plot")
                    score_plot = sample_ml.plot_scores(indices=[idx], threshold=wf.get("threshold"))
                    st.plotly_chart(score_plot)
                else:
                    st.info("Selected sample result not available.")
            
        # Display data summary
        if ml and ml.data:
            data_col, plot_col = st.columns([1, 2])

            with data_col:
                st.markdown("### Data Summary")
                ft = ml.data.get("variables", None)
                if ft is not None:
                    st.dataframe(ft.describe())
                else:
                    st.info("No feature data found.")

            # Plot selection
            available_plots = {
                "Data": ml.plot_data,
                "Scores": lambda: ml.plot_scores(threshold=wf.get("threshold")),
                "Confidence Variation": ml.plot_confidence_variation,
                "Threshold Variation": ml.plot_threshold_variation,
                "Train Time": ml.plot_train_time,
                "Model Stability": ml.plot_model_stability,
            }

            with plot_col:
                st.markdown("### Plots")
                plot_choice = st.selectbox(
                    "Select a plot to view",
                    list(available_plots.keys()),
                    key=f"plot_select_{wid}"
                )

                plot_func = available_plots.get(plot_choice)
                if plot_func:
                    plot = plot_func()
                    st.pyplot(plot)
                else:
                    st.warning("Plot function not available.")

        else:
            st.info("Workflow data not available.")
