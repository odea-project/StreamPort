import streamlit as st
import time
from app_utils.workflow_state import shared_workflow_data, workflow_lock

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
filtered_workflows = {}
with workflow_lock:
    for wid, meta in st.session_state["workflows"].items():
        shared = shared_workflow_data.get(wid)
        if not shared:
            continue
        # check if any selected status in shared status string
        if any(status in shared.get("status", "") for status in selected_statuses):
            filtered_workflows[wid] = {"meta": meta, "shared": shared}

if not filtered_workflows:
    st.info("No workflows match the selected filter.")
    st.stop()

# Create a tab for each workflow
workflow_tabs = [f"{wf['meta']['name']} ({wid})" for wid, wf in filtered_workflows.items()]
workflow_keys = list(filtered_workflows.keys())
tabs = st.tabs(workflow_tabs)

for idx, tab in enumerate(tabs):
    wid = workflow_keys[idx]
    wf_meta = filtered_workflows[wid]["meta"]
    wf_shared = filtered_workflows[wid]["shared"]

    with tab:
        st.subheader(f"Workflow {wid} — {wf_shared['status']}")
        st.markdown(f"**Type:** {wf_meta['type']}")
        st.markdown(f"**Start Time:** {wf_meta.get('start_time', 'N/A')}")
        with st.expander("Details", expanded=False):
            st.dataframe(wf_meta.get("processor").parameters)
            st.markdown("**Scaler:** " + str(wf_meta.get("scaler", "N/A")))
            st.markdown("**Threshold:** " + str(wf_meta.get("threshold", "N/A")))
        # Progress bar if not completed
        if wf_shared["status"] != "Completed":
            st.progress(wf_shared.get("progress", 0))
            if st.button("Refresh results", key=f"refresh_{wid}"):
                st.info("Workflow is running... refreshing for updates.")
                time.sleep(1)
                st.rerun()

        if wf_shared["status"] == "Running":
            if st.button("Cancel Workflow", key=f"cancel_{wid}"):
                with workflow_lock:
                    shared_workflow_data[wid]["cancel"] = True
                    shared_workflow_data[wid]["status"] = "Cancelling..."
                st.warning(f"Cancel requested for workflow {wid}")
        
        results = wf_shared.get("results", {})
        if results:
            st.markdown("### Per-sample Test Results")
            sample_indices = sorted(results.keys())
            selected_index = st.selectbox(
                "Select test sample",
                sample_indices,
                key=f"index_select_{wid}"
            )
            
            sample_result = results.get(selected_index)
            if sample_result:
                sample_ml = sample_result.get("ml", None)
                if sample_ml and hasattr(sample_ml, "data") and sample_ml.data:
                    sample_ft = sample_ml.data.get("variables", None)
                    if sample_ft is not None and selected_index in sample_ft.index:
                        st.markdown("#### Sample Features")
                        st.dataframe(sample_ft.loc[[selected_index]].T)

                    st.markdown("#### Sample Score Plot")
                    score_plot = sample_ml.plot_scores(threshold=wf_meta.get("threshold"))
                    st.plotly_chart(score_plot)       

                else:
                    st.info("Selected sample result not available.")    
            
            else:
                st.info("No results found for selected sample index.")
        
        else:
            st.info("No sample results available yet.")

        # Show data summary and plots from latest ML model, if available
        ml = wf_shared.get("ml", None)
        if ml and hasattr(ml, "data") and ml.data is not None:
            data_col, plot_col = st.columns([1, 2])
            with data_col:
                st.markdown("### Data Summary")
                ft = ml.get_results(summarize=True)
                if ft is not None:
                    st.text_area("Summary of test results per index", ft, height = 500)
                else:
                    st.info("No feature data found.")

            available_plots = {
                "Data": ml.plot_data(),
                "Scores": ml.plot_scores(threshold=wf_meta.get("threshold")),
                "Confidence Variation": ml.plot_confidence_variation(),
                "Threshold Variation": ml.plot_threshold_variation(),
                "Train Time": ml.plot_train_time(),
                "Model Stability": ml.plot_model_stability(),
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
                    plot = plot_func
                    st.plotly_chart(plot)
                else:
                    st.warning("Plot function not available.")

        else:
            st.info("Workflow ML model data not available.")