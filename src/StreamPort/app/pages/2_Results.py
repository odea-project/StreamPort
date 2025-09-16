import streamlit as st

st.set_page_config(page_title="Anomaly Detection - Results", layout="wide")

if "workflows" not in st.session_state or not st.session_state["workflows"]:
    st.warning("No workflows running or completed yet.")
    st.stop()

st.title("Results")

# Create a tab for each workflow
workflow_tabs = []
workflow_keys = []
for wid, wf in st.session_state["workflows"].items():
    label = f"{wf['type']} ({wid})"
    workflow_tabs.append(label)
    workflow_keys.append(wid)

tabs = st.tabs(workflow_tabs)

for idx, tab in enumerate(tabs):
    wid = workflow_keys[idx]
    wf = st.session_state["workflows"][wid]
    ml = wf.get("ml", None)

    with tab:
        st.subheader(f"Workflow {wid} — {wf['status']}")
        st.markdown(f"**Type:** {wf['type']}")
        st.markdown(f"**Start Time:** {wf.get('start_time', 'N/A')}")

        # Progress bar if not completed
        if wf["status"] != "Completed":
            st.progress(wf["progress"])
            continue  # Skip rendering plots for incomplete workflows

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
