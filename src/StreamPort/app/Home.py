import streamlit as st
import base64
from app_utils.workflow_state import workflow_lock, shared_workflow_data

st.set_page_config(page_title="StreamPort", layout="wide")

## HomePage Illustrations ##
def read_local_gif(path):
    """### gif from local file"""
    file_ = open(path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode(encoding='ascii')
    file_.close()
    return data_url

###---------------------------------------------------------------------------------------------------

## Known Paths ##
#C:/Users/Sandeep/Desktop/ExtractedSignals
#C:/Users/Sandeep/Desktop/Error-LC/Method-Data

###---------------------------------------------------------------------------------------------------

# Initialize GIFs
# if ["conf_gif", "thresh_gif", "scores_gif", "curves_gif"] not in st.session_state:
#     conf_gif = read_local_gif("src/StreamPort/confidence.gif")
#     st.session_state["conf_gif"] = conf_gif

#     thresh_gif = read_local_gif("src/StreamPort/thresholds.gif")
#     st.session_state["thresh_gif"] = thresh_gif
    
#     scores_gif = read_local_gif("src/StreamPort/scores.gif")
#     st.session_state["scores_gif"] = scores_gif
    
#     curves_gif = read_local_gif("src/StreamPort/curves.gif")
#     st.session_state["curves_gif"] = curves_gif

# conf_gif = st.session_state["conf_gif"]
# thresh_gif = st.session_state["thresh_gif"]
# scores_gif = st.session_state["scores_gif"]
# curves_gif = st.session_state["curves_gif"]

# Start
st.markdown("<h1 style='text-align: center;'>StreamPort</h1>", unsafe_allow_html=True)

st.divider()

st.markdown("### This is a prototype of the StreamPort tool for Anomaly Detection in LCMS instrument data.")

# pc_col, score_col = st.columns(2)

# with pc_col:
#     st.markdown(
#         f'<img src="data:image/gif;base64,{curves_gif}" alt="curves_gif">',
#         unsafe_allow_html=True,
#     )
#     # st.text("")
#     # st.markdown(
#     #     f'<img src="data:image/gif;base64,{conf_gif}" alt="confidence_gif">',
#     #     unsafe_allow_html=True,
#     # )

# with score_col:
#     st.markdown(
#         f'<img src="data:image/gif;base64,{scores_gif}" alt="scores_gif">',
#         unsafe_allow_html=True,
#     )
#     # st.text("")
#     # st.markdown(
#     #     f'<img src="data:image/gif;base64,{thresh_gif}" alt="threshold_gif">',
#     #     unsafe_allow_html=True,
#     # )

st.text("")
st.markdown("#### Create a new workflow to begin.")
pipeline = st.selectbox("New Workflow", ["--", "Anomaly Detection"])
if pipeline == "Anomaly Detection":
    st.switch_page("pages/1_Session.py")

if st.session_state.get("workflows", None):
    st.subheader("Currently running workflows:")

    available_statuses = ["Running", "Completed", "Cancelled", "Failed"]
    selected_statuses = st.multiselect("Show workflows with status:", available_statuses, default=["Running", "Completed"])

    if not selected_statuses:
        selected_statuses = available_statuses  # Show all if none selected

    filtered_workflows = {}

    with workflow_lock:
        for wid, meta in st.session_state["workflows"].items():
            shared = shared_workflow_data.get(wid)
            if not shared:
                continue
            if any(status in shared.get("status", "") for status in selected_statuses):
                filtered_workflows[wid] = {"meta": meta, "shared": shared}

    if not filtered_workflows:
        st.info("No workflows match your selected filter.")
    else:
        for wid, wf in filtered_workflows.items():
            meta = wf["meta"]
            shared = wf["shared"]

            with st.expander(f"{meta['name']} — {shared['status']}"):
                st.write(f"**Type:** {meta['type']}")
                st.write(f"**Start Time:** {meta.get('start_time', 'N/A')}")
                st.progress(shared.get("progress", 0))

                # Cancel button
                if shared["status"] == "Running":
                    if st.button("Cancel Workflow", key=f"cancel_{wid}"):
                        with workflow_lock:
                            shared_workflow_data[wid]["cancel"] = True
                            shared_workflow_data[wid]["status"] = "Cancelling..."  # Optional: interim label
                        st.warning(f"Workflow {wid} is being cancelled. Refresh to see updates.")

                # Remove workflow 
                if shared["status"] in ["Cancelled", "Failed", "Completed"]:
                    if st.button("Remove Workflow", key=f"remove_{wid}"):
                        with workflow_lock:
                            shared_workflow_data.pop(wid, None)
                        st.session_state["workflows"].pop(wid, None)
                        st.success(f"Workflow {wid} removed.")
                        st.rerun()


