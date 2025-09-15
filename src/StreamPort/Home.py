import streamlit as st
import base64

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

# UI - Start
st.markdown("<h1 style='text-align: center;'>StreamPort</h1>", unsafe_allow_html=True)

st.divider()

st.markdown("### This is a prototype of the StreamPort tool for Anomaly Detection in LCMS instrument data.")

conf_gif = read_local_gif("dev/figures/gif_figs/confidence.gif")
thresh_gif = read_local_gif("dev/figures/gif_figs/thresholds.gif")
scores_gif = read_local_gif("dev/figures/gif_figs/scores.gif")
curves_gif = read_local_gif("dev/figures/gif_figs/curves.gif")

pc_col, score_col = st.columns(2)

with pc_col:
    st.markdown(
        f'<img src="data:image/gif;base64,{curves_gif}" alt="curves_gif">',
        unsafe_allow_html=True,
    )
    st.text("")
    st.markdown(
        f'<img src="data:image/gif;base64,{conf_gif}" alt="confidence_gif">',
        unsafe_allow_html=True,
    )

with score_col:
    st.markdown(
        f'<img src="data:image/gif;base64,{scores_gif}" alt="scores_gif">',
        unsafe_allow_html=True,
    )
    st.text("")
    st.markdown(
        f'<img src="data:image/gif;base64,{thresh_gif}" alt="threshold_gif">',
        unsafe_allow_html=True,
    )

st.text("")
st.markdown("#### Create a new workflow to begin.")
pipeline = st.selectbox("New Workflow", ["--", "Anomaly Detection"])
if pipeline == "Anomaly Detection":
    st.switch_page("pages/1_Session.py")

if st.session_state.get("workflows", None):
    st.subheader("Currently running Workflows")
    for wid, wf in st.session_state["workflows"].items():
        with st.expander(f"{wf['type']} Workflow {wid[:8]} — {wf['status']}"):
            st.progress(wf["progress"])
            st.write(f"Type: {wf['type']}")
            if st.button(f"Cancel Workflow {wid[:8]}", key=f"cancel_{wid}"):
                wf["cancel"] = True
                st.session_state["workflows"][wid] = wf
                st.warning("Cancel requested.")

        if wf["status"] in ["Completed", "Cancelled", "Failed"]:
            if st.button(f"Remove Workflow {wid[:8]}", key=f"remove_{wid}"):
                del st.session_state["workflows"][wid]

    
    


