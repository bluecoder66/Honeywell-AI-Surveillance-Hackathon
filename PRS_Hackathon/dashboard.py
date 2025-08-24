# dashboard.py

import streamlit as st
import cv2
import graphviz
import os
from main import process_frame, object_info # Import our engine

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PRS Dashboard | Honeywell Hackathon",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
# Inject custom CSS for a more professional look
st.markdown("""
<style>
    /* Main page layout */
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    /* Main container */
    .st-emotion-cache-1y4p8pa {
        padding-top: 0rem;
        max-width: 100%;
    }
    /* Sidebar styling */
    .st-emotion-cache-z5fcl4 {
        padding-top: 4rem;
    }
    /* Video and Graph containers */
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR ---
with st.sidebar:
    st.title("PRS Control Panel")
    st.info("Predictive Risk Synthesis Engine")
    
    st.header("Video Source")
    video_folder = 'official_videos'
    try:
        # List only the fixed mp4 files
        video_files = [f for f in os.listdir(video_folder) if f.endswith('_fixed.mp4')]
        if not video_files:
            st.warning("No '_fixed.mp4' videos found in the 'official_videos' folder. Please run ffmpeg conversion.")
            selected_video = None
        else:
            selected_video = st.selectbox("Select a video to analyze:", video_files)
            VIDEO_PATH = os.path.join(video_folder, selected_video)
    except FileNotFoundError:
        st.error(f"Folder '{video_folder}' not found. Please create it and add your fixed videos.")
        VIDEO_PATH = None

    st.header("About")
    st.markdown("""
    This dashboard demonstrates the **Predictive Risk Synthesis (PRS)** system, an AI-powered surveillance analyst.
    
    Instead of simple anomaly detection, PRS uses a **Causal Inference Engine** to understand behavior and infer intent, providing **Explainable AI (XAI)** alerts.
    """)

# --- HELPER FUNCTION FOR GRAPH ---
def create_graph_viz(possessions_data):
    """Creates a Graphviz object from the current possessions data."""
    # Hardcoded path for Graphviz on Windows as a fallback
    try:
        graphviz.backend.DOT_BINARY = r'C:\Program Files\Graphviz\bin\dot.exe'
    except Exception as e:
        print(f"Graphviz path not set, hoping it's in system PATH. Error: {e}")

    dot = graphviz.Digraph(comment='Human-Object Interaction Graph')
    dot.attr('graph', bgcolor='transparent', rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#2E2E2E', fontcolor='white', color='#00FF00')
    dot.attr('edge', color='white', fontcolor='white')

    all_nodes = set()
    for obj_id, info in object_info.items():
        label = f"{info['class'].capitalize()} {obj_id}\nState: {info['state']}"
        dot.node(str(obj_id), label)
        all_nodes.add(obj_id)

    for bag_id, person_id in possessions_data.items():
        if person_id in all_nodes and bag_id in all_nodes:
            dot.edge(str(person_id), str(bag_id), label='[possesses]')
            
    return dot

# --- PAGE TITLE & METRICS ---
st.title("Predictive Risk Synthesis (PRS) Dashboard")
st.text("An AI-powered surveillance analyst that infers intent from behavior.")

metric1, metric2, metric3 = st.columns(3)
tracked_objects_metric = metric1.empty()
persons_metric = metric2.empty()
alerts_metric = metric3.empty()

st.divider()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Surveillance Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-Time HOI Graph")
    graph_placeholder = st.empty()
    
    st.subheader("System Alerts Log")
    alerts_placeholder = st.empty()


# --- VIDEO PROCESSING ---
if 'selected_video' in locals() and selected_video:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {VIDEO_PATH}")
    else:
        all_alerts = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("Video analysis complete.")
                break

            annotated_frame, new_alerts, graph_data = process_frame(frame)

            if new_alerts:
                all_alerts.extend(a for a in new_alerts if a not in all_alerts)

            # --- UPDATE UI ELEMENTS ---
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

            persons_count = sum(1 for info in object_info.values() if info['class'] == 'person')
            tracked_objects_metric.metric("Total Tracked Objects", len(object_info))
            persons_metric.metric("Persons Detected", persons_count)
            alerts_metric.metric("Critical Alerts", len([a for a in all_alerts if "CRITICAL" in a]))

            graph_viz = create_graph_viz(graph_data)
            graph_placeholder.graphviz_chart(graph_viz)

            with alerts_placeholder.container():
                if not all_alerts:
                    st.info("System is nominal. No threats detected.")
                else:
                    for alert in reversed(all_alerts):
                        if "CRITICAL" in alert:
                            st.error(alert)
                        elif "HIGH" in alert:
                            st.warning(alert)
                        else:
                            st.info(alert)
            
        cap.release()
else:
    st.warning("Please select a video from the sidebar to begin analysis.")