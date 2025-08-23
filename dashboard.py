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
# Inject custom CSS for a professional, card-based layout with smaller fonts
st.markdown("""
<style>
    /* Main page layout */
    html, body, [class*="st-"] {
        font-size: 14px; /* Smaller base font size */
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 1.5rem;
    }
    /* Main container */
    .st-emotion-cache-1y4p8pa {
        padding-top: 0rem;
        max-width: 100%;
    }
    /* Sidebar styling */
    .st-emotion-cache-z5fcl4 {
        padding-top: 3rem;
    }
    /* Create bordered containers (cards) */
    .card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #444444;
        height: 100%; /* Make cards in columns equal height */
    }
    /* Center the graphviz chart */
    .stGraphVizChart {
        display: flex;
        justify-content: center;
    }
    /* Alert container styling */
    .alert-container {
        height: 450px; /* Adjust height to match video */
        overflow-y: auto;
        padding: 10px;
        border-radius: 5px;
        background-color: #111111;
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
    """Creates a Graphviz object from the current possessions data with a minimal color scheme."""
    # Hardcoded path for Graphviz on Windows as a fallback
    try:
        graphviz.backend.DOT_BINARY = r'C:\Program Files\Graphviz\bin\dot.exe'
    except Exception as e:
        print(f"Graphviz path not set, hoping it's in system PATH. Error: {e}")

    dot = graphviz.Digraph(comment='Human-Object Interaction Graph')
    dot.attr('graph', bgcolor='transparent', rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fontcolor='white', fontsize='12')
    dot.attr('edge', color='#CCCCCC', fontcolor='#CCCCCC', fontsize='10')

    all_nodes = set()
    for obj_id, info in object_info.items():
        label = f"{info['class'].capitalize()} {obj_id}\nState: {info['state']}"
        
        # Minimalist and professional color scheme
        color = '#555555' # Default grey
        border_color = '#CCCCCC' # Light grey border
        if info['state'] == 'possessed': 
            color = '#2E4053' # Dark blue
            border_color = '#5DADE2' # Light blue border
        if info['state'] == 'abandoned_candidate': 
            color = '#7E5109' # Dark orange
            border_color = '#F39C12' # Orange border
        if info['state'] == 'alert': 
            color = '#922B21' # Dark red
            border_color = '#E74C3C' # Red border

        dot.node(str(obj_id), label, fillcolor=color, color=border_color)
        all_nodes.add(obj_id)

    for bag_id, person_id in possessions_data.items():
        if person_id in all_nodes and bag_id in all_nodes:
            dot.edge(str(person_id), str(bag_id), label='[possesses]')
            
    return dot

# --- PAGE TITLE ---
st.title("Predictive Risk Synthesis (PRS) Dashboard")
st.text("An AI-powered surveillance analyst that infers intent from behavior.")
st.divider()

# --- MAIN LAYOUT (Video and Alerts side-by-side, Graph below) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Surveillance Feed")
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System Alerts Log")
    alerts_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Real-Time HOI Graph")
graph_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)


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

            graph_viz = create_graph_viz(graph_data)
            graph_placeholder.graphviz_chart(graph_viz, use_container_width=True)

            with alerts_placeholder.container():
                st.markdown('<div class="alert-container">', unsafe_allow_html=True)
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
                st.markdown("</div>", unsafe_allow_html=True)
            
        cap.release()
else:
    st.warning("Please select a video from the sidebar to begin analysis.")
