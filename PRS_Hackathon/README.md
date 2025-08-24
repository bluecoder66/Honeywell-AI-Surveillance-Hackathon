# 🚨 Predictive Risk Synthesis (PRS) - Honeywell Hackathon

**A next-generation AI surveillance system that infers intent from human behavior.**

---

## 🚀 The Idea: Beyond Anomaly Detection

Traditional surveillance systems are reactive; they detect events that have already happened. Our **Predictive Risk Synthesis (PRS)** system is proactive. It functions as an AI-powered security analyst, using a **Causal Inference Engine** to understand the *'why'* behind an action, not just the *'what'*.

Instead of training a model on a limited dataset, we built a flexible expert system that understands universal behavioral primitives (like crouching, loitering, and suspicious throws). This provides **Explainable AI (XAI)**, a critical feature for real-world security operations.

**[Link to Your 2-Minute Demo Video Here]**

---

## ✨ Key Features

* **Causal Inference Engine:** Detects multi-step "Threat Narratives" (e.g., `[Possession -> Crouch -> Abandon]`).
* **Explainable Alerts (XAI):** Provides clear, human-readable reasons for every critical alert.
* **Real-Time HOI Graph:** Visually maps the dynamic relationships between people and objects in the scene.
* **Multi-Behavior Analysis:** Detects object abandonment, loitering, suspicious throws, and running.
* **Professional Dashboard:** A clean, intuitive UI built with Streamlit for real-time monitoring.

---

## 🛠️ Technical Approach

* **Language:** Python 3
* **Core Libraries:** OpenCV, PyTorch, Streamlit, MediaPipe, Graphviz
* **Object Detection:** Pre-trained YOLOv5s for high-performance person and object identification.
* **Tracking:** A custom Centroid Tracker for object permanence across frames.
* **Behavioral Analysis:** Google's MediaPipe Pose for human keypoint detection and our custom logic for velocity and loitering analysis.

![Screenshot of the PRS Dashboard in Action](link_to_your_screenshot.jpg)

---

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    ```
2.  **Set up the environment:**
    ```bash
    cd [your-repo-name]
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will also need to install Graphviz on your system and add it to your PATH.)*

4.  **Launch the dashboard:**
    ```bash
    streamlit run dashboard.py
    ```
