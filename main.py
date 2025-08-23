# main.py

import cv2
import torch
import mediapipe as mp
import numpy as np
from tracker import CentroidTracker
from scipy.spatial import distance as dist
from datetime import datetime
from collections import deque

# --- DEBUG SWITCH ---
DEBUG_MODE = True

# --- INITIALIZATION ---
print("Loading models and setting up...")

# 1. YOLOv5 Model
model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)
model.classes = [0, 24, 26, 28] # person, backpack, handbag, suitcase
model.conf = 0.25

# 2. MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. Centroid Tracker
ct = CentroidTracker(maxDisappeared=40)

# --- DATA STRUCTURES ---
object_info = {} 
alerts = []
# New: Track overall crowd movement
flow_vectors = deque(maxlen=100) 

# --- CONSTANTS (TUNE THESE VALUES) ---
POSSESSION_THRESHOLD = 150 
CROUCH_FRAMES_THRESHOLD = 10
ABANDON_FRAMES_THRESHOLD = 100
VELOCITY_THRESHOLD = 15
RUNNING_VELOCITY_THRESHOLD = 30 # New: Higher threshold for running
LOITER_FRAMES_THRESHOLD = 150
LOITER_DISTANCE_THRESHOLD = 50

# --- VIDEO SETUP ---
VIDEO_PATH = 'official_videos/test_02_fixed.mp4' 

def process_frame(frame):
    global object_info, alerts, flow_vectors

    # --- DETECTION & TRACKING ---
    results = model(frame)
    detections = results.pandas().xyxy[0]
    
    rects = []
    temp_object_classes = {} 
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        box = (x1, y1, x2, y2)
        rects.append(box)
        temp_object_classes[box] = row['name']
        
    objects = ct.update(rects)

    # --- UPDATE OBJECT INFO DICTIONARY ---
    current_object_ids = set(objects.keys())
    for object_id in list(object_info.keys()):
        if object_id not in current_object_ids:
            del object_info[object_id]

    for (objectID, centroid) in objects.items():
        if rects: 
            best_match_box = min(rects, key=lambda r: dist.euclidean(centroid, ( (r[0]+r[2])/2, (r[1]+r[3])/2 ) ))
            
            if objectID not in object_info:
                 object_info[objectID] = {
                     'class': temp_object_classes.get(best_match_box, 'unknown'),
                     'box': best_match_box, 'centroid': centroid,
                     'state': 'normal', 'timers': {'crouch': 0, 'abandon': 0, 'loiter': 0, 'wrong_dir': 0},
                     'last_owner': -1, 'positions': deque(maxlen=20),
                     'velocity': 0, 'thrown': False, 'debug_dist': -1
                 }
            else:
                # Calculate velocity and update crowd flow
                if len(object_info[objectID]['positions']) > 2:
                    dx = centroid[0] - object_info[objectID]['positions'][-2][0]
                    dy = centroid[1] - object_info[objectID]['positions'][-2][1]
                    object_info[objectID]['velocity'] = (dx**2 + dy**2)**0.5
                    # Add to crowd flow if it's a person moving normally
                    if object_info[objectID]['class'] == 'person' and object_info[objectID]['velocity'] > 1:
                        flow_vectors.append((dx, dy))

                object_info[objectID]['positions'].append(centroid)
                object_info[objectID]['box'] = best_match_box
                object_info[objectID]['centroid'] = centroid

    persons = {oid: info for oid, info in object_info.items() if info['class'] == 'person'}
    bags = {oid: info for oid, info in object_info.items() if info['class'] in ['backpack', 'handbag', 'suitcase']}

    # --- BEHAVIORAL ANALYSIS ---
    # Calculate dominant flow
    dominant_flow = (0,0)
    if len(flow_vectors) > 10:
        dominant_flow = np.mean(flow_vectors, axis=0)

    for person_id, person_data in persons.items():
        # Pose Estimation (Crouch)
        # ... (same as before)

        # Running Detection
        if person_data['velocity'] > RUNNING_VELOCITY_THRESHOLD:
            person_data['state'] = 'running'
        elif person_data['state'] == 'running': # Reset state if they stop running
            person_data['state'] = 'normal'

        # Wrong Direction Detection
        if len(person_data['positions']) > 10:
            person_vector = (person_data['positions'][-1][0] - person_data['positions'][0][0],
                             person_data['positions'][-1][1] - person_data['positions'][0][1])
            # Cosine similarity to check if moving against the flow
            dot_product = np.dot(person_vector, dominant_flow)
            if dot_product < -0.5: # Moving in opposite direction
                person_data['timers']['wrong_dir'] += 1
            else:
                person_data['timers']['wrong_dir'] = 0
            
            if person_data['timers']['wrong_dir'] > 30: # If moving wrong way for 30 frames
                person_data['state'] = 'wrong_direction'
                alert_msg = f"Person ID {person_id} is moving against the crowd flow."
                if not any(f"Person ID {person_id}" in s for s in alerts):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alerts.append(f"[{timestamp}] HIGH ALERT: {alert_msg}")

    # --- HOI GRAPH & ABANDONMENT LOGIC ---
    # ... (same as before)

    # --- DRAWING ---
    for oid, data in object_info.items():
        centroid = data['centroid']
        state = data['state']
        color = (0, 255, 0) # Normal
        if state == 'possessed': color = (255, 255, 0)
        if state == 'abandoned_candidate': color = (0, 165, 255)
        if state == 'alert': color = (0, 0, 255)
        if state == 'running': color = (255, 100, 0) # New color for running
        if state == 'wrong_direction': color = (255, 0, 255) # New color for wrong direction
        
        label = f"{data['class'].capitalize()} {oid} [{state}]"
        cv2.putText(frame, label, (centroid[0] - 20, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

        if DEBUG_MODE:
            debug_text = f"V: {data['velocity']:.1f}"
            if data['class'] in ['backpack', 'handbag', 'suitcase'] and data['debug_dist'] != -1:
                debug_text += f" D: {data['debug_dist']:.0f}"
            cv2.putText(frame, debug_text, (centroid[0] - 20, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, alerts, {} # Return empty dict for graph for now
