# main.py

import cv2
import torch
import mediapipe as mp
import numpy as np
import requests
from tracker import CentroidTracker
from scipy.spatial import distance as dist
from datetime import datetime
from collections import deque

# --- DEBUG SWITCH ---
# Set this to True to see live data on the video feed for tuning
DEBUG_MODE = True

# --- INITIALIZATION ---
print("Loading models and setting up...")

# 1. YOLOv5 Model
model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)
model.classes = [0, 24, 26, 28] # person, backpack, handbag, suitcase
model.conf = 0.4

# 2. MediaPipe Pose Model
# static_image_mode=True tells MediaPipe to treat each image independently, preventing timestamp errors.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. Centroid Tracker
ct = CentroidTracker(maxDisappeared=40)

# --- DATA STRUCTURES ---
object_info = {} 
alerts = []
sent_alerts = set() # To track which alerts have been sent to Discord
flow_vectors = deque(maxlen=100) # For wrong direction detection

# --- CONSTANTS (TUNE THESE VALUES) ---
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1409070598759518238/AMaSBr8jPxdItsQkfd0OFK6r_216TLgk4S4LCMkLdzvTJouSkJU_r6wQWiVdBGNK31SO"
POSSESSION_THRESHOLD = 150 
CROUCH_FRAMES_THRESHOLD = 10
ABANDON_FRAMES_THRESHOLD = 100
VELOCITY_THRESHOLD = 15
RUNNING_VELOCITY_THRESHOLD = 30
LOITER_FRAMES_THRESHOLD = 150
LOITER_DISTANCE_THRESHOLD = 50

# --- VIDEO SETUP ---
VIDEO_PATH = 'official_videos/test_02_fixed.mp4' 

def reset_engine_states():
    """Clears all the stateful variables for a new video run."""
    global object_info, alerts, sent_alerts, flow_vectors, ct
    object_info.clear()
    alerts.clear()
    sent_alerts.clear()
    flow_vectors.clear()
    # Re-initialize the CentroidTracker to reset its internal state
    ct = CentroidTracker(maxDisappeared=40)


def process_frame(frame):
    global object_info, alerts, flow_vectors, sent_alerts

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
                if len(object_info[objectID]['positions']) > 1:
                    dx = centroid[0] - object_info[objectID]['positions'][-1][0]
                    dy = centroid[1] - object_info[objectID]['positions'][-1][1]
                    object_info[objectID]['velocity'] = (dx**2 + dy**2)**0.5
                    if object_info[objectID]['class'] == 'person' and object_info[objectID]['velocity'] > 1:
                        flow_vectors.append((dx, dy))
                
                object_info[objectID]['positions'].append(centroid)
                object_info[objectID]['box'] = best_match_box
                object_info[objectID]['centroid'] = centroid

    persons = {oid: info for oid, info in object_info.items() if info['class'] == 'person'}
    bags = {oid: info for oid, info in object_info.items() if info['class'] in ['backpack', 'handbag', 'suitcase']}

    # --- BEHAVIORAL ANALYSIS ---
    dominant_flow = (0,0)
    if len(flow_vectors) > 20:
        dominant_flow = np.mean(flow_vectors, axis=0)

    for person_id, person_data in persons.items():
        if person_data['state'] in ['running', 'wrong_direction']:
            person_data['state'] = 'normal'

        if 'box' not in person_data: continue
        p_box = person_data['box']
        person_crop = frame[p_box[1]:p_box[3], p_box[0]:p_box[2]]
        if person_crop.size == 0: continue
        
        person_crop.flags.writeable = False
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(person_crop_rgb)
        person_crop.flags.writeable = True

        if pose_results.pose_landmarks:
            landmarks_to_draw = pose_results.pose_landmarks
            for landmark in landmarks_to_draw.landmark:
                landmark.x = p_box[0] + landmark.x * (p_box[2] - p_box[0])
                landmark.y = p_box[1] + landmark.y * (p_box[3] - p_box[1])
            mp_drawing.draw_landmarks(frame, landmarks_to_draw, mp_pose.POSE_CONNECTIONS)
            
            shoulder_y = (landmarks_to_draw.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks_to_draw.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_y = (landmarks_to_draw.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks_to_draw.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            if shoulder_y > hip_y - 20:
                person_data['timers']['crouch'] = min(CROUCH_FRAMES_THRESHOLD + 5, person_data['timers']['crouch'] + 1)
            else:
                 person_data['timers']['crouch'] = max(0, person_data['timers']['crouch'] - 1)
        
        positions = person_data['positions']
        if len(positions) == 20:
            max_dist = dist.pdist(np.array(positions)).max()
            if max_dist < LOITER_DISTANCE_THRESHOLD:
                person_data['timers']['loiter'] += 1
            else:
                person_data['timers']['loiter'] = 0
        
        if person_data['velocity'] > RUNNING_VELOCITY_THRESHOLD:
            person_data['state'] = 'running'
        
        if len(person_data['positions']) > 10 and np.linalg.norm(dominant_flow) > 0.5:
            person_vector = (person_data['positions'][-1][0] - person_data['positions'][0][0], person_data['positions'][-1][1] - person_data['positions'][0][1])
            if np.linalg.norm(person_vector) > 1:
                cos_similarity = np.dot(person_vector, dominant_flow) / (np.linalg.norm(person_vector) * np.linalg.norm(dominant_flow))
                if cos_similarity < -0.5:
                    person_data['timers']['wrong_dir'] += 1
                else:
                    person_data['timers']['wrong_dir'] = 0
            
            if person_data['timers']['wrong_dir'] > 30:
                person_data['state'] = 'wrong_direction'
                alert_msg = f"Person ID {person_id} is moving against the crowd flow."
                if not any(f"Person ID {person_id}" in s for s in alerts):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alerts.append(f"[{timestamp}] HIGH ALERT: {alert_msg}")

    # --- HOI GRAPH & CAUSAL CHAIN LOGIC ---
    current_possessions = {}
    for bag in bags.values(): bag['debug_dist'] = -1

    for person_id, person_data in persons.items():
        closest_bag_id, min_dist = -1, float('inf')
        for bag_id, bag_data in bags.items():
            if bag_data['state'] == 'normal':
                d = dist.euclidean(person_data['centroid'], bag_data['centroid'])
                if d < min_dist:
                    min_dist = d
                    closest_bag_id = bag_id
        
        if min_dist < POSSESSION_THRESHOLD:
            if closest_bag_id != -1: bags[closest_bag_id]['debug_dist'] = min_dist
            if closest_bag_id not in current_possessions or min_dist < dist.euclidean(persons[current_possessions[closest_bag_id]]['centroid'], bags[closest_bag_id]['centroid']):
                 current_possessions[closest_bag_id] = person_id

    for bag_id, person_id in current_possessions.items():
        object_info[bag_id]['state'] = 'possessed'
        if object_info[person_id]['timers']['crouch'] >= CROUCH_FRAMES_THRESHOLD:
            object_info[bag_id]['last_owner'] = person_id
        if object_info[bag_id]['velocity'] > VELOCITY_THRESHOLD:
            object_info[bag_id]['last_owner'] = person_id
            object_info[bag_id]['thrown'] = True

    for bag_id, bag_data in bags.items():
        if bag_id not in object_info: continue
        is_possessed = bag_id in current_possessions
        if not is_possessed and bag_data['last_owner'] != -1:
            bag_data['state'] = 'abandoned_candidate'
            bag_data['timers']['abandon'] += 1
        elif not is_possessed:
            if bag_data['state'] != 'alert': bag_data['state'] = 'normal'
            bag_data['timers']['abandon'] = 0

        if bag_data['timers']['abandon'] > ABANDON_FRAMES_THRESHOLD:
            owner_id = bag_data['last_owner']
            narrative = "[Possession -> Suspicious Throw -> Abandon]" if bag_data['thrown'] else "[Possession -> Crouch -> Abandon]"
            alert_msg = f"Bag ID {bag_id} abandoned by Person ID {owner_id}. Causal Chain: {narrative}"
            if not any(f"Bag ID {bag_id}" in s for s in alerts):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alerts.append(f"[{timestamp}] CRITICAL ALERT: {alert_msg}")
            bag_data['state'] = 'alert'
            bag_data['timers']['abandon'] = 0

        if bag_data['state'] == 'abandoned_candidate':
            for person_id, person_data in persons.items():
                if person_data['timers']['loiter'] > LOITER_FRAMES_THRESHOLD:
                    if dist.euclidean(person_data['centroid'], bag_data['centroid']) < POSSESSION_THRESHOLD * 1.5:
                        alert_msg = f"Person ID {person_id} is loitering near abandoned candidate Bag ID {bag_id}."
                        if not any(alert_msg in s for s in alerts):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            alerts.append(f"[{timestamp}] HIGH ALERT: {alert_msg}")
                        bag_data['state'] = 'alert'
                        person_data['timers']['loiter'] = 0

    # --- DRAWING ---
    for bag_id, person_id in current_possessions.items():
        if person_id in persons and bag_id in bags:
            p_center = object_info[person_id]['centroid']
            b_center = object_info[bag_id]['centroid']
            cv2.line(frame, tuple(p_center), tuple(b_center), (255, 255, 0), 2)

    for oid, data in object_info.items():
        centroid = data['centroid']
        state = data['state']
        color = (0, 255, 0)
        if state == 'possessed': color = (255, 255, 0)
        if state == 'abandoned_candidate': color = (0, 165, 255)
        if state == 'alert': color = (0, 0, 255)
        if state == 'running': color = (255, 100, 0)
        if state == 'wrong_direction': color = (255, 0, 255)
        
        label = f"{data['class'].capitalize()} {oid} [{state}]"
        cv2.putText(frame, label, (centroid[0] - 20, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

        if DEBUG_MODE:
            debug_text = f"V: {data['velocity']:.1f}"
            if data['class'] in ['backpack', 'handbag', 'suitcase'] and data['debug_dist'] != -1:
                debug_text += f" D: {data['debug_dist']:.0f}"
            cv2.putText(frame, debug_text, (centroid[0] - 20, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- CENTRALIZED DISCORD ALERT LOGIC ---
    if len(alerts) > 0 and alerts[-1] not in sent_alerts:
        new_alert = alerts[-1]
        sent_alerts.add(new_alert)

        if "CRITICAL" in new_alert:
            message_header = "🚨 **CRITICAL ALERT** 🚨"
        elif "HIGH" in new_alert:
            message_header = "⚠️ **HIGH ALERT** ⚠️"
        else:
            message_header = "ℹ️ INFO ℹ️"

        if DISCORD_WEBHOOK_URL != "YOUR_WEBHOOK_URL_HERE":
            payload = { "content": f"{message_header}\n**Details:** {new_alert}" }
            try:
                requests.post(DISCORD_WEBHOOK_URL, json=payload)
            except Exception as e:
                print(f"Error sending Discord alert: {e}")

    return frame, alerts, current_possessions