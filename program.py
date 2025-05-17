import cv2
import numpy as np
import math
import time
from collections import deque
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import tensorflow as tf
import os
from threading import Thread
from queue import Queue
import platform
import logging

class GestureFlow:
    def __init__(self):
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps_target = 30
        
        self.use_mediapipe = True
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.5
        
        self.smoothing_factor = 0.7
        self.feature_scaling = 1.0
        self.detection_threshold = 3000
        self.k_value = 20
        self.parallel_processing = True
        
        self.camera = None
        self.roi = None
        self.light_condition = "auto"
        self.display_mode = "standard"
        self.hand_detected_time = 0
        self.last_time_check = time.time()
        self.hand_present = False
        self.adaptive_mode = True
        
        self.fps_history = deque(maxlen=30)
        self.fingertips_history = deque(maxlen=15)
        self.palm_history = deque(maxlen=10)
        self.motion_trails = {}
        self.hand_orientation_history = deque(maxlen=10)
        
        self.processing_times = {
            'skin_detection': deque(maxlen=30),
            'contour_finding': deque(maxlen=30),
            'fingertip_detection': deque(maxlen=30),
            'rendering': deque(maxlen=30),
            'total': deque(maxlen=30)
        }
        
        self._init_mediapipe()
        self._configure_logging()

    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def _configure_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='hand_tracking.log'
        )
        self.logger = logging.getLogger('GestureFlow')

    def initialize_camera(self):
        camera_opened = False
        
        for method in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]:
            try:
                self.camera = cv2.VideoCapture(self.camera_index, method)
                if self.camera.isOpened():
                    camera_opened = True
                    break
            except:
                continue
        
        if not camera_opened:
            for idx in range(1, 5):
                try:
                    self.camera = cv2.VideoCapture(idx)
                    if self.camera.isOpened():
                        self.camera_index = idx
                        camera_opened = True
                        break
                except:
                    continue
        
        if not camera_opened:
            self.logger.error("Failed to access any camera")
            raise RuntimeError("Could not access any camera")
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True

    def detect_skin(self, frame):
        start_time = time.time()
        
        if self.adaptive_mode and self.light_condition == "auto":
            avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if avg_brightness < 80:
                effective_light = "dark"
            elif avg_brightness > 180:
                effective_light = "bright"
            else:
                effective_light = "normal"
        else:
            effective_light = self.light_condition
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        
        if effective_light == "bright":
            lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
            lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            lower_skin_lab = np.array([20, 130, 120], dtype=np.uint8)
            upper_skin_lab = np.array([220, 170, 150], dtype=np.uint8)
        elif effective_light == "dark":
            lower_skin_hsv = np.array([0, 10, 50], dtype=np.uint8)
            upper_skin_hsv = np.array([30, 255, 255], dtype=np.uint8)
            lower_skin_ycrcb = np.array([0, 125, 65], dtype=np.uint8)
            upper_skin_ycrcb = np.array([255, 190, 145], dtype=np.uint8)
            lower_skin_lab = np.array([10, 120, 110], dtype=np.uint8)
            upper_skin_lab = np.array([230, 180, 160], dtype=np.uint8)
        else:
            lower_skin_hsv = np.array([0, 15, 60], dtype=np.uint8)
            upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
            lower_skin_ycrcb = np.array([0, 130, 75], dtype=np.uint8)
            upper_skin_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
            lower_skin_lab = np.array([20, 130, 110], dtype=np.uint8)
            upper_skin_lab = np.array([220, 175, 155], dtype=np.uint8)
        
        mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        mask_lab = cv2.inRange(lab, lower_skin_lab, upper_skin_lab)
        
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.bitwise_and(mask, mask_lab)
        
        kernel_elliptical = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel = np.ones((3, 3), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_elliptical, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        self.processing_times['skin_detection'].append(time.time() - start_time)
        return mask

    def find_hand_contour(self, mask):
        start_time = time.time()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.processing_times['contour_finding'].append(time.time() - start_time)
            return None
        
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) < self.detection_threshold:
            self.processing_times['contour_finding'].append(time.time() - start_time)
            return None
        
        hull = cv2.convexHull(max_contour)
        
        epsilon = 0.0025 * cv2.arcLength(hull, True)
        smoothed_contour = cv2.approxPolyDP(hull, epsilon, True)
        
        self.processing_times['contour_finding'].append(time.time() - start_time)
        return smoothed_contour

    def find_palm_center(self, contour, frame):
        if contour is None or len(contour) < 5:
            return None, None, frame
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        palm_center = max_loc
        palm_radius = int(max_val * 0.9)
        
        alpha = 0.5
        overlay = frame.copy()
        cv2.circle(overlay, palm_center, palm_radius, [0, 200, 100], 2, cv2.LINE_AA)
        cv2.circle(overlay, palm_center, palm_radius, [0, 150, 50], 1, cv2.LINE_AA)
        cv2.circle(overlay, palm_center, 8, [0, 100, 0], -1, cv2.LINE_AA)
        cv2.circle(overlay, palm_center, 8, [0, 0, 0], 1, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        self.palm_history.append((palm_center, palm_radius))
        
        if len(self.palm_history) > 3:
            smoothed_center_x = int(np.mean([p[0][0] for p in self.palm_history[-3:]]))
            smoothed_center_y = int(np.mean([p[0][1] for p in self.palm_history[-3:]]))
            smoothed_radius = int(np.mean([p[1] for p in self.palm_history[-3:]]))
            
            palm_center = (smoothed_center_x, smoothed_center_y)
            palm_radius = smoothed_radius
        
        return palm_center, palm_radius, frame

    def compute_angle(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        cos_angle = dot / (mag1 * mag2 + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
        
        return angle

    def compute_k_curvature(self, contour, k=None):
        if k is None:
            k = self.k_value
            
        n_points = len(contour)
        if n_points < 2*k + 1:
            return [], []
        
        curvature_values = []
        for i in range(n_points):
            prev_k = (i - k) % n_points
            next_k = (i + k) % n_points
            
            p1 = contour[prev_k][0]
            p2 = contour[i][0]
            p3 = contour[next_k][0]
            
            angle = self.compute_angle(p1, p2, p3)
            curvature_values.append(angle)
        
        return contour, np.array(curvature_values)

    def find_convexity_defects(self, contour):
        if contour is None or len(contour) < 5:
            return []
            
        hull = cv2.convexHull(contour, returnPoints=False)
        
        try:
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return []
                
            filtered_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                if d / 256.0 > 15:
                    angle = self.compute_angle(start, far, end)
                    if angle < 120:
                        filtered_defects.append((start, end, far, d / 256.0))
            
            return filtered_defects
        except:
            return []

    def detect_fingertips(self, contour, frame, palm_center, palm_radius):
        start_time = time.time()
        viz_frame = frame.copy()
        
        if contour is None or len(contour) < 5 or palm_center is None:
            self.processing_times['fingertip_detection'].append(time.time() - start_time)
            return [], viz_frame
        
        hull = cv2.convexHull(contour)
        convex_defects = self.find_convexity_defects(contour)
        
        contour_points, curvature = self.compute_k_curvature(contour)
        
        candidate_fingertips = []
        
        for i in range(len(hull)):
            point = tuple(hull[i][0])
            
            dist_from_palm = np.sqrt((point[0] - palm_center[0])**2 + (point[1] - palm_center[1])**2)
            
            if dist_from_palm > palm_radius * 1.2:
                candidate_fingertips.append((point, dist_from_palm))
        
        for i in range(len(contour_points)):
            if curvature[i] < 80:
                point = tuple(contour_points[i][0])
                dist_from_palm = np.sqrt((point[0] - palm_center[0])**2 + (point[1] - palm_center[1])**2)
                
                if dist_from_palm > palm_radius * 1.3:
                    candidate_fingertips.append((point, dist_from_palm))
        
        points_array = np.array([p[0] for p in candidate_fingertips])
        
        if len(points_array) > 0:
            try:
                clustering = DBSCAN(eps=30, min_samples=1).fit(points_array)
                labels = clustering.labels_
                
                clustered_fingertips = []
                for label in set(labels):
                    indices = np.where(labels == label)[0]
                    cluster_points = points_array[indices]
                    
                    centroid = tuple(map(int, np.mean(cluster_points, axis=0)))
                    centroid_dist = np.sqrt((centroid[0] - palm_center[0])**2 + (centroid[1] - palm_center[1])**2)
                    
                    clustered_fingertips.append((centroid, centroid_dist))
            except:
                clustered_fingertips = candidate_fingertips
        else:
            clustered_fingertips = []
        
        filtered_fingertips = []
        for pt, dist in clustered_fingertips:
            vec = (pt[0] - palm_center[0], pt[1] - palm_center[1])
            vec_len = np.sqrt(vec[0]**2 + vec[1]**2)
            
            if vec_len > 0:
                vec_norm = (vec[0]/vec_len, vec[1]/vec_len)
                
                if vec_norm[1] < 0:
                    filtered_fingertips.append(pt)
        
        filtered_fingertips = sorted(filtered_fingertips, key=lambda p: p[0])
        
        if self.fingertips_history:
            smoothed_fingertips = self.apply_smooth_motion_filter(self.fingertips_history, filtered_fingertips, alpha=0.6)
            filtered_fingertips = smoothed_fingertips
        
        for i, pt in enumerate(filtered_fingertips):
            hue = int(180 * i / max(1, len(filtered_fingertips)))
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            cv2.circle(viz_frame, pt, 18, [0, 0, 0], 2, cv2.LINE_AA)
            cv2.circle(viz_frame, pt, 16, rgb_color, -1, cv2.LINE_AA)
            cv2.circle(viz_frame, pt, 6, [255, 255, 255], -1, cv2.LINE_AA)
            
            cv2.putText(viz_frame, str(i+1), 
                       (pt[0]-4, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2, cv2.LINE_AA)
            cv2.putText(viz_frame, str(i+1), 
                       (pt[0]-4, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
        
        for start, end, far, depth in convex_defects:
            cv2.circle(viz_frame, far, 5, [0, 0, 255], -1, cv2.LINE_AA)
        
        self.processing_times['fingertip_detection'].append(time.time() - start_time)
        return filtered_fingertips, viz_frame

    def identify_fingers(self, fingertips, palm_center, palm_radius):
        if len(fingertips) == 0 or palm_center is None:
            return {}
        
        if len(fingertips) > 1:
            tip_centroid = np.mean(fingertips, axis=0)
            orientation_vector = tip_centroid - palm_center
            angle = math.atan2(orientation_vector[1], orientation_vector[0]) * 180 / math.pi
            
            self.hand_orientation_history.append(angle)
            avg_orientation = np.mean(self.hand_orientation_history)
            
            is_right_hand = True
            
            if -45 < avg_orientation < 45:
                thumb_likely_bottom = any(pt[1] > palm_center[1] for pt in fingertips)
                is_right_hand = not thumb_likely_bottom
            elif 135 < avg_orientation or avg_orientation < -135:
                thumb_likely_bottom = any(pt[1] > palm_center[1] for pt in fingertips)
                is_right_hand = thumb_likely_bottom
        else:
            is_right_hand = True
        
        if is_right_hand:
            sorted_fingertips = sorted(fingertips, key=lambda pt: pt[0])
        else:
            sorted_fingertips = sorted(fingertips, key=lambda pt: -pt[0])
        
        finger_names = {}
        finger_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
        if len(sorted_fingertips) <= 5:
            for i, finger in enumerate(sorted_fingertips):
                if i < len(finger_labels):
                    finger_names[tuple(finger)] = finger_labels[i]
        
        return finger_names

    def track_finger_movement(self, fingertips, prev_fingertips, max_tracks=5):
        tracks = []
        
        if prev_fingertips and fingertips:
            for curr in fingertips[:max_tracks]:
                closest_dist = float('inf')
                closest_point = None
                
                for prev in prev_fingertips:
                    dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    if dist < closest_dist and dist < 50:
                        closest_dist = dist
                        closest_point = prev
                
                if closest_point:
                    tracks.append((closest_point, curr))
        
        return tracks

    def apply_smooth_motion_filter(self, points_history, current_points, alpha=None):
        if alpha is None:
            alpha = self.smoothing_factor
            
        if not points_history or not current_points:
            return current_points
        
        smoothed_points = []
        
        for curr_pt in current_points:
            if not points_history[-1]:
                smoothed_points.append(curr_pt)
                continue
                
            closest_hist_pt = None
            min_dist = float('inf')
            
            for hist_pt in points_history[-1]:
                dist = np.sqrt((curr_pt[0] - hist_pt[0])**2 + (curr_pt[1] - hist_pt[1])**2)
                if dist < min_dist and dist < 60:
                    min_dist = dist
                    closest_hist_pt = hist_pt
            
            if closest_hist_pt:
                smoothed_x = int((1 - alpha) * closest_hist_pt[0] + alpha * curr_pt[0])
                smoothed_y = int((1 - alpha) * closest_hist_pt[1] + alpha * curr_pt[1])
                smoothed_points.append((smoothed_x, smoothed_y))
            else:
                smoothed_points.append(curr_pt)
        
        return smoothed_points

    def process_mediapipe(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_landmarks = []
        detected_handedness = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                h, w, _ = frame.shape
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                
                detected_landmarks.append(landmarks)
                detected_handedness.append(handedness)
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, detected_landmarks, detected_handedness

    def create_visualization(self, frame, fingertips, palm_center, palm_radius):
        start_time = time.time()
        
        info_panel = np.ones((frame.shape[0], 250, 3), dtype=np.uint8) * 240
        
        header_gradient = np.ones((60, info_panel.shape[1], 3), dtype=np.uint8)
        for i in range(header_gradient.shape[1]):
            header_gradient[:, i] = [30, 50 + i//5, 100 + i//3]
        info_panel[0:60, :] = header_gradient
        
        cv2.putText(info_panel, "GestureFlow", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        if palm_center is not None:
            cv2.putText(info_panel, "Hand Detected", (20, 90), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 100, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(info_panel, "Waiting for hand...", (20, 90), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
        
        cv2.rectangle(info_panel, (20, 110), (230, 180), (220, 220, 220), -1)
        cv2.rectangle(info_panel, (20, 110), (230, 180), (180, 180, 180), 2)
        
        if fingertips:
            cv2.putText(info_panel, f"Tracking {len(fingertips)} points", (30, 140), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(info_panel, "No points tracked", (30, 140), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
            
        if palm_center and palm_radius:
            cv2.putText(info_panel, f"Palm radius: {palm_radius}px", (30, 165), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1, cv2.LINE_AA)
        
        cv2.rectangle(info_panel, (20, 200), (230, 280), (230, 230, 230), -1)
        cv2.rectangle(info_panel, (20, 200), (230, 280), (190, 190, 190), 1)
        
        cv2.putText(info_panel, "Performance", (75, 225), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
        
        avg_fps = int(np.mean(self.fps_history)) if self.fps_history else 0
        cv2.putText(info_panel, f"FPS: {avg_fps}", (30, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        
        minutes = int(self.hand_detected_time // 60)
        seconds = int(self.hand_detected_time % 60)
        cv2.putText(info_panel, f"Detection time: {minutes:02d}:{seconds:02d}", (20, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        
        cv2.putText(info_panel, "Processing time (ms):", (20, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        
        y_pos = 370
        for component, times in self.processing_times.items():
            if times:
                avg_time = int(np.mean(times) * 1000)
                cv2.putText(info_panel, f"- {component}: {avg_time}", (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
                y_pos += 20
        
        cv2.putText(info_panel, "Controls:", (20, info_panel.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(info_panel, "q - Quit", (30, info_panel.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(info_panel, "d - Change display mode", (30, info_panel.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        
        combined_frame = np.hstack((frame, info_panel))
        
        self.processing_times['rendering'].append(time.time() - start_time)
        return combined_frame

    def run(self):
        try:
            if not self.initialize_camera():
                return False
            
            while True:
                loop_start = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                if self.use_mediapipe:
                    mp_frame, mp_landmarks, mp_handedness = self.process_mediapipe(frame.copy())
                    
                    if mp_landmarks:
                        landmarks = mp_landmarks[0]
                        
                        fingertip_indices = [4, 8, 12, 16, 20]
                        fingertips = [landmarks[idx] for idx in fingertip_indices]
                        
                        palm_center = tuple(map(int, np.mean([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)))
                        
                        palm_radius = int(np.linalg.norm(np.array(palm_center) - np.array(landmarks[0])))
                        
                        if self.fingertips_history:
                            finger_tracks = self.track_finger_movement(fingertips, self.fingertips_history[-1])
                            
                            for prev_pt, curr_pt in finger_tracks:
                                finger_id = str(prev_pt)
                                if finger_id not in self.motion_trails:
                                    self.motion_trails[finger_id] = deque(maxlen=15)
                                
                                self.motion_trails[finger_id].append(curr_pt)
                        
                        self.fingertips_history.append(fingertips)
                        
                        if not self.hand_present:
                            self.hand_present = True
                            self.last_time_check = time.time()
                        else:
                            self.hand_detected_time += time.time() - self.last_time_check
                            self.last_time_check = time.time()
                        
                        result_frame = self.create_visualization(
                            mp_frame, fingertips, palm_center, palm_radius)
                    else:
                        self.hand_present = False
                        self.last_time_check = time.time()
                        result_frame = self.create_visualization(
                            mp_frame, [], None, None)
                else:
                    if self.roi is not None:
                        x, y, w, h = self.roi
                        x = max(0, x - 30)
                        y = max(0, y - 30)
                        w = min(frame.shape[1] - x, w + 60)
                        h = min(frame.shape[0] - y, h + 60)
                        roi_frame = frame[y:y+h, x:x+w]
                        skin_mask = self.detect_skin(roi_frame)
                    else:
                        skin_mask = self.detect_skin(frame)
                        roi_frame = frame
                        x, y = 0, 0
                    
                    hand_contour = self.find_hand_contour(skin_mask)
                    
                    if hand_contour is not None:
                        if not self.hand_present:
                            self.hand_present = True
                            self.last_time_check = time.time()
                        else:
                            self.hand_detected_time += time.time() - self.last_time_check
                            self.last_time_check = time.time()
                            
                        if self.roi is not None:
                            hand_contour = hand_contour.copy()
                            hand_contour[:, :, 0] += x
                            hand_contour[:, :, 1] += y
                        
                        contour_viz = frame.copy()
                        cv2.drawContours(contour_viz, [hand_contour], -1, (0, 220, 0), 2, cv2.LINE_AA)
                        frame = cv2.addWeighted(frame, 0.7, contour_viz, 0.3, 0)
                        
                        palm_center, palm_radius, frame = self.find_palm_center(hand_contour, frame)
                        
                        defects = self.find_convexity_defects(hand_contour)
                        
                        fingertips, frame = self.detect_fingertips(hand_contour, frame, palm_center, palm_radius)
                        
                        finger_names = self.identify_fingers(fingertips, palm_center, palm_radius)
                        
                        if self.fingertips_history:
                            finger_tracks = self.track_finger_movement(fingertips, self.fingertips_history[-1])
                            
                            for prev_pt, curr_pt in finger_tracks:
                                finger_id = str(prev_pt)
                                if finger_id not in self.motion_trails:
                                    self.motion_trails[finger_id] = deque(maxlen=15)
                                
                                self.motion_trails[finger_id].append(curr_pt)
                        
                        self.fingertips_history.append(fingertips)
                        
                        x, y, w, h = cv2.boundingRect(hand_contour)
                        self.roi = (x, y, w, h)
                        
                        for pt, name in finger_names.items():
                            cv2.putText(frame, name, (pt[0] + 10, pt[1] - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, name, (pt[0] + 10, pt[1] - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        result_frame = self.create_visualization(
                            frame, fingertips, palm_center, palm_radius)
                    else:
                        self.roi = None
                        self.hand_present = False
                        self.last_time_check = time.time()
                        self.fingertips_history.clear()
                        self.motion_trails.clear()
                        
                        result_frame = self.create_visualization(
                            frame, [], None, None)
                
                    color_mask = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
                    color_mask[:,:,0] = 0
                    color_mask[:,:,1] = skin_mask
                    color_mask[:,:,2] = 0
                    
                    cv2.imshow("Skin Detection", color_mask)
                
                cv2.imshow("GestureFlow", result_frame)
                
                loop_time = time.time() - loop_start
                self.fps_history.append(1.0 / max(loop_time, 0.001))
                self.processing_times['total'].append(loop_time)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    if self.display_mode == "standard":
                        self.display_mode = "enhanced"
                    else:
                        self.display_mode = "standard"
                elif key == ord('l'):
                    if self.light_condition == "auto":
                        self.light_condition = "normal"
                    elif self.light_condition == "normal":
                        self.light_condition = "bright"
                    elif self.light_condition == "bright":
                        self.light_condition = "dark"
                    else:
                        self.light_condition = "auto"
                elif key == ord('m'):
                    self.use_mediapipe = not self.use_mediapipe
        
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            
        return True

def main():
    system = GestureFlow()
    system.run()

if __name__ == "__main__":
    main()