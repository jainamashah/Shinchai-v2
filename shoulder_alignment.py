"""
Shoulder alignment detection module.
Provides functions to check if shoulders are level and straight.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_EAR = 7
RIGHT_EAR = 8
NOSE = 0


class ShoulderAlignmentDetector:
    """Detector for shoulder alignment"""
    
    def __init__(self, model_path=None, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        Initialize the Shoulder Alignment Detector
        
        Args:
            model_path: Path to the MediaPipe pose landmarker model
                       If None, will download pose_landmarker_lite.task
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        # If no model path provided, download lite model
        if model_path is None:
            model_path = 'pose_landmarker_full.task'
            if not os.path.exists(model_path):
                model_path = download_pose_model(model_path)
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download a pose landmarker model from:\n"
                f"https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models"
            )
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print(f"Pose detector initialized with model: {model_path}")
    
    def detect_pose(self, frame):
        """
        Detect pose landmarks in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            MediaPipe pose detection results
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        return results
    
    def get_shoulder_data(self, landmarks, image_width, image_height):
        """
        Extract shoulder alignment data from pose landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks (NormalizedLandmarkList)
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Dictionary containing shoulder alignment data
        """
        def to_pixel(landmark):
            return np.array([
                landmark.x * image_width,
                landmark.y * image_height,
                landmark.z * image_width  # z is relative depth
            ])
        
        # Get key points
        left_shoulder = to_pixel(landmarks[LEFT_SHOULDER])
        right_shoulder = to_pixel(landmarks[RIGHT_SHOULDER])
        left_hip = to_pixel(landmarks[LEFT_HIP])
        right_hip = to_pixel(landmarks[RIGHT_HIP])
        left_ear = to_pixel(landmarks[LEFT_EAR])
        right_ear = to_pixel(landmarks[RIGHT_EAR])
        
        # Calculate centers
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        head_center = (left_ear + right_ear) / 2
        
        # Calculate shoulder line vector
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_distance = np.linalg.norm(shoulder_vector[:2])
        
        # Calculate shoulder tilt angle (deviation from horizontal)
        # We want 0 degrees when shoulders are level
        shoulder_tilt = np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180 / np.pi
        # Normalize to -90 to 90 range (deviation from horizontal)
        if shoulder_tilt > 90:
            shoulder_tilt = shoulder_tilt - 180
        elif shoulder_tilt < -90:
            shoulder_tilt = shoulder_tilt + 180
        
        # Check shoulder depth difference (one shoulder forward)
        shoulder_depth_diff = abs(left_shoulder[2] - right_shoulder[2])
        shoulder_depth_ratio = shoulder_depth_diff / shoulder_distance if shoulder_distance > 0 else 0
        
        return {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_hip': left_hip,
            'right_hip': right_hip,
            'shoulder_center': shoulder_center,
            'hip_center': hip_center,
            'head_center': head_center,
            'shoulder_vector': shoulder_vector,
            'shoulder_distance': shoulder_distance,
            'shoulder_tilt': shoulder_tilt,  # Deviation from horizontal (-90 to 90)
            'shoulder_depth_diff': shoulder_depth_diff,
            'shoulder_depth_ratio': shoulder_depth_ratio
        }
    
    def is_shoulders_level(self, shoulder_tilt, threshold=10):
        """
        Check if shoulders are level (not tilted)
        
        Args:
            shoulder_tilt: Shoulder tilt angle in degrees (deviation from horizontal)
            threshold: Maximum allowed tilt in degrees
            
        Returns:
            Boolean indicating if shoulders are level
        """
        return abs(shoulder_tilt) <= threshold
    
    def is_shoulders_square(self, shoulder_depth_ratio, threshold=0.3):
        """
        Check if shoulders are square to camera (not rotated)
        
        Args:
            shoulder_depth_ratio: Ratio of depth difference to shoulder width
            threshold: Maximum allowed ratio
            
        Returns:
            Boolean indicating if shoulders are square
        """
        return shoulder_depth_ratio <= threshold
    
    def is_shoulders_straight(self, shoulder_data,
                             level_threshold=10,
                             square_threshold=0.2):
        """
        Check if shoulders are straight (level and square)
        
        Args:
            shoulder_data: Dictionary from get_shoulder_data()
            level_threshold: Max shoulder tilt in degrees
            square_threshold: Max shoulder rotation ratio
            
        Returns:
            Boolean indicating if shoulders are straight
        """
        level = self.is_shoulders_level(shoulder_data['shoulder_tilt'], level_threshold)
        square = self.is_shoulders_square(shoulder_data['shoulder_depth_ratio'], 
                                         square_threshold)
        
        return level and square
    
    def check_alignment(self, frame,
                       level_threshold=10,
                       square_threshold=0.2):
        """
        Check shoulder alignment for person in frame
        
        Args:
            frame: BGR image from OpenCV
            level_threshold: Max shoulder tilt in degrees
            square_threshold: Max shoulder rotation ratio
            
        Returns:
            Dictionary with alignment results
        """
        h, w, _ = frame.shape
        results = self.detect_pose(frame)
        
        # Check if any poses were detected
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return {
                'passed': False,
                'detected': False,
                'person_data': None,
                'message': 'No person detected'
            }
        
        # Get the first pose (we set num_poses=1 in options)
        pose_landmarks = results.pose_landmarks[0]
        
        # Get shoulder data
        shoulder_data = self.get_shoulder_data(pose_landmarks, w, h)
        
        # Check alignment
        is_straight = self.is_shoulders_straight(
            shoulder_data,
            level_threshold,
            square_threshold
        )
        
        # Add individual check results
        shoulder_data['is_level'] = self.is_shoulders_level(
            shoulder_data['shoulder_tilt'], level_threshold
        )
        shoulder_data['is_square'] = self.is_shoulders_square(
            shoulder_data['shoulder_depth_ratio'], square_threshold
        )
        shoulder_data['is_straight'] = is_straight
        
        # Determine specific message
        if not shoulder_data['is_level']:
            if shoulder_data['shoulder_tilt'] > 0:
                message = "Level your shoulders - right side is lower"
            else:
                message = "Level your shoulders - left side is lower"
        elif not shoulder_data['is_square']:
            message = "Square your shoulders to camera"
        else:
            message = "Good shoulder alignment!"
        
        return {
            'passed': is_straight,
            'detected': True,
            'person_data': shoulder_data,
            'message': message
        }
    
    def close(self):
        """Clean up resources"""
        self.detector.close()


def draw_shoulder_lines(frame, shoulder_data, is_aligned=False):
    """
    Draw shoulder alignment visualization
    
    Args:
        frame: Image to draw on
        shoulder_data: Shoulder data dictionary
        is_aligned: Whether shoulders are aligned (affects color)
    """
    # Choose color based on alignment
    color = (0, 255, 0) if is_aligned else (0, 0, 255)
    
    # Draw shoulder line
    left_shoulder = shoulder_data['left_shoulder'][:2].astype(int)
    right_shoulder = shoulder_data['right_shoulder'][:2].astype(int)
    cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), color, 4)
    
    # Draw shoulder points
    cv2.circle(frame, tuple(left_shoulder), 10, color, -1)
    cv2.circle(frame, tuple(right_shoulder), 10, color, -1)
    
    # Draw horizontal reference through shoulder center
    shoulder_center = shoulder_data['shoulder_center'][:2].astype(int)
    ref_left = shoulder_center + np.array([-150, 0])
    ref_right = shoulder_center + np.array([150, 0])
    cv2.line(frame, tuple(ref_left.astype(int)), tuple(ref_right.astype(int)), 
             (200, 200, 200), 2, cv2.LINE_AA)
    
    # Draw center point
    cv2.circle(frame, tuple(shoulder_center), 6, (255, 255, 0), -1)
    
    return frame


def draw_shoulder_skeleton(frame, shoulder_data, is_aligned=False):
    """
    Draw a simple skeleton showing shoulder position
    
    Args:
        frame: Image to draw on
        shoulder_data: Shoulder data dictionary
        is_aligned: Whether shoulders are aligned
    """
    color = (0, 255, 0) if is_aligned else (0, 165, 255)
    
    # Get points
    left_shoulder = shoulder_data['left_shoulder'][:2].astype(int)
    right_shoulder = shoulder_data['right_shoulder'][:2].astype(int)
    left_hip = shoulder_data['left_hip'][:2].astype(int)
    right_hip = shoulder_data['right_hip'][:2].astype(int)
    head_center = shoulder_data['head_center'][:2].astype(int)
    shoulder_center = shoulder_data['shoulder_center'][:2].astype(int)
    
    # Draw torso (light)
    cv2.line(frame, tuple(left_shoulder), tuple(left_hip), color, 1, cv2.LINE_AA)
    cv2.line(frame, tuple(right_shoulder), tuple(right_hip), color, 1, cv2.LINE_AA)
    
    # Draw neck (light)
    cv2.line(frame, tuple(shoulder_center), tuple(head_center), color, 1, cv2.LINE_AA)
    
    # Draw head (light)
    cv2.circle(frame, tuple(head_center), 15, color, 1)
    
    return frame


def draw_shoulder_info(frame, shoulder_data, is_aligned):
    """
    Draw shoulder alignment information on frame
    
    Args:
        frame: Image to draw on
        shoulder_data: Shoulder data dictionary
        is_aligned: Whether shoulders are straight
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Position text near shoulder center
    text_pos = shoulder_data['shoulder_center'][:2].astype(int)
    text_pos[0] += 100  # Offset to the right
    text_pos[1] -= 30
    
    # Shoulder tilt (deviation from horizontal)
    tilt_color = (0, 255, 0) if shoulder_data['is_level'] else (0, 0, 255)
    cv2.putText(frame, f"Shoulder Tilt: {shoulder_data['shoulder_tilt']:.1f}deg", 
               tuple(text_pos), font, 0.6, tilt_color, 2)
    text_pos[1] += 25
    
    # Shoulder square (depth difference)
    square_color = (0, 255, 0) if shoulder_data['is_square'] else (0, 0, 255)
    cv2.putText(frame, f"Depth Ratio: {shoulder_data['shoulder_depth_ratio']:.2f}", 
               tuple(text_pos), font, 0.6, square_color, 2)
    
    return frame
