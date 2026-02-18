"""
Face alignment detection module.
Provides functions to check if faces are properly aligned and oriented.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Key landmark indices for Face Mesh
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152
FOREHEAD = 10
LEFT_EAR = 234   # NEW: Left ear tragus (the small pointed eminence of the external ear)
RIGHT_EAR = 454  # NEW: Right ear tragus

class FaceAlignmentDetector:
    """Detector for face alignment and orientation"""
    
    def __init__(self, model_path='face_landmarker.task'):
        """
        Initialize the Face Alignment Detector
        
        Args:
            model_path: Path to the MediaPipe face landmarker model
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            MediaPipe detection results
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)
    
    def get_face_orientation(self, landmarks, image_width, image_height):
        """
        Calculate face orientation from facial landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Dictionary containing face orientation data
        """
        def to_pixel(landmark):
            return np.array([
                landmark.x * image_width,
                landmark.y * image_height,
                landmark.z * image_width
            ])
        
        # Get key points
        left_eye = to_pixel(landmarks[LEFT_EYE])
        right_eye = to_pixel(landmarks[RIGHT_EYE])
        nose_tip = to_pixel(landmarks[NOSE_TIP])
        mouth_left = to_pixel(landmarks[MOUTH_LEFT])
        mouth_right = to_pixel(landmarks[MOUTH_RIGHT])
        chin = to_pixel(landmarks[CHIN])
        forehead = to_pixel(landmarks[FOREHEAD])
        
        # NEW: Get ear landmarks
        left_ear = to_pixel(landmarks[LEFT_EAR])
        right_ear = to_pixel(landmarks[RIGHT_EAR])
        
        # Calculate key centers
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (mouth_left + mouth_right) / 2
        
        # Calculate face vectors
        face_vertical = mouth_center - eye_center
        face_horizontal = right_eye - left_eye
        face_normal = np.cross(face_horizontal, face_vertical)
        
        # Normalize vectors
        face_normal = face_normal / np.linalg.norm(face_normal)
        face_vertical = face_vertical / np.linalg.norm(face_vertical)
        face_horizontal = face_horizontal / np.linalg.norm(face_horizontal)
        
        # Calculate rotation angles
        eye_distance = np.linalg.norm(right_eye - left_eye)
        yaw = np.arctan2(face_normal[0], face_normal[2]) * 180 / np.pi
        pitch = np.arctan2(face_normal[1], face_normal[2]) * 180 / np.pi
        roll = np.arctan2(face_horizontal[1], face_horizontal[0]) * 180 / np.pi
        
        # Calculate verticality (angle from true vertical)
        true_vertical = np.array([0, 1, 0])
        verticality_angle = np.arccos(np.clip(np.dot(face_vertical, true_vertical), -1.0, 1.0)) * 180 / np.pi
        
        return {
            'normal': face_normal,
            'horizontal': face_horizontal,
            'vertical': face_vertical,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'verticality_angle': verticality_angle,
            'eye_center': eye_center,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_ear': left_ear,      # NEW
            'right_ear': right_ear,    # NEW
            'nose': nose_tip,
            'mouth_center': mouth_center,
            'chin': chin,
            'forehead': forehead,
            'eye_distance': eye_distance
        }
    
    def is_face_vertical(self, verticality_angle, threshold=20):
        """
        Check if face is oriented vertically (upright)
        
        Args:
            verticality_angle: Angle from true vertical in degrees
            threshold: Maximum allowed deviation from vertical
            
        Returns:
            Boolean indicating if face is vertical
        """
        return verticality_angle <= threshold
    
    def is_face_aligned(self, face_data, 
                       yaw_threshold=15, 
                       pitch_threshold=15, 
                       roll_threshold=20, 
                       verticality_threshold=20):
        """
        Check if face has good alignment based on all orientation parameters
        
        Args:
            face_data: Dictionary from get_face_orientation()
            yaw_threshold: Maximum yaw angle in degrees
            pitch_threshold: Maximum pitch angle in degrees
            roll_threshold: Maximum roll angle in degrees
            verticality_threshold: Maximum verticality angle in degrees
            
        Returns:
            Boolean indicating if face is well aligned
        """
        angle_check = (abs(face_data['yaw']) <= yaw_threshold and 
                      abs(face_data['pitch']) <= pitch_threshold and 
                      abs(face_data['roll']) <= roll_threshold)
        
        vertical_check = self.is_face_vertical(face_data['verticality_angle'], 
                                               verticality_threshold)
        
        return angle_check and vertical_check
    
    def check_alignment(self, frame, 
                       yaw_threshold=15,
                       pitch_threshold=15, 
                       roll_threshold=20,
                       verticality_threshold=20):
        """
        Check alignment for all faces in a frame
        
        Args:
            frame: BGR image from OpenCV
            yaw_threshold: Maximum yaw angle in degrees
            pitch_threshold: Maximum pitch angle in degrees
            roll_threshold: Maximum roll angle in degrees
            verticality_threshold: Maximum verticality angle in degrees
            
        Returns:
            Dictionary with alignment results
        """
        h, w, _ = frame.shape
        results = self.detect_faces(frame)
        
        face_data_list = []
        all_aligned = True
        
        if not results.face_landmarks:
            return {
                'passed': False,
                'num_faces': 0,
                'faces': [],
                'all_aligned': False,
                'message': 'No faces detected'
            }
        
        for face_landmarks in results.face_landmarks:
            face_data = self.get_face_orientation(face_landmarks, w, h)
            is_aligned = self.is_face_aligned(
                face_data, 
                yaw_threshold, 
                pitch_threshold, 
                roll_threshold,
                verticality_threshold
            )
            face_data['is_aligned'] = is_aligned
            face_data_list.append(face_data)
            
            if not is_aligned:
                all_aligned = False
        
        # Calculate averages
        avg_yaw = np.mean([f['yaw'] for f in face_data_list])
        avg_pitch = np.mean([f['pitch'] for f in face_data_list])
        avg_roll = np.mean([f['roll'] for f in face_data_list])
        avg_vert = np.mean([f['verticality_angle'] for f in face_data_list])
        
        # Determine message
        if not self.is_face_vertical(avg_vert, verticality_threshold):
            message = "Stand up straight!"
        elif abs(avg_yaw) > yaw_threshold:
            message = "Turn left" if avg_yaw > 0 else "Turn right"
        elif abs(avg_pitch) > pitch_threshold:
            message = "Look up" if avg_pitch < 0 else "Look down"
        elif abs(avg_roll) > roll_threshold:
            message = "Tilt head straight"
        else:
            message = "Good alignment!"
        
        return {
            'passed': all_aligned,
            'num_faces': len(face_data_list),
            'faces': face_data_list,
            'all_aligned': all_aligned,
            'avg_yaw': avg_yaw,
            'avg_pitch': avg_pitch,
            'avg_roll': avg_roll,
            'avg_verticality': avg_vert,
            'message': message
        }
    
    def close(self):
        """Clean up resources"""
        self.detector.close()
