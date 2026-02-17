"""
Background quality checker module.
Checks if the background is suitable for passport photos (clean, uniform color).
"""

import cv2
import numpy as np
from collections import Counter


class BackgroundChecker:
    """Checker for background quality in passport photos"""
    
    def __init__(self):
        """Initialize the background checker"""
        pass
    
    def check_background(self, frame, face_data, shoulder_data,
                        uniformity_threshold=30,
                        acceptable_colors=['white', 'light_blue', 'light_gray'],
                        min_uniformity_percentage=60):
        """
        Check if background is suitable for passport photos
        
        Args:
            frame: Image to check
            face_data: Face data with landmarks
            shoulder_data: Shoulder data with keypoints
            uniformity_threshold: Max standard deviation for uniform background (0-255)
            acceptable_colors: List of acceptable background colors
            min_uniformity_percentage: Minimum percentage of background that must be uniform
            
        Returns:
            Dictionary with check results
        """
        # Extract background region (exclude person)
        background_mask = self._create_background_mask(frame, face_data, shoulder_data)
        background_pixels = frame[background_mask]
        
        if len(background_pixels) == 0:
            return {
                'passed': False,
                'detected': False,
                'message': 'Cannot detect background area',
                'check_name': 'Background Quality'
            }
        
        # Check 1: Color uniformity
        is_uniform, uniformity_score, std_dev = self._check_uniformity(
            background_pixels, uniformity_threshold
        )
        
        # Check 2: Dominant color detection
        dominant_color, color_name, is_acceptable_color = self._check_color(
            background_pixels, acceptable_colors
        )
        
        # Check 3: Background coverage (no busy patterns)
        coverage_ok, coverage_percentage = self._check_coverage(
            background_mask, min_uniformity_percentage
        )
        
        # Overall assessment
        passed = is_uniform and is_acceptable_color and coverage_ok
        
        # Generate message
        if passed:
            message = f"Good {color_name} background"
        else:
            issues = []
            if not is_uniform:
                issues.append(f"not uniform (σ={std_dev:.1f})")
            if not is_acceptable_color:
                issues.append(f"wrong color ({color_name})")
            if not coverage_ok:
                issues.append(f"insufficient coverage ({coverage_percentage:.0f}%)")
            message = "Background: " + ", ".join(issues)
        
        return {
            'passed': passed,
            'detected': True,
            'is_uniform': is_uniform,
            'uniformity_score': uniformity_score,
            'std_dev': std_dev,
            'dominant_color': dominant_color,
            'color_name': color_name,
            'is_acceptable_color': is_acceptable_color,
            'coverage_ok': coverage_ok,
            'coverage_percentage': coverage_percentage,
            'background_mask': background_mask,
            'message': message,
            'check_name': 'Background Quality'
        }
    
    def _create_background_mask(self, frame, face_data, shoulder_data):
        """
        Create a mask for the background region (excluding person)
        
        Args:
            frame: Image
            face_data: Face landmarks
            shoulder_data: Shoulder keypoints
            
        Returns:
            Boolean mask where True = background, False = person
        """
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=bool)  # Start with all background
        
        # Create person mask from face and shoulder points
        person_points = []
        
        # Add face contour points
        if 'forehead' in face_data:
            forehead = face_data['forehead'][:2].astype(int)
            chin = face_data['chin'][:2].astype(int)
            left_eye = face_data['left_eye'][:2].astype(int)
            right_eye = face_data['right_eye'][:2].astype(int)
            
            # Expand face region
            face_width = np.linalg.norm(right_eye - left_eye) * 2.5
            face_height = np.linalg.norm(forehead - chin) * 1.3
            face_center = (forehead + chin) / 2
            
            # Create rectangular region around face
            x1 = int(face_center[0] - face_width / 2)
            x2 = int(face_center[0] + face_width / 2)
            y1 = int(face_center[1] - face_height / 2)
            y2 = int(face_center[1] + face_height / 2)
            
            # Ensure bounds are within frame
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            mask[y1:y2, x1:x2] = False
        
        # Add shoulder region
        if 'left_shoulder' in shoulder_data and 'right_shoulder' in shoulder_data:
            left_shoulder = shoulder_data['left_shoulder'][:2].astype(int)
            right_shoulder = shoulder_data['right_shoulder'][:2].astype(int)
            
            # Create region below face to shoulders
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder) * 1.5
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Extended shoulder region
            x1 = int(shoulder_center[0] - shoulder_width / 2)
            x2 = int(shoulder_center[0] + shoulder_width / 2)
            y1 = int(min(left_shoulder[1], right_shoulder[1]) - 50)
            y2 = int(max(left_shoulder[1], right_shoulder[1]) + 100)
            
            # Ensure bounds are within frame
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            mask[y1:y2, x1:x2] = False
        
        return mask
    
    def _check_uniformity(self, pixels, threshold):
        """
        Check if background pixels are uniform in color
        
        Args:
            pixels: Array of background pixels (N, 3)
            threshold: Maximum standard deviation for uniformity
            
        Returns:
            Tuple of (is_uniform, uniformity_score, std_dev)
        """
        # Calculate standard deviation for each channel
        std_dev = np.std(pixels, axis=0)
        avg_std = np.mean(std_dev)
        
        # Uniformity score (0-100, higher is better)
        uniformity_score = max(0, 100 - (avg_std / threshold * 100))
        
        is_uniform = avg_std < threshold
        
        return is_uniform, uniformity_score, avg_std
    
    def _check_color(self, pixels, acceptable_colors):
        """
        Check if the dominant background color is acceptable
        
        Args:
            pixels: Array of background pixels (N, 3) in BGR
            acceptable_colors: List of acceptable color names
            
        Returns:
            Tuple of (dominant_color_bgr, color_name, is_acceptable)
        """
        # Calculate average color
        avg_color = np.mean(pixels, axis=0).astype(int)
        
        # Determine color name and acceptability
        color_name = self._classify_color(avg_color)
        is_acceptable = color_name in acceptable_colors
        
        return avg_color, color_name, is_acceptable
    
    def _classify_color(self, bgr_color):
        """
        Classify a BGR color into a named category
        
        Args:
            bgr_color: BGR color array [B, G, R]
            
        Returns:
            String color name
        """
        b, g, r = bgr_color
        
        # Convert to brightness
        brightness = (int(r) + int(g) + int(b)) / 3
        
        # Color classification
        if brightness > 200:  # Very bright
            if abs(r - g) < 30 and abs(g - b) < 30:
                return 'white'
            elif b > r and b > g:
                return 'light_blue'
            else:
                return 'light_gray'
        
        elif brightness > 150:  # Moderately bright
            if b > r + 20 and b > g + 20:
                return 'blue'
            elif abs(r - g) < 20 and abs(g - b) < 20:
                return 'gray'
            else:
                return 'light_colored'
        
        elif brightness > 100:  # Medium
            if b > r + 30:
                return 'blue'
            elif r > g + 30 and r > b + 30:
                return 'red'
            elif g > r + 30 and g > b + 30:
                return 'green'
            else:
                return 'gray'
        
        else:  # Dark
            return 'dark'
    
    def _check_coverage(self, background_mask, min_percentage):
        """
        Check if enough of the frame is background
        
        Args:
            background_mask: Boolean mask of background
            min_percentage: Minimum required percentage
            
        Returns:
            Tuple of (coverage_ok, coverage_percentage)
        """
        total_pixels = background_mask.size
        background_pixels = np.sum(background_mask)
        coverage_percentage = (background_pixels / total_pixels) * 100
        
        coverage_ok = coverage_percentage >= min_percentage
        
        return coverage_ok, coverage_percentage
    
    def visualize_background_check(self, frame, result):
        """
        Visualize background check results
        
        Args:
            frame: Original frame
            result: Result dictionary from check_background()
            
        Returns:
            Frame with visualization
        """
        if not result['detected']:
            return frame
        
        vis_frame = frame.copy()
        
        # Show background mask overlay
        if 'background_mask' in result:
            mask = result['background_mask']
            
            # Create colored overlay for background
            overlay = vis_frame.copy()
            
            if result['passed']:
                # Green tint for good background
                overlay[mask] = cv2.addWeighted(
                    overlay[mask], 0.7,
                    np.full_like(overlay[mask], [0, 255, 0]), 0.3,
                    0
                )
            else:
                # Red tint for bad background
                overlay[mask] = cv2.addWeighted(
                    overlay[mask], 0.7,
                    np.full_like(overlay[mask], [0, 0, 255]), 0.3,
                    0
                )
            
            cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
        
        # Draw info box
        self._draw_background_info(vis_frame, result)
        
        return vis_frame
    
    def _draw_background_info(self, frame, result):
        """
        Draw background check information on frame
        
        Args:
            frame: Frame to draw on
            result: Check result dictionary
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        
        # Position in bottom-right corner
        panel_x = w - 320
        panel_y = h - 180
        panel_width = 310
        panel_height = 170
        
        # Draw semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_pos = panel_y + 30
        x_pos = panel_x + 10
        
        # Title
        status_color = (0, 255, 0) if result['passed'] else (0, 0, 255)
        cv2.putText(frame, "BACKGROUND CHECK", 
                   (x_pos, y_pos), font, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        # Status
        status = "PASS" if result['passed'] else "FAIL"
        cv2.putText(frame, f"Status: {status}", 
                   (x_pos, y_pos), font, 0.5, status_color, 2)
        y_pos += 25
        
        # Color info
        color_status = "✓" if result['is_acceptable_color'] else "✗"
        color_display = result['color_name'].replace('_', ' ').title()
        cv2.putText(frame, f"{color_status} Color: {color_display}", 
                   (x_pos, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        # Uniformity info
        uniform_status = "✓" if result['is_uniform'] else "✗"
        cv2.putText(frame, f"{uniform_status} Uniformity: {result['uniformity_score']:.0f}%", 
                   (x_pos, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        # Coverage info
        coverage_status = "✓" if result['coverage_ok'] else "✗"
        cv2.putText(frame, f"{coverage_status} Coverage: {result['coverage_percentage']:.0f}%", 
                   (x_pos, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 25
        
        # Color sample
        if 'dominant_color' in result:
            color_bgr = tuple(map(int, result['dominant_color']))
            cv2.rectangle(frame, 
                         (x_pos, y_pos - 15), 
                         (x_pos + 80, y_pos + 5),
                         color_bgr, -1)
            cv2.rectangle(frame, 
                         (x_pos, y_pos - 15), 
                         (x_pos + 80, y_pos + 5),
                         (255, 255, 255), 1)
            cv2.putText(frame, "Sample", 
                       (x_pos + 90, y_pos), 
                       font, 0.4, (200, 200, 200), 1)

