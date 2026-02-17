"""
Bounding box drawing module.
Provides functions to draw 3D bounding boxes around detected faces.
"""

import cv2
import numpy as np


class BoundingBoxDrawer:
    """Drawer for 3D bounding boxes around faces"""
    
    def __init__(self):
        """Initialize the bounding box drawer"""
        pass

    def draw_passport_box(self, frame, face_data, shoulder_data, check_frame_fit=True):
        """
        Draw a passport-style 2D box with 35:45 (width:height) ratio
        that includes face and both shoulders.
        Only drawn when both face and shoulder alignment checks pass.
        
        Args:
            frame: Image to draw on
            face_data: Face orientation data containing landmarks
            shoulder_data: Shoulder data containing shoulder positions
            check_frame_fit: If True, only show "ready" when box edges touch frame edges
            
        Returns:
            Tuple of (frame with passport box drawn, is_ready_for_capture)
        """
        # Get key points
        left_shoulder = shoulder_data['left_shoulder'][:2]
        right_shoulder = shoulder_data['right_shoulder'][:2]
        forehead = face_data['forehead'][:2]
        chin = face_data['chin'][:2]
        
        # Calculate the center point between face and shoulders
        face_center = (forehead + chin) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        box_center = (face_center + shoulder_center) / 2
        
        # Calculate required width to include both shoulders with some padding
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        required_width = shoulder_width * 1.4  # Add 40% padding
        
        # Calculate height based on 35:45 ratio (45/35 = 1.2857)
        aspect_ratio = 45.0 / 35.0
        box_width = required_width
        box_height = box_width * aspect_ratio
        
        # Ensure the box includes face top and bottom with padding
        face_height = np.linalg.norm(forehead - chin)
        shoulder_to_forehead = np.linalg.norm(shoulder_center - forehead)
        required_height = shoulder_to_forehead + face_height * 1.5
        
        # Use the larger of the two heights
        if required_height > box_height:
            box_height = required_height
            box_width = box_height / aspect_ratio
        
        # Calculate box corners
        half_width = box_width / 2
        half_height = box_height / 2
        
        # Calculate rotation angle from shoulder line
        shoulder_vector = right_shoulder - left_shoulder
        rotation_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        
        # Create rotation matrix
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        
        # Define corners relative to center (unrotated)
        corners_relative = np.array([
            [-half_width, -half_height],  # Top-left
            [half_width, -half_height],   # Top-right
            [half_width, half_height],    # Bottom-right
            [-half_width, half_height]    # Bottom-left
        ])
        
        # Rotate corners
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        corners_rotated = np.dot(corners_relative, rotation_matrix.T)
        
        # Translate to box center
        corners = corners_rotated + box_center
        corners = corners.astype(np.int32)
        
        # ============================================
        # NEW: Check if PARALLEL OPPOSITE edges touch frame edges
        # ============================================
        frame_height, frame_width = frame.shape[:2]
        edge_threshold = 20  # pixels - how close to edge counts as "touching"
        
        is_ready_for_capture = False
        
        if check_frame_fit:
            # Define the four edges of the passport box
            # Edge 0: Top (corners[0] to corners[1])
            # Edge 1: Right (corners[1] to corners[2])
            # Edge 2: Bottom (corners[2] to corners[3])
            # Edge 3: Left (corners[3] to corners[0])
            
            edges = [
                (corners[0], corners[1]),  # Top edge
                (corners[1], corners[2]),  # Right edge
                (corners[2], corners[3]),  # Bottom edge
                (corners[3], corners[0])   # Left edge
            ]
            
            # Check which edges are touching frame boundaries
            top_touches = self._edge_touches_frame_boundary(
                edges[0], frame_width, frame_height, 'top', edge_threshold
            )
            right_touches = self._edge_touches_frame_boundary(
                edges[1], frame_width, frame_height, 'right', edge_threshold
            )
            bottom_touches = self._edge_touches_frame_boundary(
                edges[2], frame_width, frame_height, 'bottom', edge_threshold
            )
            left_touches = self._edge_touches_frame_boundary(
                edges[3], frame_width, frame_height, 'left', edge_threshold
            )
            
            # Check if BOTH parallel opposite sides touch
            vertical_edges_touch = (top_touches and bottom_touches)
            horizontal_edges_touch = (left_touches and right_touches)
            
            # Ready if EITHER both vertical OR both horizontal edges touch
            is_ready_for_capture = vertical_edges_touch or horizontal_edges_touch
            
            # Draw visual feedback for edges
            if not is_ready_for_capture:
                self._draw_edge_status(
                    frame, edges, 
                    [top_touches, right_touches, bottom_touches, left_touches],
                    frame_width, frame_height
                )
        
        # Choose color based on readiness
        if is_ready_for_capture:
            box_color = (0, 255, 0)  # Green - ready to capture
            thickness = 4
        else:
            box_color = (0, 255, 255)  # Yellow - needs adjustment
            thickness = 3
        
        # Draw four sides
        for i in range(4):
            start = tuple(corners[i])
            end = tuple(corners[(i + 1) % 4])
            cv2.line(frame, start, end, box_color, thickness)
        
        # Draw corner markers
        corner_size = 6
        for corner in corners:
            cv2.circle(frame, tuple(corner), corner_size, box_color, -1)
        
        # Add dimension annotations
        self._draw_dimension_labels(frame, corners, box_width, box_height)
        
        # Only draw "ALIGNED" stamp if ready for capture
        if is_ready_for_capture:
            self._draw_aligned_stamp(frame, corners)
        else:
            # Draw "ADJUST FRAMING" message instead
            self._draw_adjust_framing_message(frame, corners, 
                                            top_touches, right_touches, 
                                            bottom_touches, left_touches)
        
        return frame, is_ready_for_capture

    def _edge_touches_frame_boundary(self, edge, frame_width, frame_height, 
                                    boundary_side, threshold):
        """
        Check if a passport box edge touches a specific frame boundary.
        
        Args:
            edge: Tuple of (start_point, end_point) defining the edge
            frame_width: Width of the frame
            frame_height: Height of the frame
            boundary_side: Which boundary to check ('top', 'right', 'bottom', 'left')
            threshold: Distance threshold for "touching"
            
        Returns:
            True if the edge touches the specified boundary
        """
        start_point, end_point = edge
        
        # Get multiple points along the edge for better detection
        num_samples = 10
        edge_points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start_point + t * (end_point - start_point)
            edge_points.append(point)
        
        # Check if any point on the edge is close to the boundary
        touches = False
        
        if boundary_side == 'top':
            # Check if any point is near the top edge (y ≈ 0)
            min_y = min(point[1] for point in edge_points)
            touches = min_y <= threshold
            
        elif boundary_side == 'bottom':
            # Check if any point is near the bottom edge (y ≈ frame_height)
            max_y = max(point[1] for point in edge_points)
            touches = max_y >= (frame_height - threshold)
            
        elif boundary_side == 'left':
            # Check if any point is near the left edge (x ≈ 0)
            min_x = min(point[0] for point in edge_points)
            touches = min_x <= threshold
            
        elif boundary_side == 'right':
            # Check if any point is near the right edge (x ≈ frame_width)
            max_x = max(point[0] for point in edge_points)
            touches = max_x >= (frame_width - threshold)
        
        return touches

    def _draw_edge_status(self, frame, edges, touches_status, frame_width, frame_height):
        """
        Draw visual feedback showing which edges are touching and which aren't
        
        Args:
            frame: Image to draw on
            edges: List of 4 edges [(start, end), ...]
            touches_status: List of 4 booleans [top, right, bottom, left]
            frame_width: Width of frame
            frame_height: Height of frame
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        top_touches, right_touches, bottom_touches, left_touches = touches_status
        
        # Calculate midpoints of each edge
        edge_midpoints = []
        for start, end in edges:
            midpoint = ((start + end) / 2).astype(int)
            edge_midpoints.append(midpoint)
        
        # Draw status indicators for each edge
        edge_names = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT']
        
        for i, (midpoint, touches, name) in enumerate(zip(edge_midpoints, touches_status, edge_names)):
            if touches:
                color = (0, 255, 0)  # Green if touching
                symbol = "✓"
            else:
                color = (0, 0, 255)  # Red if not touching
                symbol = "✗"
            
            # Position text near the edge
            text_pos = midpoint.copy()
            
            # Adjust text position based on which edge
            if i == 0:  # Top
                text_pos[1] -= 30
            elif i == 1:  # Right
                text_pos[0] += 30
            elif i == 2:  # Bottom
                text_pos[1] += 40
            elif i == 3:  # Left
                text_pos[0] -= 80
            
            cv2.putText(frame, f"{symbol} {name}", 
                    tuple(text_pos), font, 0.5, color, 2)
        
        # Draw instruction message
        instruction_y = 60
        
        if top_touches and bottom_touches:
            cv2.putText(frame, "Top & Bottom OK - Move closer to fill width", 
                    (20, instruction_y), font, 0.6, (0, 255, 255), 2)
        elif left_touches and right_touches:
            cv2.putText(frame, "Left & Right OK - Move closer to fill height", 
                    (20, instruction_y), font, 0.6, (0, 255, 255), 2)
        else:
            # Show which pair needs work
            missing_pairs = []
            if not (top_touches and bottom_touches):
                missing_pairs.append("Top & Bottom")
            if not (left_touches and right_touches):
                missing_pairs.append("Left & Right")
            
            message = f"Need parallel edges: {' OR '.join(missing_pairs)}"
            cv2.putText(frame, message, 
                    (20, instruction_y), font, 0.6, (0, 165, 255), 2)
            
            # Show specific guidance
            if not top_touches and not bottom_touches:
                cv2.putText(frame, "→ Move closer to camera", 
                        (20, instruction_y + 30), font, 0.5, (255, 255, 255), 1)
            if not left_touches and not right_touches:
                cv2.putText(frame, "→ Move closer or center yourself", 
                        (20, instruction_y + 30), font, 0.5, (255, 255, 255), 1)

    def _draw_adjust_framing_message(self, frame, corners, 
                                    top_touches, right_touches, 
                                    bottom_touches, left_touches):
        """
        Draw "ADJUST FRAMING" message with specific guidance
        
        Args:
            frame: Image to draw on
            corners: Box corner points
            top_touches, right_touches, bottom_touches, left_touches: Edge touch status
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        center = np.mean(corners, axis=0).astype(int)
        stamp_pos = center.copy()
        stamp_pos[1] = corners[2][1] - 30
        
        # Determine what needs adjustment
        vertical_ok = top_touches and bottom_touches
        horizontal_ok = left_touches and right_touches
        
        if vertical_ok:
            text = "FILL WIDTH"
            subtext = "Move closer horizontally"
        elif horizontal_ok:
            text = "FILL HEIGHT"
            subtext = "Move closer vertically"
        else:
            text = "MOVE CLOSER"
            subtext = "Fill frame edges"
        
        font_scale = 0.7
        thickness = 2
        
        # Main text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        bg_top_left = (stamp_pos[0] - text_width // 2 - 10, 
                    stamp_pos[1] - text_height - 5)
        bg_bottom_right = (stamp_pos[0] + text_width // 2 + 10, 
                        stamp_pos[1] + 5)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, text, 
                (stamp_pos[0] - text_width // 2, stamp_pos[1]), 
                font, font_scale, (0, 255, 255), thickness)
        
        # Subtext
        subtext_pos = stamp_pos.copy()
        subtext_pos[1] += 25
        (sub_width, sub_height), _ = cv2.getTextSize(
            subtext, font, 0.4, 1
        )
        cv2.putText(frame, subtext, 
                (subtext_pos[0] - sub_width // 2, subtext_pos[1]), 
                font, 0.4, (200, 200, 200), 1)


    
    def _draw_dimension_labels(self, frame, corners, width, height):
        """
        Draw dimension labels on the passport box
        
        Args:
            frame: Image to draw on
            corners: Box corner points [top-left, top-right, bottom-right, bottom-left]
            width: Box width in pixels
            height: Box height in pixels
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        thickness = 2
        
        # Calculate ratio display (should be close to 35:45)
        ratio_width = 35
        ratio_height = 45
        
        # Top center - width dimension
        top_center = ((corners[0] + corners[1]) / 2).astype(int)
        top_center[1] -= 15
        cv2.putText(frame, f"{ratio_width}mm width", 
                   tuple(top_center - np.array([50, 0])), 
                   font, font_scale, font_color, thickness)
        
        # Right center - height dimension
        right_center = ((corners[1] + corners[2]) / 2).astype(int)
        right_center[0] += 15
        cv2.putText(frame, f"{ratio_height}mm", 
                   tuple(right_center), 
                   font, font_scale, font_color, thickness)
    
    def _draw_aligned_stamp(self, frame, corners):
        """
        Draw an "ALIGNED" stamp on the passport box
        
        Args:
            frame: Image to draw on
            corners: Box corner points
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate center of box
        center = np.mean(corners, axis=0).astype(int)
        
        # Position at bottom of box
        stamp_pos = center.copy()
        stamp_pos[1] = corners[2][1] - 30  # Near bottom edge
        
        # Draw background rectangle for stamp
        text = "ALIGNED"
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Background rectangle
        bg_top_left = (stamp_pos[0] - text_width // 2 - 10, 
                       stamp_pos[1] - text_height - 5)
        bg_bottom_right = (stamp_pos[0] + text_width // 2 + 10, 
                          stamp_pos[1] + 5)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 200, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, 
                   (stamp_pos[0] - text_width // 2, stamp_pos[1]), 
                   font, font_scale, (0, 255, 0), thickness)
    
    def draw_3d_face_box(self, frame, face_data, is_aligned=False):
        """
        Draw a 3D box around the face
        
        Args:
            frame: Image to draw on
            face_data: Face orientation data containing:
                - eye_center: Center point between eyes
                - normal: Face normal vector
                - horizontal: Face horizontal vector
                - vertical: Face vertical vector
                - eye_distance: Distance between eyes
            is_aligned: Whether face is aligned (affects color)
            
        Returns:
            Frame with 3D box drawn
        """
        eye_center = face_data['eye_center']
        normal = face_data['normal']
        horizontal = face_data['horizontal']
        vertical = face_data['vertical']
        eye_distance = face_data['eye_distance']
        
        # Box dimensions based on face size
        width = eye_distance * 2.5
        height = eye_distance * 3.5
        depth = eye_distance * 2.0
        
        # Center of the box (slightly in front of eyes)
        center = eye_center - normal * depth * 0.3
        
        # Create 8 corners of 3D box
        corners_3d = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-0.5, 0.5]:
                    corner = (center + 
                             dx * width/2 * horizontal + 
                             dy * height/2 * vertical + 
                             dz * depth * normal)
                    corners_3d.append(corner)
        
        corners_3d = np.array(corners_3d)
        
        # Project to 2D
        corners_2d = self._project_to_2d(corners_3d, frame.shape)
        
        # Choose colors based on alignment
        if is_aligned:
            front_color = (0, 255, 0)  # Green for front face
            back_color = (0, 200, 0)   # Darker green for back edges
            fill_color = (0, 255, 0)   # Green fill
        else:
            front_color = (0, 255, 255)  # Yellow for front face
            back_color = (255, 150, 0)   # Orange for back edges
            fill_color = (0, 165, 255)   # Orange fill
        
        # Draw the box
        frame = self._draw_box_edges(frame, corners_2d, front_color, back_color)
        frame = self._draw_box_fill(frame, corners_2d, fill_color)
        
        return frame
    
    def draw_simple_box(self, frame, face_data, is_aligned=False):
        """
        Draw a simple 2D bounding box around the face
        
        Args:
            frame: Image to draw on
            face_data: Face data containing landmark positions
            is_aligned: Whether face is aligned (affects color)
            
        Returns:
            Frame with simple box drawn
        """
        # Get bounding coordinates from face landmarks
        left_eye = face_data['left_eye'][:2].astype(int)
        right_eye = face_data['right_eye'][:2].astype(int)
        nose = face_data['nose'][:2].astype(int)
        mouth = face_data['mouth_center'][:2].astype(int)
        chin = face_data['chin'][:2].astype(int)
        forehead = face_data['forehead'][:2].astype(int)
        
        # Calculate bounding box
        all_points = np.array([left_eye, right_eye, nose, mouth, chin, forehead])
        x_min = np.min(all_points[:, 0]) - 20
        x_max = np.max(all_points[:, 0]) + 20
        y_min = np.min(all_points[:, 1]) - 40
        y_max = np.max(all_points[:, 1]) + 20
        
        # Choose color
        color = (0, 255, 0) if is_aligned else (0, 0, 255)
        
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        return frame
    
    def draw_face_axes(self, frame, face_data, length=100):
        """
        Draw 3D coordinate axes on the face
        
        Args:
            frame: Image to draw on
            face_data: Face orientation data
            length: Length of axis arrows
            
        Returns:
            Frame with axes drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        center = face_data['eye_center'][:2].astype(int)
        
        # X-axis (horizontal - red)
        x_end = (center + face_data['horizontal'][:2] * length).astype(int)
        cv2.arrowedLine(frame, tuple(center), tuple(x_end), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(frame, "X", tuple(x_end + 10), font, 0.6, (0, 0, 255), 2)
        
        # Y-axis (vertical - green)
        y_end = (center + face_data['vertical'][:2] * length).astype(int)
        cv2.arrowedLine(frame, tuple(center), tuple(y_end), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(frame, "Y", tuple(y_end + 10), font, 0.6, (0, 255, 0), 2)
        
        # Z-axis (normal - blue)
        z_end = (center + face_data['normal'][:2] * length).astype(int)
        cv2.arrowedLine(frame, tuple(center), tuple(z_end), (255, 0, 0), 3, tipLength=0.3)
        cv2.putText(frame, "Z", tuple(z_end + 10), font, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def draw_facial_landmarks(self, frame, face_data):
        """
        Draw key facial landmarks
        
        Args:
            frame: Image to draw on
            face_data: Face data with landmark positions
            
        Returns:
            Frame with landmarks drawn
        """
        # Draw eyes
        cv2.circle(frame, tuple(face_data['left_eye'][:2].astype(int)), 5, (255, 255, 0), -1)
        cv2.circle(frame, tuple(face_data['right_eye'][:2].astype(int)), 5, (255, 255, 0), -1)
        
        # Draw eye center
        cv2.circle(frame, tuple(face_data['eye_center'][:2].astype(int)), 5, (0, 255, 255), -1)
        
        # Draw nose
        cv2.circle(frame, tuple(face_data['nose'][:2].astype(int)), 5, (255, 0, 255), -1)
        
        # Draw mouth
        cv2.circle(frame, tuple(face_data['mouth_center'][:2].astype(int)), 5, (0, 255, 0), -1)
        
        # Draw line between eyes
        cv2.line(frame, 
                tuple(face_data['left_eye'][:2].astype(int)),
                tuple(face_data['right_eye'][:2].astype(int)),
                (255, 255, 0), 2)
        
        return frame
    
    def draw_alignment_info(self, frame, face_data, is_aligned):
        """
        Draw alignment information on frame
        
        Args:
            frame: Image to draw on
            face_data: Face orientation data
            is_aligned: Whether face is aligned
            
        Returns:
            Frame with info drawn
        """
        text_pos = face_data['forehead'][:2].astype(int)
        text_pos[1] -= 40
        text_color = (0, 255, 0) if is_aligned else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, f"Yaw: {face_data['yaw']:.1f}", 
                   tuple(text_pos), font, 0.5, text_color, 2)
        text_pos[1] += 20
        cv2.putText(frame, f"Pitch: {face_data['pitch']:.1f}", 
                   tuple(text_pos), font, 0.5, text_color, 2)
        text_pos[1] += 20
        cv2.putText(frame, f"Roll: {face_data['roll']:.1f}", 
                   tuple(text_pos), font, 0.5, text_color, 2)
        text_pos[1] += 20
        
        is_vertical = face_data['verticality_angle'] <= 20
        vert_color = (0, 255, 0) if is_vertical else (0, 0, 255)
        cv2.putText(frame, f"Vert: {face_data['verticality_angle']:.1f}", 
                   tuple(text_pos), font, 0.5, vert_color, 2)
        
        return frame
    
    def _project_to_2d(self, corners_3d, frame_shape):
        """
        Project 3D points to 2D using perspective projection
        
        Args:
            corners_3d: Array of 3D corner points
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Array of 2D corner points
        """
        focal_length = frame_shape[1]
        camera_center = np.array([frame_shape[1]/2, frame_shape[0]/2])
        
        corners_2d = []
        for corner in corners_3d:
            # Perspective projection
            if corner[2] != 0:
                scale = focal_length / (focal_length + corner[2])
            else:
                scale = 1
            
            x_2d = int(corner[0] * scale + camera_center[0] * (1 - scale))
            y_2d = int(corner[1] * scale + camera_center[1] * (1 - scale))
            corners_2d.append([x_2d, y_2d])
        
        return np.array(corners_2d)
    
    def _draw_box_edges(self, frame, corners_2d, front_color, back_color):
        """
        Draw the edges of the 3D box
        
        Args:
            frame: Image to draw on
            corners_2d: 2D corner points
            front_color: Color for front edges
            back_color: Color for back edges
            
        Returns:
            Frame with edges drawn
        """
        # Define edges of the box
        edges = [
            # Front face
            (0, 1), (0, 2), (1, 3), (2, 3),
            # Back face
            (4, 5), (4, 6), (5, 7), (6, 7),
            # Connecting edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw edges
        for start, end in edges:
            color = front_color if start < 4 and end < 4 else back_color
            cv2.line(frame, tuple(corners_2d[start]), 
                    tuple(corners_2d[end]), color, 2)
        
        return frame
    
    def _draw_box_fill(self, frame, corners_2d, fill_color):
        """
        Draw filled front face of the box with transparency
        
        Args:
            frame: Image to draw on
            corners_2d: 2D corner points
            fill_color: Color for fill
            
        Returns:
            Frame with fill drawn
        """
        # Draw filled front face with transparency
        front_face = corners_2d[[0, 1, 3, 2]]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [front_face], fill_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        return frame
