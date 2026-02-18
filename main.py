"""
Main photo quality checker.
Runs multiple checks to determine if a photo is "good".
"""

import cv2
import sys
from face_alignment import FaceAlignmentDetector
from shoulder_alignment import (ShoulderAlignmentDetector, draw_shoulder_lines, 
                                draw_shoulder_skeleton, draw_shoulder_info)
from bounding_box import BoundingBoxDrawer
from background_checker import BackgroundChecker
from passport_capture import PassportPhotoCapturer

class PhotoQualityChecker:
    """Main class for checking photo quality"""
    
    def __init__(self):
        """Initialize all quality checkers"""
        self.face_alignment_detector = FaceAlignmentDetector()
        self.shoulder_alignment_detector = ShoulderAlignmentDetector()
        self.bbox_drawer = BoundingBoxDrawer()
        self.background_checker = BackgroundChecker()  # NEW
        self.checks = []
        self.results = {}
        self.passport_capturer = PassportPhotoCapturer(output_size=(350, 450))
        self.latest_passport_ready = False
        self.latest_passport_corners = None
    
    def check_face_alignment(self, frame, 
                            yaw_threshold=15,
                            pitch_threshold=15,
                            roll_threshold=20,
                            verticality_threshold=20):
        """
        Check if faces in the photo are properly aligned
        
        Args:
            frame: Image to check
            yaw_threshold: Maximum yaw angle
            pitch_threshold: Maximum pitch angle
            roll_threshold: Maximum roll angle
            verticality_threshold: Maximum verticality angle
            
        Returns:
            Dictionary with check results
        """
        result = self.face_alignment_detector.check_alignment(
            frame,
            yaw_threshold,
            pitch_threshold,
            roll_threshold,
            verticality_threshold
        )
        
        result['check_name'] = 'Face Alignment'
        return result
    
    def check_shoulder_alignment(self, frame,
                                level_threshold=10,
                                square_threshold=0.2):
        """
        Check if shoulders in the photo are properly aligned
        
        Args:
            frame: Image to check
            level_threshold: Max shoulder tilt in degrees (default: 10)
            square_threshold: Max shoulder rotation ratio (default: 0.2)
            
        Returns:
            Dictionary with check results
        """
        result = self.shoulder_alignment_detector.check_alignment(
            frame,
            level_threshold=level_threshold,
            square_threshold=square_threshold
        )
        
        result['check_name'] = 'Shoulder Alignment'
        return result
    
    def run_all_checks(self, frame):
        """
        Run all quality checks on a frame
        
        Args:
            frame: Image to check
            
        Returns:
            Dictionary with overall results
        """
        results = {}
        
        # Check 1: Face Alignment
        alignment_result = self.check_face_alignment(frame)
        results['face_alignment'] = alignment_result
        
        # Check 2: Shoulder Alignment
        shoulder_result = self.check_shoulder_alignment(frame)
        results['shoulder_alignment'] = shoulder_result
        
        # TODO: Add more checks here
        # results['lighting'] = self.check_lighting(frame)
        # results['focus'] = self.check_focus(frame)
        # results['composition'] = self.check_composition(frame)
        
        # Overall assessment
        all_passed = (alignment_result['passed'] and 
                     shoulder_result['passed'])
        
        results['overall_passed'] = all_passed
        results['overall_message'] = self._get_overall_message(results)
        
        return results
    
    def _get_overall_message(self, results):
        """Generate overall message from all check results"""
        if results['overall_passed']:
            return "Photo is good quality!"
        
        failed_checks = []
        if not results['face_alignment']['passed']:
            failed_checks.append(f"Face: {results['face_alignment']['message']}")
        if not results['shoulder_alignment']['passed']:
            failed_checks.append(f"Posture: {results['shoulder_alignment']['message']}")
        
        return "Issues: " + " | ".join(failed_checks)
    
    def visualize_results(self, frame, results, show_details=True):
        """
        Draw visualization of check results on frame
        
        Args:
            frame: Image to draw on
            results: Results from run_all_checks()
            show_details: Whether to show detailed info
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Check if both tests passed
        both_passed = results.get('overall_passed', False)
        is_ready_for_capture = False  # NEW: Initialize the flag
        # Default to no capture info each frame
        self.latest_passport_ready = False
        self.latest_passport_corners = None
        
        if both_passed:
            # Draw ONLY the passport-style box when both checks pass
            if ('face_alignment' in results and 
                'shoulder_alignment' in results and
                results['face_alignment']['num_faces'] > 0 and
                results['shoulder_alignment']['detected']):
                
                face_data = results['face_alignment']['faces'][0]
                shoulder_data = results['shoulder_alignment']['person_data']
                
                # CHANGED: Capture the return value (tuple now)
                vis_frame, is_ready_for_capture, corners = self.bbox_drawer.draw_passport_box(
                    vis_frame,
                    face_data,
                    shoulder_data,
                    check_frame_fit=True
                )

                # Save latest state so main() can capture when user presses a button
                self.latest_passport_ready = bool(is_ready_for_capture)
                self.latest_passport_corners = corners
        else:
            # Draw detailed visualizations when checks haven't passed
            if show_details:
                # Draw shoulder alignment visualizations first (behind face)
                if 'shoulder_alignment' in results:
                    shoulder = results['shoulder_alignment']
                    
                    if shoulder['detected']:
                        person_data = shoulder['person_data']
                        
                        # Draw skeleton
                        vis_frame = draw_shoulder_skeleton(
                            vis_frame,
                            person_data,
                            person_data['is_straight']
                        )
                        
                        # Draw shoulder lines
                        vis_frame = draw_shoulder_lines(
                            vis_frame,
                            person_data,
                            person_data['is_straight']
                        )
                        
                        # Draw info
                        vis_frame = draw_shoulder_info(
                            vis_frame,
                            person_data,
                            person_data['is_straight']
                        )
                
                # Draw face alignment visualizations
                if 'face_alignment' in results:
                    alignment = results['face_alignment']
                    
                    if alignment['num_faces'] > 0:
                        for face_data in alignment['faces']:
                            # Draw 3D box using BoundingBoxDrawer
                            vis_frame = self.bbox_drawer.draw_3d_face_box(
                                vis_frame, 
                                face_data, 
                                face_data['is_aligned']
                            )
                            # Draw axes
                            vis_frame = self.bbox_drawer.draw_face_axes(
                                vis_frame,
                                face_data,
                                length=80
                            )
                            # Draw landmarks
                            vis_frame = self.bbox_drawer.draw_facial_landmarks(
                                vis_frame,
                                face_data
                            )
                            # Draw info
                            vis_frame = self.bbox_drawer.draw_alignment_info(
                                vis_frame,
                                face_data,
                                face_data['is_aligned']
                            )
        
        # Draw summary panel
        panel_height = 220
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        
        y_pos = 40
        
        # Title
        cv2.putText(vis_frame, "PHOTO QUALITY CHECK", 
                (20, y_pos), font, 0.7, (255, 255, 255), 2)
        y_pos += 35
        
        # Face alignment summary
        if 'face_alignment' in results:
            alignment = results['face_alignment']
            face_color = (0, 255, 0) if alignment['passed'] else (0, 0, 255)
            face_status = "PASS" if alignment['passed'] else "FAIL"
            
            cv2.putText(vis_frame, f"Face Alignment: {face_status}", 
                    (20, y_pos), font, 0.6, face_color, 2)
            y_pos += 25
            
            if alignment['num_faces'] > 0:
                cv2.putText(vis_frame, 
                        f"  Faces: {alignment['num_faces']} | Yaw: {alignment['avg_yaw']:.1f}deg", 
                        (20, y_pos), font, 0.5, (200, 200, 200), 1)
                y_pos += 20
        
        # Shoulder alignment summary
        if 'shoulder_alignment' in results:
            shoulder = results['shoulder_alignment']
            shoulder_color = (0, 255, 0) if shoulder['passed'] else (0, 0, 255)
            shoulder_status = "PASS" if shoulder['passed'] else "FAIL"
            
            cv2.putText(vis_frame, f"Shoulder Alignment: {shoulder_status}", 
                    (20, y_pos), font, 0.6, shoulder_color, 2)
            y_pos += 25
            
            if shoulder['detected']:
                person_data = shoulder['person_data']
                cv2.putText(vis_frame, 
                        f"  Tilt: {person_data['shoulder_tilt']:.1f}deg | " +
                        f"Depth: {person_data['shoulder_depth_ratio']:.2f}", 
                        (20, y_pos), font, 0.5, (200, 200, 200), 1)
                y_pos += 20
        
        # Overall result - UPDATED LOGIC
        y_pos += 10
        
        # NEW: Three-tier status system
        if both_passed and is_ready_for_capture:
            # TIER 1: Everything perfect - ready to capture
            overall_color = (0, 255, 0)
            overall_status = "READY TO CAPTURE"
        elif both_passed:
            # TIER 2: Aligned but framing needs adjustment
            overall_color = (0, 255, 255)
            overall_status = "ALIGNED - ADJUST FRAMING"
        else:
            # TIER 3: Alignment issues
            overall_color = (0, 0, 255)
            overall_status = "NEEDS ADJUSTMENT"
        
        cv2.putText(vis_frame, f"OVERALL: {overall_status}", 
                (20, y_pos), font, 0.7, overall_color, 2)
        
        # Detailed message at bottom - UPDATED
        h = vis_frame.shape[0]
        
        if both_passed and is_ready_for_capture:
            # Perfect - ready for capture
            cv2.putText(vis_frame, "*** READY FOR PASSPORT PHOTO ***", 
                    (20, h - 20), font, 0.8, (0, 255, 0), 3)
        elif both_passed:
            # Aligned but needs framing adjustment
            cv2.putText(vis_frame, "Aligned! Move closer to fill frame edges", 
                    (20, h - 20), font, 0.7, (0, 255, 255), 2)
        else:
            # Has alignment issues
            cv2.putText(vis_frame, results['overall_message'], 
                    (20, h - 20), font, 0.6, overall_color, 2)
        
        return vis_frame

    
    def cleanup(self):
        """Clean up resources"""
        self.face_alignment_detector.close()
        self.shoulder_alignment_detector.close()


def main():
    """Main function for live camera feed"""
    checker = PhotoQualityChecker()
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("PHOTO QUALITY CHECKER")
    print("=" * 60)
    print("Checks performed:")
    print("  1. Face Alignment (yaw, pitch, roll, verticality)")
    print("  2. Shoulder Alignment (level, square)")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  's' - Save current frame with results")
    print("  'd' - Toggle detailed view")
    print("=" * 60)
    
    frame_count = 0
    show_details = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Run checks
        results = checker.run_all_checks(frame)
        
        # Visualize
        vis_frame = checker.visualize_results(frame, results, show_details=show_details)
        
        # Display
        cv2.imshow("Photo Quality Checker", vis_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == ord('s'):  # Save
            filename = f"quality_check_{frame_count}.jpg"
            cv2.imwrite(filename, vis_frame)
            print(f"\nSaved: {filename}")
            print(f"Results:")
            print(f"  Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
            print(f"  Face: {'PASS' if results['face_alignment']['passed'] else 'FAIL'}")
            print(f"  Shoulder: {'PASS' if results['shoulder_alignment']['passed'] else 'FAIL'}")
            frame_count += 1
        elif key == ord('d'):  # Toggle details
            show_details = not show_details
            print(f"Detailed view: {'ON' if show_details else 'OFF'}")
        elif key == ord('p'):
            if checker.latest_passport_corners is not None:
                passport_img = checker.passport_capturer.extract_passport_photo(
                    frame,  # IMPORTANT: use the clean frame, not vis_frame (which has drawings)
                    checker.latest_passport_corners
                )

                filename = f"passport_{frame_count}.png"
                cv2.imwrite(filename, passport_img)
                cv2.imshow("Passport Photo", passport_img)
                print(f"\nCaptured passport photo: {filename}")
                frame_count += 1
            else:
                print("\nNot ready: wait for 'READY TO CAPTURE' before pressing 'p'.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    checker.cleanup()


def check_image_file(image_path):
    """
    Check quality of a single image file
    
    Args:
        image_path: Path to image file
    """
    checker = PhotoQualityChecker()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run checks
    results = checker.run_all_checks(frame)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"QUALITY CHECK RESULTS")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    print(f"\nOVERALL: {'PASSED ✓' if results['overall_passed'] else 'FAILED ✗'}")
    print(f"Message: {results['overall_message']}")
    
    print(f"\n{'-'*60}")
    print(f"FACE ALIGNMENT: {'PASS ✓' if results['face_alignment']['passed'] else 'FAIL ✗'}")
    print(f"{'-'*60}")
    print(f"  Faces detected: {results['face_alignment']['num_faces']}")
    if results['face_alignment']['num_faces'] > 0:
        print(f"  All aligned: {results['face_alignment']['all_aligned']}")
        print(f"  Avg Yaw: {results['face_alignment']['avg_yaw']:.2f}°")
        print(f"  Avg Pitch: {results['face_alignment']['avg_pitch']:.2f}°")
        print(f"  Avg Roll: {results['face_alignment']['avg_roll']:.2f}°")
        print(f"  Avg Verticality: {results['face_alignment']['avg_verticality']:.2f}°")
        print(f"  Message: {results['face_alignment']['message']}")
    
    print(f"\n{'-'*60}")
    print(f"SHOULDER ALIGNMENT: {'PASS ✓' if results['shoulder_alignment']['passed'] else 'FAIL ✗'}")
    print(f"{'-'*60}")
    print(f"  Person detected: {results['shoulder_alignment']['detected']}")
    if results['shoulder_alignment']['detected']:
        person_data = results['shoulder_alignment']['person_data']
        print(f"  Shoulders level: {person_data['is_level']}")
        print(f"  Shoulders square: {person_data['is_square']}")
        print(f"  Shoulder tilt: {person_data['shoulder_tilt']:.2f}°")
        print(f"  Depth ratio: {person_data['shoulder_depth_ratio']:.2f}")
        print(f"  Message: {results['shoulder_alignment']['message']}")
    
    print(f"\n{'='*60}\n")
    
    # Show visualization
    vis_frame = checker.visualize_results(frame, results)
    cv2.imshow("Quality Check Result", vis_frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    checker.cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check image file
        check_image_file(sys.argv[1])
    else:
        # Run live camera
        main()
