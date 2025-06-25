# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: GazeProcessor.py
# Description: This class processes video input to detect facial landmarks and estimate
#              gaze vectors using MediaPipe. The gaze estimation results are asynchronously
#              output via a callback function. This class leverages advanced facial
#              recognition and affine transformation to map detected landmarks into a
#              3D model space, enabling precise gaze vector calculation.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import mediapipe as mp
import cv2
import time
import numpy as np
from landmarks import *
from face_model import *
from AffineTransformer import AffineTransformer
from EyeballDetector import EyeballDetector
import os

# Can be downloaded from https://developers.google.com/mediapipe/solutions/vision/face_landmarker
model_path = "./face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class GazeProcessor:
    """
    Processes video input to detect facial landmarks and estimate gaze vectors using the MediaPipe library.
    Outputs gaze vector estimates asynchronously via a provided callback function.
    """

    def __init__(self, video_file=None, camera_idx=0, callback=None, visualization_options=None, process_every_n_frames=1):
        """
        Initializes the gaze processor with optional camera settings, callback, and visualization configurations.

        Args:
        - video_file (str): Path to the video file to process. If provided, this takes precedence over camera_idx.
        - camera_idx (int): Index of the camera to be used for video capture if no video file is provided.
        - callback (function): Asynchronous callback function to output the gaze vectors.
        - visualization_options (object): Options for visual feedback on the video frame. Supports visualization options
        for calibration and tracking states.
        - process_every_n_frames (int): Process every nth frame after calibration is complete. Default is 1 (process every frame).
        """
        self.video_file = video_file
        self.camera_idx = camera_idx
        self.callback = callback
        self.vis_options = visualization_options
        self.process_every_n_frames = process_every_n_frames
        self.left_detector = EyeballDetector(DEFAULT_LEFT_EYE_CENTER_MODEL)
        self.right_detector = EyeballDetector(DEFAULT_RIGHT_EYE_CENTER_MODEL)
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        # Add timing attributes
        self.calibration_start_time = None
        self.calibration_end_time = None
        self.processing_start_time = None
        self.processing_end_time = None

    async def start(self):
        """
        Starts the video processing loop to detect facial landmarks and calculate gaze vectors.
        Continuously updates the video display and invokes callback with gaze data.
        """
        with FaceLandmarker.create_from_options(self.options) as landmarker:
            # Use video file if provided, otherwise use camera
            cap = cv2.VideoCapture(self.video_file if self.video_file else self.camera_idx)
            if not cap.isOpened():
                print(f"Error: Could not open {'video file' if self.video_file else 'camera'}.")
                return

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_file else None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Initialize video writer
            output_filename = "output.mp4"
            if self.video_file:
                base_name = os.path.splitext(os.path.basename(self.video_file))[0]
                output_filename = f"{base_name}_processed.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            # Initialize FPS calculation variables
            frame_count = 0
            calibration_frame_count = 0
            start_time = time.time()
            frame_times = []
            last_progress_update = 0
            frame_skip_counter = 0
            calibration_complete = False
            self.calibration_start_time = time.time()
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 30  # Maximum number of consecutive frame read failures before stopping

            print(f"Starting video processing...")
            if total_frames:
                print(f"Total frames: {total_frames}")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\nError: Failed to read {MAX_CONSECUTIVE_FAILURES} consecutive frames. Stopping processing.")
                        if total_frames:
                            print(f"Warning: Processed only {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                        break
                    continue

                consecutive_failures = 0  # Reset counter on successful frame read

                # Update progress for video files
                if total_frames and time.time() - last_progress_update >= 1.0:  # Update every second
                    progress = (frame_count / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}% ({frame_count}/{total_frames})", end="")
                    last_progress_update = time.time()

                # Skip frames after calibration if specified
                if calibration_complete and self.process_every_n_frames > 1:
                    frame_skip_counter += 1
                    if frame_skip_counter < self.process_every_n_frames:
                        frame_count += 1
                        continue
                    frame_skip_counter = 0

                frame_start_time = time.time()
                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if face_landmarker_result.face_landmarks:
                    lms_s = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarker_result.face_landmarks[0]])
                    lms_2 = (lms_s[:, :2] * [frame.shape[1], frame.shape[0]]).round().astype(int)

                    mp_hor_pts = [lms_s[i] for i in OUTER_HEAD_POINTS]
                    mp_ver_pts = [lms_s[i] for i in [NOSE_BRIDGE, NOSE_TIP]]
                    model_hor_pts = OUTER_HEAD_POINTS_MODEL
                    model_ver_pts = [NOSE_BRIDGE_MODEL, NOSE_TIP_MODEL]

                    at = AffineTransformer(lms_s[BASE_LANDMARKS,:], BASE_FACE_MODEL, mp_hor_pts, mp_ver_pts, model_hor_pts, model_ver_pts)

                    indices_for_left_eye_center_detection = LEFT_IRIS + ADJACENT_LEFT_EYELID_PART
                    left_eye_iris_points = lms_s[indices_for_left_eye_center_detection, :]
                    left_eye_iris_points_in_model_space = [at.to_m2(mpp) for mpp in left_eye_iris_points]
                    self.left_detector.update(left_eye_iris_points_in_model_space, timestamp_ms)

                    indices_for_right_eye_center_detection = RIGHT_IRIS + ADJACENT_RIGHT_EYELID_PART
                    right_eye_iris_points = lms_s[indices_for_right_eye_center_detection, :]
                    right_eye_iris_points_in_model_space = [at.to_m2(mpp) for mpp in right_eye_iris_points]
                    self.right_detector.update(right_eye_iris_points_in_model_space, timestamp_ms)

                    left_gaze_vector, right_gaze_vector = None, None

                    if self.left_detector.center_detected:
                        left_eyeball_center = at.to_m1(self.left_detector.eye_center)
                        left_pupil = lms_s[LEFT_PUPIL]
                        left_gaze_vector = left_pupil - left_eyeball_center
                        left_proj_point = left_pupil + left_gaze_vector*5.0

                    if self.right_detector.center_detected:
                        right_eyeball_center = at.to_m1(self.right_detector.eye_center)
                        right_pupil = lms_s[RIGHT_PUPIL]
                        right_gaze_vector = right_pupil - right_eyeball_center
                        right_proj_point = right_pupil + right_gaze_vector*5.0

                    # Check if calibration is complete
                    if not calibration_complete:
                        calibration_frame_count += 1
                        if self.left_detector.center_detected and self.right_detector.center_detected:
                            calibration_complete = True
                            self.calibration_end_time = time.time()
                            self.processing_start_time = time.time()
                            calibration_time = self.calibration_end_time - self.calibration_start_time
                            print(f"\nCalibration complete! Frames used for calibration: {calibration_frame_count}")
                            print(f"Calibration time: {calibration_time:.2f} seconds")

                    if self.callback and (left_gaze_vector is not None or right_gaze_vector is not None):
                        await self.callback(left_gaze_vector, right_gaze_vector, self.video_file, frame_count)

                    if self.vis_options:
                        if self.left_detector.center_detected and self.right_detector.center_detected:
                            p1 = relative(left_pupil[:2], frame.shape)
                            p2 = relative(left_proj_point[:2], frame.shape)
                            frame = cv2.line(frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)
                            p1 = relative(right_pupil[:2], frame.shape)
                            p2 = relative(right_proj_point[:2], frame.shape)
                            frame = cv2.line(frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)
                        else:
                            text_location = (10, frame.shape[0] - 10)
                            cv2.putText(frame, "Calibration...", text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.vis_options.color, 2)

                # Calculate and store frame processing time
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                # Write frame to output video
                out.write(frame)

                cv2.imshow('LaserGaze', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    print("\nProcessing stopped by user.")
                    break

            # Calculate and display final statistics
            self.processing_end_time = time.time()
            total_time = time.time() - start_time
            processing_time = self.processing_end_time - self.processing_start_time if self.processing_start_time else 0
            calibration_time = self.calibration_end_time - self.calibration_start_time if self.calibration_end_time else 0
            
            avg_fps = frame_count / total_time if total_time > 0 else 0
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {frame_count}")
            if total_frames:
                print(f"Expected frames: {total_frames}")
                print(f"Processing completion: {frame_count/total_frames*100:.1f}%")
            print(f"Frames used for calibration: {calibration_frame_count}")
            print(f"Calibration time: {calibration_time:.2f} seconds")
            print(f"Processing time (after calibration): {processing_time:.2f} seconds")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average frame processing time: {avg_frame_time*1000:.2f}ms")
            print(f"Output video saved as: {output_filename}")

            cap.release()
            out.release()
            cv2.destroyAllWindows()