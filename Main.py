import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import time
from collections import Counter, deque # Added deque for history
from datetime import datetime, timedelta
import os

class EmotionCognitiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Emotion & Cognitive State Detection")
        self.root.geometry("850x650")
        self.root.configure(bg="#f0f2f6")

        # --- For Temporal Smoothing ---
        self.HISTORY_LENGTH = 7 # Number of frames for smoothing history
        self.emotion_history = deque(maxlen=self.HISTORY_LENGTH)
        self.cognitive_state_history = deque(maxlen=self.HISTORY_LENGTH)
        # ---

        try:
            model_path = r"C:\Users\prana\OneDrive\Desktop\FER\FER\FERTest3.keras" # ### USER: PLEASE VERIFY THIS PATH ###' # ### USER: PLEASE VERIFY THIS PATH ###
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found at: {model_path}\nPlease ensure the path is correct.")
                self.root.quit()
                return
            self.emotion_model = load_model(model_path)
            self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            self.cognitive_labels = ['Attentive', 'Distracted', 'Drowsy']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.quit()
            return

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self.face_mesh_initialized = False
        try:
            # Added min_tracking_confidence for consistency
            self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, 
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5) 
            self.face_mesh_initialized = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize MediaPipe FaceMesh: {e}")
            self.root.quit()
            return

        self.is_running = False
        self.mode = None
        self.cap = None
        self.video_path = None
        self.video_time = 0
        self.cognitive_states_report = [] # Data for report
        self.emotions_report = []         # Data for report
        self.session_start_time = None
        self.total_frames_processed = 0
        self.output_dir = "analysis_reports"
        self.current_video_path_in_folder = None

        self.setup_ui()

    def setup_ui(self):
        self.label = tk.Label(self.root, text="Emotion & Cognitive State Detection",
                              font=("Arial", 20, "bold"), bg="#f0f2f6", fg="#1f77b4")
        self.label.pack(pady=15)

        button_frame = tk.Frame(self.root, bg="#f0f2f6")
        button_frame.pack(pady=10)

        self.webcam_button = tk.Button(button_frame, text="Start Webcam Analysis",
                                       font=("Arial", 12), bg="#4CAF50", fg="white",
                                       command=self.start_webcam, width=25, height=2)
        self.webcam_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.video_button = tk.Button(button_frame, text="Select Video for Analysis",
                                      font=("Arial", 12), bg="#4CAF50", fg="white",
                                      command=self.select_video, width=25, height=2)
        self.video_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.folder_button = tk.Button(button_frame, text="Select Folder for Analysis",
                                       font=("Arial", 12), bg="#4CAF50", fg="white",
                                       command=self.select_folder, width=25, height=2)
        self.folder_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.instructions = tk.Label(self.root,
                                     text="Instructions:\n• Choose an analysis mode above.\n• In video/webcam window, press 'Q' to quit current analysis.\n  (In folder mode, 'Q' quits current video, proceeds to next).\n• For folder analysis, reports are saved automatically.",
                                     font=("Arial", 11), bg="#e6f3ff", fg="#333",
                                     relief="solid", bd=1, padx=10, pady=10)
        self.instructions.pack(pady=15, fill=tk.X, padx=20)

        report_frame = tk.Frame(self.root, bg="#f0f2f6")
        report_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(report_frame, text="Analysis Report", font=("Arial", 16, "bold"),
                 bg="#f0f2f6", fg="#1f77b4").pack()

        self.report_text = scrolledtext.ScrolledText(report_frame, height=15, width=90,
                                                     font=("Consolas", 10), bg="#ffffff",
                                                     fg="#333333", wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.clear_button = tk.Button(report_frame, text="Clear Report Display",
                                      font=("Arial", 12), bg="#ff6b6b", fg="white",
                                      command=self.clear_report, width=20)
        self.clear_button.pack(pady=5)

    def clear_report(self):
        self.report_text.delete(1.0, tk.END)
        # Clear histories as well when clearing report for a fresh display session if needed
        self.emotion_history.clear()
        self.cognitive_state_history.clear()


    def calculate_ear(self, eye_landmarks):
        try:
            # Ensure eye_landmarks has the expected 6 points
            if len(eye_landmarks) != 6:
                return 0 # Invalid input

            v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            return (v1 + v2) / (2.0 * h) if h != 0 else 0
        except Exception:
            return 0

    def get_cognitive_state(self, landmarks):
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            nose_tip_idx = 0 # As per user's working "first code"
            face_width_lm_left = 33
            face_width_lm_right = 263
            
            max_required_idx = max(left_eye_indices + right_eye_indices + [nose_tip_idx, face_width_lm_left, face_width_lm_right])
            if len(landmarks) <= max_required_idx:
                return 'N/A (Lm Insufficient)'

            left_eye = [landmarks[i] for i in left_eye_indices]
            right_eye = [landmarks[i] for i in right_eye_indices]
            
            avg_ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0

            if avg_ear < 0.2:
                return 'Drowsy'
            
            nose_tip = landmarks[nose_tip_idx]
            face_width = abs(landmarks[face_width_lm_left][0] - landmarks[face_width_lm_right][0])
            
            if face_width == 0: return 'Attentive'

            eye_center_x = (landmarks[face_width_lm_left][0] + landmarks[face_width_lm_right][0]) / 2.0
            gaze_offset = (nose_tip[0] - eye_center_x) / face_width
            
            return 'Attentive' if abs(gaze_offset) < 0.1 else 'Distracted'
        except IndexError:
            return 'N/A (Lm Idx Error)'
        except Exception:
            return 'N/A (Cog Error)'

    def get_emotion(self, face_img):
        try:
            if face_img.size == 0:
                return 'N/A (No Face)'
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img_resized = cv2.resize(face_img_gray, (48, 48))
            face_img_normalized = face_img_resized / 255.0
            face_img_expanded = np.expand_dims(face_img_normalized, axis=[0, -1])
            pred = self.emotion_model.predict(face_img_expanded, verbose=0)
            return self.emotion_labels[np.argmax(pred)]
        except cv2.error:
            return 'N/A (CV2 Error)'
        except Exception:
            return 'N/A (Emo Error)'

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame) if self.face_mesh_initialized else None
            
            current_emotion_to_display = Counter(self.emotion_history).most_common(1)[0][0] if self.emotion_history else 'N/A'
            current_cognitive_to_display = Counter(self.cognitive_state_history).most_common(1)[0][0] if self.cognitive_state_history else 'N/A'
            face_detected_this_frame = False

            if results and results.multi_face_landmarks:
                face_detected_this_frame = True # MediaPipe detected a face
                for face_landmarks_mp in results.multi_face_landmarks:
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                                 for lm in face_landmarks_mp.landmark]
                    
                    # Max index used by get_cognitive_state, ensure landmarks list is sufficient
                    if not landmarks or len(landmarks) < 388: # A high landmark index used by right_eye
                        # If landmarks are insufficient for this face, we might skip detailed processing for it
                        # or rely on error handling in get_cognitive_state/get_emotion
                        continue # Or handle as a case where this specific face detection wasn't usable

                    x_coords, y_coords = zip(*landmarks)
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(frame.shape[1] - 1, x_max)
                    y_max = min(frame.shape[0] - 1, y_max)

                    if x_min < x_max and y_min < y_max: # Valid bounding box for face
                        face_img = frame[y_min:y_max, x_min:x_max]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Draw rectangle

                        emotion_raw = self.get_emotion(face_img)
                        cognitive_state_raw = self.get_cognitive_state(landmarks)
                        
                        # Store raw data for report if valid
                        if not (cognitive_state_raw == 'N/A' or cognitive_state_raw.startswith('N/A')):
                            self.cognitive_states_report.append(cognitive_state_raw)
                            self.cognitive_state_history.append(cognitive_state_raw)
                        if not (emotion_raw == 'N/A' or emotion_raw.startswith('N/A')):
                            self.emotions_report.append(emotion_raw)
                            self.emotion_history.append(emotion_raw)
                        
                        # Update display values from history (smoothed) or current raw if history is empty
                        current_emotion_to_display = Counter(self.emotion_history).most_common(1)[0][0] if self.emotion_history else emotion_raw
                        current_cognitive_to_display = Counter(self.cognitive_state_history).most_common(1)[0][0] if self.cognitive_state_history else cognitive_state_raw
                        
                        self.total_frames_processed += 1
                    # else: an invalid bounding box from landmarks, likely get_emotion will return N/A if face_img is empty/bad
                      # current_emotion_to_display and current_cognitive_to_display would retain their previous values or history-based defaults
            
            # If no face was detected by MediaPipe in the entire frame
            if not face_detected_this_frame:
                current_emotion_to_display = "No face detected"
                current_cognitive_to_display = "Distracted"
                
                # Add "Distracted" to cognitive history and report as it's a valid state
                self.cognitive_state_history.append("Distracted") 
                self.cognitive_states_report.append("Distracted")
                # "No face detected" for emotion is for display only, not added to history/report
                # to maintain existing emotion label set and report structure.
            
            # Draw analysis results with semi-transparent background
            text_bg_h, text_bg_w = 100, 450 
            if frame.shape[0] > 10 + text_bg_h and frame.shape[1] > 10 + text_bg_w: # Ensure overlay fits
                overlay_area = frame[10 : 10 + text_bg_h, 10 : 10 + text_bg_w]
                text_bg = np.zeros_like(overlay_area, dtype=np.uint8) # Black background for text
                alpha = 0.6 # Transparency level
                blended_overlay = cv2.addWeighted(overlay_area, 1 - alpha, text_bg, alpha, 0)
                frame[10 : 10 + text_bg_h, 10 : 10 + text_bg_w] = blended_overlay

            cv2.putText(frame, f'Emotion: {current_emotion_to_display}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Cognitive: {current_cognitive_to_display}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            is_video_or_folder_mode = self.mode == 'video' or (self.mode == 'folder' and self.current_video_path_in_folder)
            if is_video_or_folder_mode:
                cv2.putText(frame, 'S:Skip A:Back Spc:Pause Q:Quit Video', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            elif self.mode == 'webcam':
                cv2.putText(frame, 'Q:Quit Analysis', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            return frame
        except Exception as e:
            # print(f"Error in process_frame: {e}") # Optional: for debugging
            # Return the original frame to prevent the stream from crashing
            return frame

    def generate_comprehensive_report(self):
        if self.session_start_time:
            session_duration = datetime.now() - self.session_start_time
            duration_str = str(session_duration).split('.')[0]
        else:
            duration_str = "Unknown"

        report = f"""
{'='*80}
                               COMPREHENSIVE ANALYSIS REPORT
{'='*80}

Session Information:
  • Analysis Mode: {self.mode.upper() if self.mode else 'Unknown'}
"""
        if self.mode == 'folder' and self.current_video_path_in_folder:
            report += f"  • Video File: {os.path.basename(self.current_video_path_in_folder)}\n"
        elif self.mode == 'video' and self.video_path:
            report += f"  • Video File: {os.path.basename(self.video_path)}\n"
        
        report += f"""  • Session Duration: {duration_str}
  • Total Frames Analyzed (where face processing was attempted): {self.total_frames_processed:,}
  • Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
                                     COGNITIVE STATE ANALYSIS (RAW DATA)
{'='*80}
"""
        if self.cognitive_states_report:
            state_counts = Counter(self.cognitive_states_report)
            total_cognitive = len(self.cognitive_states_report)
            report += f"Total Raw Cognitive State Detections: {total_cognitive:,}\n\n"
            # Ensure all labels from self.cognitive_labels are reported, even if count is 0
            all_possible_cognitive_states = self.cognitive_labels + [s for s in state_counts if s not in self.cognitive_labels and not s.startswith('N/A')]

            for state in sorted(list(set(all_possible_cognitive_states))): # Use sorted unique states found + predefined
                count = state_counts.get(state, 0)
                percentage = (count / total_cognitive * 100) if total_cognitive > 0 else 0
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                report += f"  {state:12} │ {count:6,} │ {percentage:6.2f}% │ {bar}\n"
            dominant_cognitive = state_counts.most_common(1)[0][0] if state_counts else "None"
            report += f"\nDominant Raw Cognitive State: {dominant_cognitive}\n"
        else:
            report += "No raw cognitive states were detected/recorded for this session.\n"

        report += f"\n{'='*80}\n"
        report += "                                     EMOTION ANALYSIS (RAW DATA)\n"
        report += f"{'='*80}\n"

        if self.emotions_report:
            emotion_counts = Counter(self.emotions_report)
            total_emotions = len(self.emotions_report)
            report += f"Total Raw Emotion Detections: {total_emotions:,}\n\n"
            # Ensure all labels from self.emotion_labels are reported
            all_possible_emotion_states = self.emotion_labels + [s for s in emotion_counts if s not in self.emotion_labels and not s.startswith('N/A')]

            for emotion in sorted(list(set(all_possible_emotion_states))): # Use sorted unique states found + predefined
                count = emotion_counts.get(emotion, 0)
                percentage = (count / total_emotions * 100) if total_emotions > 0 else 0
                bar_length = int(percentage / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                report += f"  {emotion:12} │ {count:6,} │ {percentage:6.2f}% │ {bar}\n"
            dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "None"
            report += f"\nDominant Raw Emotion: {dominant_emotion}\n"
        else:
            report += "No raw emotions were detected/recorded for this session.\n"

        report += f"\n{'='*80}\n"
        report += "                                     PERFORMANCE METRICS\n"
        report += f"{'='*80}\n"
        
        if self.session_start_time and self.total_frames_processed > 0:
            elapsed_seconds = (datetime.now() - self.session_start_time).total_seconds()
            fps = self.total_frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0
            report += f"  • Average Processing Rate: {fps:.2f} FPS (based on frames where face processing was attempted)\n"
        
        cog_detections = len(self.cognitive_states_report)
        emo_detections = len(self.emotions_report)
        report += f"  • Raw Cognitive Data Points Collected: {cog_detections}\n"
        report += f"  • Raw Emotion Data Points Collected: {emo_detections}\n"

        report += f"\n{'='*80}\n"
        report += "                                     INSIGHTS & SUMMARY (BASED ON RAW DATA)\n"
        report += f"{'='*80}\n"
        
        if self.cognitive_states_report: 
            cog_counts = Counter(self.cognitive_states_report)
            total_valid_cog_detections = sum(cog_counts[s] for s in self.cognitive_labels if s in cog_counts) # only count valid labels for percentage
            if total_valid_cog_detections > 0:
                attentive_percentage = (cog_counts.get('Attentive', 0) / total_valid_cog_detections * 100)
                report += f"  • Attention Level (Attentive state): {attentive_percentage:.1f}% of detected valid raw cognitive states\n"
                if attentive_percentage > 70:
                    report += "  • Assessment: High focus and engagement levels detected.\n"
                elif attentive_percentage > 40:
                    report += "  • Assessment: Moderate focus with some distraction periods.\n"
                else:
                    report += "  • Assessment: Low focus levels, potential for disengagement.\n"
            else:
                report += "  Insufficient valid cognitive data for attention insights.\n"
        else:
            report += "  Insufficient cognitive data for attention insights.\n"

        if self.emotions_report: 
            emo_counts = Counter(self.emotions_report)
            total_valid_emo_detections = sum(emo_counts[s] for s in self.emotion_labels if s in emo_counts) # only count valid labels
            if total_valid_emo_detections > 0 :
                positive_emotions_labels = ['Happy', 'Surprise', 'Neutral']
                positive_count = sum(emo_counts.get(emotion, 0) for emotion in positive_emotions_labels)
                positive_percentage = (positive_count / total_valid_emo_detections * 100)
                report += f"  • Positive Emotional State (Happy, Surprise, Neutral): {positive_percentage:.1f}% of detected valid raw emotions\n"
            else:
                report += "  Insufficient valid emotion data for positivity insights.\n"
        else:
            report += "  Insufficient emotion data for positivity insights.\n"


        report += f"\n{'='*80}\n"
        report += "                                         END OF REPORT\n"
        report += f"{'='*80}\n\n"
        return report

    def display_report_in_gui(self, report_content):
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report_content)
        self.report_text.see(1.0)

    def _reinitialize_face_mesh(self):
        if self.face_mesh_initialized and self.face_mesh:
            try: self.face_mesh.close()
            except: pass 
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, 
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)
            self.face_mesh_initialized = True
        except Exception as e:
            self.face_mesh_initialized = False
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to re-initialize MediaPipe FaceMesh: {e}"))
            return False
        return True

    def _reset_session_data(self):
        """Resets data for a new analysis session."""
        self.cognitive_states_report = []
        self.emotions_report = []
        self.emotion_history.clear()
        self.cognitive_state_history.clear()
        self.total_frames_processed = 0
        self.session_start_time = datetime.now()
        self.current_video_path_in_folder = None
        self.video_path = None


    def start_webcam(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Analysis is already running!")
            return
        
        if not self._reinitialize_face_mesh(): return

        self.mode = 'webcam'
        self.is_running = True
        self._reset_session_data() # Reset all relevant data
        
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            for i in range(1, 4): # Try other camera indices
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened(): break
        
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            self.is_running = False 
            self.mode = None
            return
        threading.Thread(target=self.webcam_loop, daemon=True).start()

    def select_video(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Analysis is already running!")
            return
        video_path_selected = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if video_path_selected:
            if not self._reinitialize_face_mesh(): return

            self.mode = 'video'
            self.is_running = True
            self._reset_session_data()
            self.video_path = video_path_selected 
            self.video_time = 0 
            
            threading.Thread(target=self._process_one_video_in_loop, kwargs={'is_part_of_folder': False}, daemon=True).start()

    def select_folder(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Analysis is already running!")
            return
        folder_path_selected = filedialog.askdirectory()
        if folder_path_selected:
            self.mode = 'folder'
            self.is_running = True 
            os.makedirs(self.output_dir, exist_ok=True)
            threading.Thread(target=self.folder_processing_loop, args=(folder_path_selected,), daemon=True).start()

    def webcam_loop(self):
        try:
            window_title = "Webcam Analysis - Press Q to Quit Analysis"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to capture frame from webcam."))
                    break
                
                frame_processed = self.process_frame(frame.copy())
                cv2.imshow(window_title, frame_processed)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False 
                    break 
            
        except Exception as e:
            self.root.after(0, lambda err=str(e): messagebox.showerror("Error", f"Webcam loop error: {err}"))
        finally:
            self._finalize_session_and_report()


    def _process_one_video_in_loop(self, is_part_of_folder=False):
        window_title = "Video Analysis" 
        if not self.video_path:
            self.root.after(0, lambda: messagebox.showerror("Error", "Video path not set for processing."))
            if not is_part_of_folder: self._finalize_session_and_report() 
            return

        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.root.after(0, lambda vp=self.video_path: messagebox.showerror("Error", f"Failed to open video: {os.path.basename(vp)}"))
                if not is_part_of_folder: self._finalize_session_and_report()
                return
            
            video_filename_display = os.path.basename(self.video_path)
            qt_action = "Quit Video" if is_part_of_folder else "Quit Analysis"
            window_title = f"Analyzing: {video_filename_display} - S:Skip A:Back Spc:Pause Q:{qt_action}"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames_video = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration_sec = (total_frames_video / fps) if fps > 0 and total_frames_video > 0 else 0
            
            current_video_processing_active = True 
            paused = False

            while current_video_processing_active and self.cap.isOpened():
                if not self.is_running: 
                    current_video_processing_active = False
                    break
                
                key_wait_ms = 1 if not paused else 100
                key = cv2.waitKey(key_wait_ms) & 0xFF
                
                if key == ord('q'):
                    current_video_processing_active = False 
                    if not is_part_of_folder: 
                        self.is_running = False
                    break 
                elif key == ord(' '): 
                    paused = not paused
                
                if paused:
                    time.sleep(0.05) # Reduce CPU usage while paused
                    continue

                if key == ord('s'): 
                    new_time_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC) + 10000 
                    if total_duration_sec > 0 :
                        self.cap.set(cv2.CAP_PROP_POS_MSEC, min(new_time_msec, total_duration_sec * 1000))
                    else: # If duration is unknown, allow skip but it might go past end
                         self.cap.set(cv2.CAP_PROP_POS_MSEC, new_time_msec)
                elif key == ord('a'): 
                    new_time_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC) - 10000 
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, max(new_time_msec, 0))
                
                ret, frame = self.cap.read()
                if not ret:
                    current_video_processing_active = False 
                    break
                
                frame_processed = self.process_frame(frame.copy()) 
                
                current_time_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                self.video_time = current_time_sec 
                progress = (current_time_sec / total_duration_sec) if total_duration_sec > 0 else 0
                progress = min(max(progress, 0), 1) # Clamp progress between 0 and 1
                
                bar_height = 15
                bar_y_start = frame_processed.shape[0] - bar_height - 10 
                bar_width_total = frame_processed.shape[1] - 40 
                
                if bar_y_start > 0 and bar_width_total > 0 : 
                    cv2.rectangle(frame_processed, (20, bar_y_start), (20 + bar_width_total, bar_y_start + bar_height), (50,50,50), -1)
                    cv2.rectangle(frame_processed, (20, bar_y_start), (20 + int(bar_width_total * progress), bar_y_start + bar_height), (0,255,0), -1)
                    time_text = f"{int(current_time_sec//60):02d}:{int(current_time_sec%60):02d} / {int(total_duration_sec//60):02d}:{int(total_duration_sec%60):02d}" if total_duration_sec > 0 else f"{int(current_time_sec//60):02d}:{int(current_time_sec%60):02d}"
                    if bar_y_start - 5 > 0 : 
                        cv2.putText(frame_processed, time_text, (25, bar_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                cv2.imshow(window_title, frame_processed)
        except Exception as e:
            error_msg = f"Video loop error ({os.path.basename(self.video_path if self.video_path else 'N/A')}): {e}"
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", err))
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            try: 
                cv2.destroyWindow(window_title)
            except: pass 

            if not is_part_of_folder:
                self._finalize_session_and_report()

    def folder_processing_loop(self, folder_path):
        video_files = []
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for f_name in sorted(os.listdir(folder_path)):
            if any(f_name.lower().endswith(ext) for ext in valid_extensions):
                video_files.append(os.path.join(folder_path, f_name))

        if not video_files:
            self.root.after(0, lambda: messagebox.showinfo("Info", "No video files found in the selected folder."))
            self._finalize_folder_processing(None) # Pass None as there's no last report
            return

        initial_report_message = f"Starting folder analysis. {len(video_files)} video(s) found.\nReports will be saved in '{self.output_dir}'.\nProcessing...\n"
        self.root.after(0, lambda: self.report_text.insert(tk.END, initial_report_message))
        self.root.after(0, self.report_text.see(tk.END))
        
        last_report_content_for_this_folder = None

        for video_path_item in video_files:
            if not self.is_running: 
                self.root.after(0, lambda: messagebox.showinfo("Info", "Folder processing was interrupted."))
                break 
            
            if not self._reinitialize_face_mesh(): 
                self.root.after(0, lambda: messagebox.showerror("Error", "FaceMesh re-initialization failed. Stopping folder processing."))
                self.is_running = False 
                break

            self._reset_session_data()
            self.video_path = video_path_item 
            self.current_video_path_in_folder = video_path_item 

            processing_msg = f"\nProcessing: {os.path.basename(video_path_item)}...\n"
            self.root.after(0, lambda msg=processing_msg: self.report_text.insert(tk.END, msg))
            self.root.after(0, self.report_text.see(tk.END))

            self._process_one_video_in_loop(is_part_of_folder=True) 
            
            # After current video processing ends 
            last_report_content_for_this_folder = self.generate_comprehensive_report()
            
            video_filename = os.path.basename(self.current_video_path_in_folder)
            report_filename = f"{os.path.splitext(video_filename)[0]}_analysis_report.txt"
            report_save_path = os.path.join(self.output_dir, report_filename)
            
            try:
                with open(report_save_path, "w", encoding="utf-8") as f:
                    f.write(last_report_content_for_this_folder)
                msg = f"Report for {video_filename} saved to '{report_save_path}'\n"
                self.root.after(0, lambda m=msg: self.report_text.insert(tk.END, m))
            except Exception as e:
                err_msg = f"Failed to save report for {video_filename}: {e}\n"
                self.root.after(0, lambda em=err_msg: self.report_text.insert(tk.END, em))
                self.root.after(0, lambda em_box=err_msg: messagebox.showerror("Save Error", em_box))
            finally:
                 self.root.after(0, self.report_text.see(tk.END))

        self._finalize_folder_processing(last_report_content_for_this_folder)


    def _finalize_folder_processing(self, last_report_content_for_display):
        """Called after all videos in a folder are processed or if folder processing stops."""
        self.is_running = False # Ensure master flag is off
        self.mode = None
        self.current_video_path_in_folder = None 
        
        final_msg = "\nFolder analysis completed.\n"
        self.root.after(0, lambda: self.report_text.insert(tk.END, final_msg))
        
        if last_report_content_for_display: # Display the last report from the folder session
             self.root.after(0, lambda: self.display_report_in_gui(last_report_content_for_display))
        else: # If no videos or interrupted early, show a placeholder or clear
             self.root.after(0, lambda: self.report_text.insert(tk.END, "No reports to display from this folder session or it was interrupted early.\n"))

        self.root.after(0, self.report_text.see(tk.END))
        self.root.after(0, lambda: messagebox.showinfo("Info", "Folder analysis finished."))


    def _finalize_session_and_report(self):
        """Handles cleanup and report generation for webcam and single video modes."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Ensure all cv2 windows are attempted to be closed if they were part of this mode
        # Specific window names are handled in their respective loops, but this is a fallback.
        cv2.destroyAllWindows() # Attempt to close any OpenCV windows

        self.is_running = False 
        
        # Generate and display the report only if it's not part of folder processing (which has its own report saving)
        if self.mode == 'webcam' or self.mode == 'video': # 'video' here means single video
            report_content = self.generate_comprehensive_report()
            self.root.after(0, lambda: self.display_report_in_gui(report_content))
            self.root.after(0, lambda: messagebox.showinfo("Info", f"{self.mode.capitalize()} analysis finished. Report generated in GUI."))

        self.mode = None # Reset mode after session ends

    def on_close(self):
        """Handle application closing."""
        self.is_running = False # Signal any running threads to stop
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            try: self.face_mesh.close()
            except: pass
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionCognitiveApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close) # Handle window close button
    root.mainloop()