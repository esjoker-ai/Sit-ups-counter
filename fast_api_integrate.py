import cv2
import mediapipe as mp
import math
import numpy as np
import time
from collections import deque
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import threading
import asyncio
import gc

# Initialize FastAPI app
app = FastAPI()

# ─────────────────────────────────────────────────────────────
# Enhanced Pose Detector Class with Multiple Detection Strategies
# ─────────────────────────────────────────────────────────────
class EnhancedPoseDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # Primary detector with optimized settings for low light
        self.primary_pose = self.mpPose.Pose(
            model_complexity=2,  # Use more complex model
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.3,  # Lower detection confidence for low light
            min_tracking_confidence=0.3  # Lower tracking confidence
        )
        
        # Fallback detector with different settings
        self.fallback_pose = self.mpPose.Pose(
            model_complexity=1,  # Use simpler model as fallback
            smooth_landmarks=False,  # Disable smoothing for fallback
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.2,  # Very low confidence for difficult conditions
            min_tracking_confidence=0.2
        )
        
        self.current_detector = self.primary_pose
        self.detection_failures = 0
        self.max_failures = 10
        self.frame_count = 0
        
        # Drawing specifications
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=4)

    def preprocess_image(self, img):
        """Enhance image for better detection in low light conditions"""
        try:
            # Convert to LAB color space for better lighting compensation
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return img

    def findPose(self, img, draw=True):
        try:
            # Preprocess image for better detection
            enhanced_img = self.preprocess_image(img)
            
            # Try primary detector first
            imgRGB = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            self.results = self.current_detector.process(imgRGB)
            
            # If primary detector fails, try fallback
            if not self.results.pose_landmarks:
                self.detection_failures += 1
                if self.detection_failures > self.max_failures:
                    self.current_detector = self.fallback_pose
                    self.detection_failures = 0
                
                # Try fallback detector
                self.results = self.fallback_pose.process(imgRGB)
                
                # If fallback also fails, try with original image
                if not self.results.pose_landmarks:
                    imgRGB_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.results = self.fallback_pose.process(imgRGB_orig)
            else:
                # Reset to primary detector if it's working
                if self.current_detector != self.primary_pose:
                    self.current_detector = self.primary_pose
                    self.detection_failures = 0
            
            if self.results.pose_landmarks and draw:
                self.drawCustomLandmarks(img)
                self.drawCustomConnections(img)
            
            # Memory management - clear results periodically
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                gc.collect()
            
            return img
            
        except Exception as e:
            print(f"Pose detection error: {e}")
            self.results = None
            return img
    
    def drawCustomLandmarks(self, img):
        try:
            h, w, c = img.shape
            if self.results and self.results.pose_landmarks:
                # Define the landmarks we want to show (head, shoulders, hips, knees)
                target_landmarks = [0,  # nose
                                   11, 12,  # shoulders
                                   23, 24,  # hips
                                   25, 26]  # knees
                
                for id in target_landmarks:
                    if id < len(self.results.pose_landmarks.landmark):
                        lm = self.results.pose_landmarks.landmark[id]
                        if lm.visibility > 0.3:  # Lower visibility threshold
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # Color code based on visibility
                            color = (0, 255, 0) if lm.visibility > 0.7 else (0, 165, 255) if lm.visibility > 0.5 else (0, 0, 255)
                            cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)
        except Exception as e:
            print(f"Landmark drawing error: {e}")
    
    def drawCustomConnections(self, img):
        try:
            h, w, c = img.shape
            # Define the connections we want to show
            connections = [
                (11, 12),  # shoulder to shoulder
                (11, 23),  # left shoulder to left hip
                (12, 24),  # right shoulder to right hip
                (23, 24),  # hip to hip
                (23, 25),  # left hip to left knee
                (24, 26)   # right hip to right knee
            ]
            
            if self.results and self.results.pose_landmarks:
                for connection in connections:
                    start_idx, end_idx = connection
                    if (len(self.results.pose_landmarks.landmark) > max(start_idx, end_idx) and
                        self.results.pose_landmarks.landmark[start_idx].visibility > 0.3 and
                        self.results.pose_landmarks.landmark[end_idx].visibility > 0.3):
                        
                        start_point = (
                            int(self.results.pose_landmarks.landmark[start_idx].x * w),
                            int(self.results.pose_landmarks.landmark[start_idx].y * h)
                        )
                        end_point = (
                            int(self.results.pose_landmarks.landmark[end_idx].x * w),
                            int(self.results.pose_landmarks.landmark[end_idx].y * h)
                        )
                        
                        # Color code based on average visibility
                        avg_visibility = (self.results.pose_landmarks.landmark[start_idx].visibility + 
                                       self.results.pose_landmarks.landmark[end_idx].visibility) / 2
                        if avg_visibility > 0.7:
                            color = (0, 255, 0)
                        elif avg_visibility > 0.5:
                            color = (0, 165, 255)
                        else:
                            color = (0, 0, 255)
                        
                        cv2.line(img, start_point, end_point, color, 2)
        except Exception as e:
            print(f"Connection drawing error: {e}")
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        try:
            if self.results and self.results.pose_landmarks:
                h, w, c = img.shape
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    visibility = lm.visibility
                    self.lmList.append([id, cx, cy, visibility])
                    if draw and visibility > 0.3:
                        # Only draw the landmarks we want (head, shoulders, hips, knees)
                        if id in [0, 11, 12, 23, 24, 25, 26]:
                            # Color code based on visibility
                            color = (0, 255, 0) if visibility > 0.7 else (0, 165, 255) if visibility > 0.5 else (0, 0, 255)
                            cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)
        except Exception as e:
            print(f"Position finding error: {e}")
            self.lmList = []
        
        return self.lmList
        
    def findAngle(self, img, p1, p2, p3, draw=True):   
        try:
            if len(self.lmList) < max(p1, p2, p3) + 1:
                return 0
                
            # Lower visibility threshold for angle calculation
            if (self.lmList[p1][3] < 0.3 or 
                self.lmList[p2][3] < 0.3 or 
                self.lmList[p3][3] < 0.3):
                return 0
                
            x1, y1 = self.lmList[p1][1:3]
            x2, y2 = self.lmList[p2][1:3]
            x3, y3 = self.lmList[p3][1:3]
            
            angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                                 math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle += 360
            if angle > 180:
                angle = 360 - angle
            
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 4)
                for (x,y) in [(x1,y1),(x2,y2),(x3,y3)]:
                    cv2.circle(img, (x, y), 8, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, str(int(angle)), (x2-50, y2-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            return angle
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0

    def getDetectionQuality(self):
        """Return detection quality score based on landmark visibility"""
        try:
            if not self.results or not self.results.pose_landmarks:
                return 0.0
            
            # Key landmarks for sit-up detection
            key_landmarks = [11, 12, 23, 24, 25, 26]  # shoulders, hips, knees
            total_visibility = 0
            valid_landmarks = 0
            
            for lm_id in key_landmarks:
                if lm_id < len(self.results.pose_landmarks.landmark):
                    lm = self.results.pose_landmarks.landmark[lm_id]
                    if lm.visibility > 0.3:
                        total_visibility += lm.visibility
                        valid_landmarks += 1
            
            if valid_landmarks == 0:
                return 0.0
            
            return total_visibility / valid_landmarks
        except Exception as e:
            print(f"Detection quality error: {e}")
            return 0.0

# ─────────────────────────────────────────────────────────────
# Enhanced Rep Counter with Fake Rep Detection
# ─────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self):
        self.count = 0
        self.state = "unknown"
        self.initial_state_set = False
        self.angle_history = deque(maxlen=15)
        self.rep_start_time = 0
        self.last_rep_time = 0
        self.movement_quality = deque(maxlen=10)
        self.fake_rep_threshold = 0.8  # Minimum quality score for valid rep
        
        # Enhanced thresholds
        self.DOWN_THRESHOLD = 150
        self.UP_THRESHOLD = 120
        self.MIN_REP_DURATION = 0.8  # Minimum time for valid rep
        self.MAX_REP_DURATION = 5.0   # Maximum time for valid rep
        self.MIN_MOVEMENT_QUALITY = 0.6  # Minimum movement smoothness
        
    def updateState(self, angle, detection_quality):
        """Update rep counter state with enhanced validation"""
        try:
            if angle <= 0 or detection_quality < 0.3:
                return "No valid pose detected"
            
            self.angle_history.append(angle)
            
            # Determine current position
            if angle < self.UP_THRESHOLD:
                current_position = "up"
            elif angle > self.DOWN_THRESHOLD:
                current_position = "down"
            else:
                current_position = "transition"
            
            # Set initial state
            if not self.initial_state_set:
                if current_position == "down":
                    self.state = "down"
                    self.initial_state_set = True
                    return "Start from down position"
                elif current_position == "up":
                    self.state = "up"
                    self.initial_state_set = True
                    return "Please go to down position first"
                else:
                    return "Assume down position to start"
            
            current_time = time.time()
            feedback = ""
            
            # State transitions with enhanced validation
            if self.state == "down" and current_position == "up":
                if self.rep_start_time > 0:
                    duration = current_time - self.rep_start_time
                    
                    # Validate rep quality
                    if self._isValidRep(duration, detection_quality):
                        self.count += 1
                        self.last_rep_time = current_time
                        feedback = f"Good rep! ({duration:.1f}s)"
                    else:
                        feedback = f"Invalid rep - too fast or poor quality ({duration:.1f}s)"
                else:
                    feedback = "No valid down hold detected"
                
                self.state = "up"
                
            elif self.state == "up" and current_position == "down":
                self.rep_start_time = current_time
                feedback = "Go up!"
                self.state = "down"
            
            # Movement quality feedback
            if len(self.angle_history) >= 5:
                smoothness = self._calculateMovementSmoothness()
                self.movement_quality.append(smoothness)
                
                if smoothness < self.MIN_MOVEMENT_QUALITY:
                    feedback += " - Move smoothly!"
            
            return feedback
        except Exception as e:
            print(f"State update error: {e}")
            return "Error updating state"
    
    def _isValidRep(self, duration, detection_quality):
        """Enhanced validation for rep quality"""
        try:
            # Check duration
            if duration < self.MIN_REP_DURATION or duration > self.MAX_REP_DURATION:
                return False
            
            # Check detection quality
            if detection_quality < self.fake_rep_threshold:
                return False
            
            # Check movement smoothness
            if len(self.movement_quality) >= 3:
                avg_quality = np.mean(list(self.movement_quality))
                if avg_quality < self.MIN_MOVEMENT_QUALITY:
                    return False
            
            # Check for rapid successive reps (potential fake reps)
            current_time = time.time()
            if self.last_rep_time > 0:
                time_since_last = current_time - self.last_rep_time
                if time_since_last < 1.0:  # Minimum 1 second between reps
                    return False
            
            return True
        except Exception as e:
            print(f"Rep validation error: {e}")
            return False
    
    def _calculateMovementSmoothness(self):
        """Calculate how smooth the movement is"""
        try:
            if len(self.angle_history) < 3:
                return 1.0
            
            angles = list(self.angle_history)
            # Calculate rate of change
            changes = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
            avg_change = np.mean(changes)
            
            # Normalize to 0-1 scale (lower is smoother)
            smoothness = max(0, 1 - (avg_change / 50))  # 50 degrees is max expected change
            return smoothness
        except Exception as e:
            print(f"Smoothness calculation error: {e}")
            return 1.0
    
    def getStats(self):
        """Get current statistics"""
        try:
            return {
                'count': self.count,
                'state': self.state,
                'detection_quality': np.mean(list(self.movement_quality)) if self.movement_quality else 0,
                'last_rep_time': self.last_rep_time
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {'count': 0, 'state': 'error', 'detection_quality': 0, 'last_rep_time': 0}
    
    def reset(self):
        """Reset counter state"""
        try:
            self.count = 0
            self.state = "unknown"
            self.initial_state_set = False
            self.angle_history.clear()
            self.rep_start_time = 0
            self.last_rep_time = 0
            self.movement_quality.clear()
        except Exception as e:
            print(f"Reset error: {e}")

# ─────────────────────────────────────────────────────────────
# Global variables
# ─────────────────────────────────────────────────────────────
ptime = 0
detector = None
rep_counter = None

# Initialize video capture
cap = None

def initialize_system():
    """Initialize the detection system with error handling"""
    global detector, rep_counter, cap
    
    try:
        detector = EnhancedPoseDetector()
        rep_counter = RepCounter()
        cap = cv2.VideoCapture(0)  # Use webcam (0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
            
        print("System initialized successfully")
        return True
        
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

# Frame generation function
def generate_frames():
    global ptime
    
    if not initialize_system():
        # Create error frame
        error_img = np.zeros((640, 960, 3), dtype=np.uint8)
        cv2.putText(error_img, "System Initialization Failed", (300, 320), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_img)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    while True:
        try:
            ret, img = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
                
            img = cv2.resize(img, (960, 640))
            
            # Add error handling for pose detection
            try:
                img = detector.findPose(img, True)
                lmList = detector.findPosition(img, True)
            except Exception as e:
                print(f"Pose detection error: {e}")
                # Continue with empty landmark list
                lmList = []
                # Draw error message on image
                cv2.putText(img, "Pose detection error", (50, 50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            if len(lmList) >= 25:
                angle = 0
                valid_angle_found = False
                
                # Try multiple angle calculations for better reliability
                try:
                    if len(lmList) > 25:
                        # Try left side first
                        angle = detector.findAngle(img, 11, 23, 25, True)
                        if angle > 0: 
                            valid_angle_found = True
                        else:
                            # Try right side
                            angle = detector.findAngle(img, 12, 24, 26, True)
                            if angle > 0: 
                                valid_angle_found = True
                    
                    # Fallback to different landmarks if needed
                    if not valid_angle_found and len(lmList) > 28:
                        angle = detector.findAngle(img, 11, 23, 27, True)
                        if angle > 0: 
                            valid_angle_found = True
                        else:
                            angle = detector.findAngle(img, 12, 24, 28, True)
                            if angle > 0: 
                                valid_angle_found = True
                except Exception as e:
                    print(f"Angle calculation error: {e}")
                    valid_angle_found = False
                    angle = 0
                
                if valid_angle_found:
                    try:
                        detection_quality = detector.getDetectionQuality()
                        feedback = rep_counter.updateState(angle, detection_quality)
                        
                        # Display angle and quality
                        cv2.putText(img, f"Angle: {int(angle)}", (50, 50), 
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        cv2.putText(img, f"Quality: {detection_quality:.2f}", (50, 80), 
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"State update error: {e}")
                        feedback = "Error updating state"
                        cv2.putText(img, f"Angle: {int(angle)}", (50, 50), 
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                else:
                    feedback = "Pose not detected properly"
                    rep_counter.rep_start_time = 0
            else:
                feedback = "Not enough landmarks detected"
                rep_counter.rep_start_time = 0

            # Display feedback and state
            cv2.putText(img, feedback, (50, 120), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            try:
                stats = rep_counter.getStats()
                cv2.putText(img, f"State: {stats['state']}", (50, 160), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            except Exception as e:
                print(f"Stats error: {e}")
                cv2.putText(img, "State: Error", (50, 160), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # Display rep count
            cv2.rectangle(img, (0, 380), (130, 480), (255, 0, 0), cv2.FILLED)
            try:
                count_display = str(int(stats.get('count', 0)))
            except:
                count_display = "0"
            cv2.putText(img, count_display, (25, 455),
                        cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

            # Display FPS
            ctime = time.time()
            fps = int(1 / (ctime - ptime)) if ptime else 0
            ptime = ctime
            cv2.putText(img, f"FPS={fps}", (20, 370),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            # Encode frame to JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', img)
                if not ret:
                    print("Failed to encode frame")
                    continue
                    
                frame = buffer.tobytes()
                
                # Yield frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                       
            except Exception as e:
                print(f"Frame encoding error: {e}")
                continue
                
        except Exception as e:
            print(f"Frame generation error: {e}")
            # Create a simple error frame
            error_img = np.zeros((640, 960, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera Error", (400, 320), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            
            try:
                ret, buffer = cv2.imencode('.jpg', error_img)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                pass
            
            # Small delay to prevent infinite loop
            time.sleep(0.1)
            continue

# ─────────────────────────────────────────────────────────────
# FastAPI Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    # HTML page with embedded video stream
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Sit-up Counter</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f8ff;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                text-align: center;
            }
            h1 {
                color: #2c3e50;
                background-color: #98FB98;
                padding: 15px;
                border-radius: 10px;
            }
            .video-container {
                margin: 20px 0;
                border: 2px solid #2c3e50;
                border-radius: 10px;
                overflow: hidden;
            }
            .controls {
                margin: 20px 0;
            }
            button {
                background-color: #98FB98;
                border: none;
                padding: 12px 24px;
                margin: 0 10px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #77dd77;
            }
            .info {
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Enhanced Sit-up Counter</h1>
            <div class="info">
                <h3>Features:</h3>
                <ul>
                    <li>Improved detection in low light and difficult angles</li>
                    <li>Fake rep detection and prevention</li>
                    <li>Movement quality assessment</li>
                    <li>Multiple detection strategies</li>
                    <li>Image enhancement for better accuracy</li>
                    <li>Enhanced error handling and stability</li>
                </ul>
            </div>
            <div class="video-container">
                <img src="/video_feed" width="960" height="640">
            </div>
            <div class="controls">
                <button onclick="clearCount()">Clear Count</button>
                <button onclick="getStats()">Get Stats</button>
            </div>
        </div>
        
        <script>
            function clearCount() {
                fetch('/clear', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Count cleared:', data);
                    location.reload();
                });
            }
            
            function getStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    alert('Current Stats:\\nCount: ' + data.count + '\\nState: ' + data.state + '\\nQuality: ' + data.detection_quality.toFixed(2));
                });
            }
        </script>
    </body>
    </html>
    """

@app.get('/video_feed')
async def video_feed():
    # Video streaming route
    return StreamingResponse(
        generate_frames(), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.post("/clear")
async def clear_count():
    if rep_counter:
        rep_counter.reset()
        return {"message": "Count cleared", "count": rep_counter.count}
    return {"message": "Counter not initialized", "count": 0}

@app.get("/stats")
async def get_stats():
    if rep_counter:
        return rep_counter.getStats()
    return {"count": 0, "state": "not initialized", "detection_quality": 0, "last_rep_time": 0}

# Cleanup when application shuts down
@app.on_event("shutdown")
def shutdown_event():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    gc.collect()

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

