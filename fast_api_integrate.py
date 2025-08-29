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

# Initialize FastAPI app
app = FastAPI()

# ─────────────────────────────────────────────────────────────
# Pose Detector Class
# ─────────────────────────────────────────────────────────────
class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(mode, complexity, smooth_landmarks,
                                     enable_segmentation, smooth_segmentation,
                                     detectionCon, trackCon)
        
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=4)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            # Only draw the landmarks we want (head, shoulders, hips)
            self.drawCustomLandmarks(img)
            self.drawCustomConnections(img)
        return img
    
    def drawCustomLandmarks(self, img):
        h, w, c = img.shape
        if self.results.pose_landmarks:
            # Define the landmarks we want to show (head, shoulders, hips)
            target_landmarks = [0,  # nose
                               11, 12,  # shoulders
                               23, 24]  # hips
            
            for id in target_landmarks:
                if id < len(self.results.pose_landmarks.landmark):
                    lm = self.results.pose_landmarks.landmark[id]
                    if lm.visibility > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
    
    def drawCustomConnections(self, img):
        h, w, c = img.shape
        # Define the connections we want to show
        connections = [
            (11, 12),  # shoulder to shoulder
            (11, 23),  # left shoulder to left hip
            (12, 24),  # right shoulder to right hip
            (23, 24)   # hip to hip
        ]
        
        if self.results.pose_landmarks:
            for connection in connections:
                start_idx, end_idx = connection
                if (len(self.results.pose_landmarks.landmark) > max(start_idx, end_idx) and
                    self.results.pose_landmarks.landmark[start_idx].visibility > 0.5 and
                    self.results.pose_landmarks.landmark[end_idx].visibility > 0.5):
                    
                    start_point = (
                        int(self.results.pose_landmarks.landmark[start_idx].x * w),
                        int(self.results.pose_landmarks.landmark[start_idx].y * h)
                    )
                    end_point = (
                        int(self.results.pose_landmarks.landmark[end_idx].x * w),
                        int(self.results.pose_landmarks.landmark[end_idx].y * h)
                    )
                    cv2.line(img, start_point, end_point, (0, 165, 255), 2)
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                visibility = lm.visibility
                self.lmList.append([id, cx, cy, visibility])
                if draw and visibility > 0.5:
                    # Only draw the landmarks we want (head, shoulders, hips)
                    if id in [0, 11, 12, 23, 24]:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
        return self.lmList
        
    def findAngle(self, img, p1, p2, p3, draw=True):   
        if len(self.lmList) < max(p1, p2, p3) + 1:
            return 0
            
        if (self.lmList[p1][3] < 0.5 or 
            self.lmList[p2][3] < 0.5 or 
            self.lmList[p3][3] < 0.5):
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

# ─────────────────────────────────────────────────────────────
# Global variables
# ─────────────────────────────────────────────────────────────
count = 0
feedback = ""
state = "unknown"
initial_state_set = False
angle_history = deque(maxlen=10)
rep_start_time = 0
ptime = 0

# Initialize video capture and detector
cap = cv2.VideoCapture(0)  # Use webcam (0)
detector = PoseDetector()

DOWN_THRESHOLD = 150
UP_THRESHOLD   = 120

# Frame generation function
def generate_frames():
    global ptime, count, feedback, state, angle_history, rep_start_time, initial_state_set
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        img = cv2.resize(img, (960, 640))
        img = detector.findPose(img, True)
        lmList = detector.findPosition(img, True)

        if len(lmList) >= 25:
            angle = 0
            valid_angle_found = False
            if len(lmList) > 25:
                angle = detector.findAngle(img, 11, 23, 25, True)
                if angle > 0: valid_angle_found = True
                else:
                    angle = detector.findAngle(img, 12, 24, 26, True)
                    if angle > 0: valid_angle_found = True
            if not valid_angle_found and len(lmList) > 28:
                angle = detector.findAngle(img, 11, 23, 27, True)
                if angle > 0: valid_angle_found = True
                else:
                    angle = detector.findAngle(img, 12, 24, 28, True)
                    if angle > 0: valid_angle_found = True
            
            if valid_angle_found:
                angle_history.append(angle)
                if angle < UP_THRESHOLD:
                    current_position = "up"
                elif angle > DOWN_THRESHOLD:
                    current_position = "down"
                else:
                    current_position = "transition"

                if not initial_state_set:
                    if current_position == "down":
                        state = "down"
                        initial_state_set = True
                        feedback = "Start from down position"
                    elif current_position == "up":
                        state = "up"
                        initial_state_set = True
                        feedback = "Please go to down position first"
                    else:
                        feedback = "Assume down position to start"

                current_time = time.time()

                if state == "down" and current_position == "up":
                    if rep_start_time > 0:
                        duration = current_time - rep_start_time
                        if duration >= 0.25:
                            count += 1
                            feedback = f"Good rep! ({duration:.1f}s)"
                        else:
                            feedback = f"Too fast ({duration:.1f}s) - not counted!"
                    else:
                        feedback = "No valid down hold detected"
                    state = "up"

                elif state == "up" and current_position == "down":
                    rep_start_time = current_time
                    feedback = "Go up!"
                    state = "down"

                if len(angle_history) >= 5:
                    if np.std(list(angle_history)) > 30:
                        feedback = "Move smoothly!"

                cv2.putText(img, f"Angle: {int(angle)}", (50, 50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                feedback = "Pose not detected properly"
                rep_start_time = 0
        else:
            feedback = "Not enough landmarks detected"
            rep_start_time = 0

        cv2.putText(img, feedback, (50, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"State: {state}", (50, 150), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.rectangle(img, (0, 380), (130, 480), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (25, 455),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

        ctime = time.time()
        fps = int(1 / (ctime - ptime)) if ptime else 0
        ptime = ctime
        cv2.putText(img, f"FPS={fps}", (20, 370),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
        <title>Sit-up Counter</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sit-up Counter</h1>
            <div class="video-container">
                <img src="/video_feed" width="960" height="640">
            </div>
            <div class="controls">
                <button onclick="clearCount()">Clear Count</button>
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
    global count, state, angle_history, rep_start_time, initial_state_set
    count = 0
    state = "unknown"
    initial_state_set = False
    angle_history = deque(maxlen=10)
    rep_start_time = 0
    return {"message": "Count cleared", "count": count}

# Cleanup when application shuts down
@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
