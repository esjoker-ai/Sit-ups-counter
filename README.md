# 🏃‍♂️ Enhanced Sit-up Counter

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-purple.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time computer vision application that uses **MediaPipe pose detection** to automatically count sit-up repetitions with advanced features like fake rep detection, movement quality assessment, and posture discrimination.

## 🌟 Features

### 🎯 Core Functionality
- **Real-time Pose Detection**: Advanced MediaPipe pose tracking with 33 landmarks
- **Intelligent Rep Counting**: Automatic sit-up counting with fake rep detection
- **Movement Quality Assessment**: Evaluates exercise form and smoothness
- **Posture Discrimination**: Distinguishes sit-ups from squats and other movements
- **Low-light Enhancement**: Improved detection in challenging lighting conditions

### 🖥️ Web Interface
- **Live Video Stream**: Real-time pose visualization with color-coded landmarks
- **Interactive Dashboard**: User-friendly controls and statistics
- **Debug Information**: Real-time feedback and detection parameters
- **Responsive Design**: Works on different screen sizes

### 🔧 Advanced Features
- **Multi-model Detection**: Primary and fallback detection strategies
- **Image Enhancement**: CLAHE and noise reduction for better accuracy
- **State Management**: Tracks exercise progression and timing
- **Error Handling**: Robust error management and recovery

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Camera (USB webcam or built-in)
- Linux (tested on Ubuntu 6.12.10)

### Installation

#### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-situp-counter.git
cd enhanced-situp-counter

# Run the installation script
./install.sh
```

#### Option 2: Manual Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-situp-counter.git
cd enhanced-situp-counter

# Create virtual environment
python3 -m venv situp_counter_env
source situp_counter_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Activate virtual environment (if not already activated)
source situp_counter_env/bin/activate

# Start the application
python3 fast_api_integrate.py
```

### Access the Application
1. Open your web browser
2. Navigate to `http://localhost:8000`
3. Allow camera access when prompted
4. Start doing sit-ups! 🏃‍♂️

## 📁 Project Structure

```
enhanced-situp-counter/
├── fast_api_integrate.py          # Main application file
├── requirements.txt               # Python dependencies
├── install.sh                     # Automated installation script
├── create_documentation_simple.py # Documentation generator
├── documentation.docx             # Complete technical documentation
├── cloud_mobile_specification.md  # Cloud & mobile integration specs
├── README.md                      # This file
├── test*.mp4                      # Test video files
└── yolo*.pt                       # YOLO model files (if used)
```

## 🎮 Usage Instructions

### Performing Sit-ups
1. **Positioning**: Lie down on your back with knees bent
2. **Camera Setup**: Ensure full body is visible in camera frame
3. **Starting Position**: Begin in down position (lying flat)
4. **Exercise**: Perform sit-ups with controlled movement
5. **Counting**: System automatically counts valid repetitions

### Interface Controls
- **Clear Count**: Reset counter to zero
- **Get Stats**: View current statistics
- **Live Stream**: Real-time video feed with pose detection

### Visual Feedback
- 🟢 **Green Landmarks**: High visibility landmarks
- 🟠 **Orange Landmarks**: Medium visibility landmarks
- 🔴 **Red Landmarks**: Low visibility landmarks
- 📐 **Angle Display**: Shows current joint angle
- 📊 **Quality Score**: Detection quality indicator

## 🔧 Configuration

Key parameters can be adjusted in `fast_api_integrate.py`:

```python
# Angle Thresholds
UP_THRESHOLD = 110      # Degrees for up position
DOWN_THRESHOLD = 150    # Degrees for down position

# Timing Parameters
MIN_REP_DURATION = 0.3  # Minimum rep duration (seconds)
MAX_REP_DURATION = 5.0  # Maximum rep duration (seconds)

# Quality Parameters
fake_rep_threshold = 0.5      # Minimum quality score
MIN_MOVEMENT_QUALITY = 0.3    # Minimum movement smoothness
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | Live video stream |
| `/clear` | POST | Reset counter |
| `/stats` | GET | Get current statistics |

### Example API Usage
```bash
# Get current statistics
curl http://localhost:8000/stats

# Clear counter
curl -X POST http://localhost:8000/clear
```

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
- Check camera connection and permissions
- Try different camera indices (0, 1, 2)
- Ensure camera is not being used by another application

#### Poor Detection Quality
- Improve lighting conditions
- Ensure full body is visible in camera frame
- Clean camera lens
- Check camera positioning

#### Performance Issues
- Close other applications
- Reduce camera resolution if needed
- Check system resources
- Use lower model complexity

## 📊 Performance

### System Requirements
- **CPU**: Multi-core recommended
- **Memory**: Minimum 4GB RAM
- **GPU**: Optional acceleration
- **Network**: Minimal usage for local deployment

### Optimization Tips
- **Model Complexity**: Balance accuracy vs performance
- **Frame Processing**: Optimize image preprocessing
- **Memory Management**: Regular garbage collection
- **Camera Settings**: Optimize resolution and FPS

## 🔮 Future Roadmap

### 🚀 Cloud Deployment & Mobile Integration
The next major phase will transform this into a scalable cloud-based mobile application:

#### Cloud Infrastructure
- **Alibaba Cloud ECS** or **Tencent Cloud CVM** hosting
- **Load balancing** and auto-scaling
- **GPU acceleration** for improved processing
- **CDN integration** for global delivery

#### Flutter Mobile App
- **Cross-platform** iOS and Android support
- **Real-time camera integration** with live processing
- **Cloud communication** via RESTful APIs
- **Offline capability** when server unavailable

#### Advanced Features
- **Multi-user support** with concurrent processing
- **User authentication** and profile management
- **Workout history** and analytics
- **Social features** and achievements

### Implementation Timeline
1. **Phase 1**: Cloud Infrastructure (Weeks 1-2)
2. **Phase 2**: API Development (Weeks 3-4)
3. **Phase 3**: Flutter App Development (Weeks 5-8)
4. **Phase 4**: Real-time Integration (Weeks 9-10)
5. **Phase 5**: User Management (Weeks 11-12)
6. **Phase 6**: Performance Optimization (Week 13)
7. **Phase 7**: Testing & Deployment (Week 14)

## 📚 Documentation

- **[Complete Documentation](documentation.docx)** - Comprehensive technical documentation
- **[Cloud & Mobile Specs](cloud_mobile_specification.md)** - Detailed cloud deployment and mobile app specifications
- **[API Reference](documentation.docx#api-endpoints)** - Complete API documentation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run tests
pytest

# Run linting
flake8 fast_api_integrate.py
```


