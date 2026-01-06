# Tennis Match Analysis System

A comprehensive computer vision solution for automated tennis match analysis. This system leverages deep learning object detection, keypoint detection, and multi-object tracking to provide real-time insights into player performance and match dynamics through advanced video analysis.

## Overview

The Tennis Match Analysis System processes video footage to extract meaningful metrics and insights about tennis matches. By integrating YOLO object detection, PyTorch-based keypoint detection, and OpenCV video processing, the system delivers detailed analytics including player tracking, ball trajectory analysis, and performance metrics visualization.

## Key Features

### Detection & Tracking
- **Dual-Model Player Detection**: Utilizes YOLOv8x for robust player detection across varying lighting and occlusion conditions
- **Ball Tracking**: Custom-trained ball detection model with interpolation for smoother trajectories
- **Court Keypoint Detection**: Deep learning-based identification of court boundary points and reference lines

### Analytics & Metrics
- **Shot Analysis**: Automatic detection and classification of shot events with speed calculations
- **Speed Measurements**: 
  - Ball velocity in km/h
  - Player movement speed and agility metrics
- **Performance Statistics**: Shot count, average speed, and temporal performance tracking
- **Historical Data**: Full match-long statistics aggregation with forward-fill interpolation

### Visualization
- **Real-time Annotations**: Bounding boxes and tracking overlays on video frames
- **Mini Court Representation**: Bird's-eye view of player and ball positions mapped to court coordinates
- **Player Statistics Display**: Live performance metrics overlaid on video output
- **Frame-by-frame Analysis**: Detailed logging of all events and measurements

## Technical Architecture

### Core Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Player Tracker** | Detect and track tennis players | YOLOv8x |
| **Ball Tracker** | Detect and track tennis ball | Custom YOLOv8 (Fine-tuned) |
| **Court Line Detector** | Identify court boundaries and keypoints | PyTorch CNN |
| **Mini Court** | Project detections to 2D court view | Coordinate transformation |
| **Statistics Engine** | Calculate performance metrics | NumPy, Pandas |

### Video Processing Pipeline
1. **Input**: Video file loading and frame extraction
2. **Detection**: Parallel detection of players and ball across frames
3. **Court Calibration**: Detect court keypoints from first frame
4. **Filtering**: Select relevant players based on court positioning
5. **Coordinate Mapping**: Transform pixel coordinates to court space
6. **Analytics**: Calculate shots, speeds, and player movements
7. **Visualization**: Generate annotated output video with overlays
8. **Output**: Save processed video with all analysis results

## Project Structure

```
├── main.py                          # Main execution pipeline
├── yolo_inference.py               # YOLO model inference utilities
├── constants/                       # Configuration constants
├── court_line_detector/            # Court keypoint detection module
├── trackers/                       # Player and ball tracking modules
│   ├── player_tracker.py
│   └── ball_tracker.py
├── mini_court/                     # Court coordinate projection system
├── utils/                          # Utility functions
│   ├── bbox_utils.py              # Bounding box operations
│   ├── conversions.py             # Unit and coordinate conversions
│   ├── video_utils.py             # Video I/O operations
│   └── player_stats_drawer_utils.py # Statistics visualization
├── training/                       # Model training notebooks
│   ├── tennis-ball-detector-training.ipynb
│   └── tennis-court-keypoints-training.ipynb
├── input_videos/                  # Input video directory
├── outputs/                       # Processed output videos
└── tracker_stubs/                 # Cached detection results
```

## Dependencies

### Core Libraries
- **ultralytics**: YOLO v8 object detection and tracking
- **PyTorch**: Deep learning framework for keypoint detection
- **OpenCV (cv2)**: Video processing and frame manipulation
- **Pandas**: Data aggregation and statistics computation
- **NumPy**: Numerical operations and array processing

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- GPU recommended for real-time inference (CUDA 11.8+)

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd Tennis-System-analysis
```

2. Install required dependencies:
```bash
pip install ultralytics torch opencv-python pandas numpy
```

3. Download or prepare required models:
   - YOLOv8x weights (auto-downloaded on first run)
   - Custom ball detector: `./models/last.pt`
   - Court keypoints model: `./models/keypoints_model.pth`

4. Prepare input video:
   - Place video file at `./input_videos/input_video.mp4`

## Usage

### Running the Analysis Pipeline

```bash
python main.py
```

This will execute the full pipeline:
1. Load input video
2. Detect all players and ball across frames
3. Detect court keypoints
4. Calculate performance statistics
5. Generate annotated output video

**Output**: Processed video saved to `./outputs/output.mp4`

### Using Cached Detections

For faster iteration during development, the system supports caching detection results:
- Player detections: `./tracker_stubs/player_detections.pkl`
- Ball detections: `./tracker_stubs/ball_detections.pkl`

Subsequent runs will load cached results instead of re-running inference (controlled by `read_from_stub` parameter).

## Model Training

Training notebooks for custom models are provided:
- **Ball Detector Training**: `training/tennis-ball-detector-training.ipynb`
- **Court Keypoints Training**: `training/tennis-court-keypoints-training.ipynb`

These notebooks detail the data preparation, model architecture, and training procedure for fine-tuning models on custom datasets.

## Performance Metrics Explained

### Shot Speed (km/h)
- Calculated from ball displacement between shot detection frames
- Based on court dimensions and temporal resolution (24 fps)
- Provides insight into shot power and pace

### Player Speed (km/h)
- Computed from player movement between frames
- Indicates response time and court coverage capability
- Useful for assessing player fitness and positioning

### Shot Count
- Cumulative count of distinct shot events per player
- Helps identify dominant players and match statistics

### Average Metrics
- Time-averaged statistics computed over entire match duration
- Provides high-level performance summary

## Limitations & Considerations

- **Lighting Conditions**: Performance may degrade in poor lighting or high glare
- **Occlusion**: Multiple overlapping players can affect tracking accuracy
- **Frame Rate**: Default 24 fps assumption; adjust time calculations for different rates
- **Court Variations**: Model trained on standard court dimensions; custom courts may require fine-tuning

## Future Enhancements

- Multi-court support for simultaneous match analysis
- Advanced metrics (serve speed, rally length, winner/unforced error classification)
- Real-time live stream processing
- Web-based dashboard for match statistics visualization
- Integration with broadcast feeds

## Contributing

Contributions are welcome! Areas for improvement:
- Enhanced model performance and robustness
- Additional statistical metrics and analytics
- Improved visualization and reporting
- Documentation and examples

## License

[Specify your license here]

## References & Demo

- **Live Demo**: [View on LinkedIn](https://www.linkedin.com/posts/mohamed-el-bialy-6a0874268_ai-machinelearning-computervision-activity-7311881633270325266-m8fa?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEGftdgB5NLCLy34489p9vvZKXtNOgElOv0)
- **YOLOv8 Documentation**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **PyTorch**: [Deep Learning Framework](https://pytorch.org/)

