#  AI Drowsiness Detection System

##  Overview

This project is a real-time **AI-based drowsiness detection system** built using Computer Vision.
It monitors eye movements using facial landmarks and alerts the user when signs of drowsiness are detected.

The system helps in preventing accidents caused by driver fatigue by providing instant alerts.



##  Features

*  Real-time face & eye tracking
*  Drowsiness detection using eye closure analysis
*  Alarm alert system when drowsiness is detected
*  Fast and efficient processing
*  Reduced false alerts using time threshold logic



##  Technologies Used

* Python
* OpenCV
* MediaPipe
* Pygame



##  Project Structure

```
drowsiness_detection/
│── main.py
│── alarm.mp3
│── README.md
```



##  How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/drowsiness-detection.git
```

2. Navigate to the project folder:

```
cd drowsiness-detection
```

3. Install dependencies:

```
pip install opencv-python mediapipe pygame
```

4. Run the project:

```
python main.py
```

---

##  How It Works

* The system uses MediaPipe Face Mesh to detect facial landmarks.
* Eye landmarks are tracked continuously.
* If eyes remain closed for a specific duration, the system detects drowsiness.
* An alarm is triggered to alert the user.



##  Applications

* Driver safety systems
* Smart attendance monitoring
* Workplace fatigue detection
* AI surveillance systems



##  Future Enhancements

*  Mobile app integration
*  Cloud-based monitoring
*  Drowsiness analytics dashboard
*  Integration with smart vehicles


