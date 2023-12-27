# Emotion Recognition 

The Emotion Recognition project is a simple yet effective system designed to recognize facial expressions and classify them into different emotional states. The system utilizes facial landmarks and a machine learning model to identify emotions in real-time through a webcam.

## Key Components

### 1. Trainer.py

The `trainer.py` script captures facial landmarks and associates them with user-provided emotional labels, creating a dataset for training. It employs the dlib library for facial landmark detection and OpenCV for webcam interaction. The data is saved in a NumPy array format for subsequent model training.

### 2. Detect.py

The `detect.py` script loads the pre-trained machine learning model and applies it to recognize emotions in real-time webcam frames. It utilizes the KNeighborsClassifier from scikit-learn for classification and overlays emotion labels on the webcam feed.

## Requirements

- Python 3.x
- OpenCV
- dlib
- NumPy
- scikit-learn

## Getting Started

1. Clone the project repository.

    ```bash
    git clone https://github.com/Shreyasb1015/Emotion-Recognition
    cd Emotion-Recognition
    ```

2. Install the required libraries.

    ```bash
    pip install opencv-python dlib numpy scikit-learn
    ```

3. Download the shape predictor file (`shape_predictor_68_face_landmarks.dat`) from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the "Emotion-Recognition" directory.

4. Run the training script to collect facial landmarks and emotions.

    ```bash
    python trainer.py
    ```

5. Execute the detection script to see real-time emotion recognition.

    ```bash
    python detect.py
    ```

## Notes

- Ensure that the shape predictor file is correctly placed in the project directory.
- Press 'a' during the training phase to capture facial landmarks labeled with the current emotion.
- Press 'Esc' (27) to exit the real-time emotion detection.

***
Feel free to experiment, enhance, and adapt the project to suit your needs. Happy coding!
