# face-recognition

## Description

face detection and recognition module with opencv , mediapipe, and deepface packages. User Interface were created using Tkinter.


## Getting started

Python version : python 3.7

### Step 1: Installation with requirements:

```
$ git clone https://github.com/tan800630/face-recognition.git
$ cd face-recognition
$ pip install -r requirements.txt
```

You can also installed required packages with the following commands.

```
$ pip install deepface==0.0.73
$ pip install mediapipe==0.8.10
```

### Step 2: Download SSD model configuration/weights and move into "./models/opencv_face_detector/"

- [model structure](https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt)
- [pre-trained weights](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

### Step 3: Put some facial image into "./img/database" folder (there's already one sample image in database)

- note : make sure the images in database folder are clear-cropped faces.

## Basic usage

```
python main.py
```


## To Do

- apply more face detection and recognition methods.
- configuration of UI
- tools for face-cropping (to those who want to create database but without face-only images)