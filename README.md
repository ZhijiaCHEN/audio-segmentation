# Video Segmentation
Segment video clip based on silent intervals.
## Usage
`python video-segmentation.py target`

## Dependency
numpy, moviepy

# Leading Casts Extraction
Recognize faces in the target video and extract shots for each main role based on the clustered faces.

## Usage
`python leading-cast.py target`

## Dependency
numpy, moviepy, face_recognition, imutils, sklearn, opencv-python, [google-cloud-videointelligence](https://github.com/googleapis/python-videointelligence)

## Acknowledgement
Implementation of [face encoding and clustering](face.py) are adapted from [Face-Clustering](https://github.com/kunalagarwal101/Face-Clustering).