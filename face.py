# import the necessary packages
from imutils import paths, build_montages
# face_recognition library by @ageitgey
import face_recognition
# argument parser
import argparse
# pickle to save the encodings
import pickle
# openCV
import cv2
# operating system
import os
from sklearn.cluster import DBSCAN
import numpy as np


CLUSTERING_RESULT_PATH = os.getcwd()

def move_image(image,id,labelID):
    path = os.path.join(CLUSTERING_RESULT_PATH, 'face'+str(labelID))
    # os.path.exists() method in Python is used to check whether the specified path exists or not.
    # os.mkdir() method in Python is used to create a directory named path with the specified numeric mode.
    if os.path.exists(path) == False:
        os.mkdir(path)

    filename = str(id) +'.jpg'
    # Using cv2.imwrite() method 
    # Saving the image 
    
    cv2.imwrite(os.path.join(path , filename), image)
    
    return

def encode_face(dataset: str, method: str = 'cnn'):
    """encode faces of the input dataset.

    Args:
        dataset (str): path to input directory of faces + images
        method (str): face detection model to use: either `hog` or `cnn`
    """
    # grab the paths to the input images in our dataset, then initialize
    # out data list (which we'll soon populate)
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))
    data = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        print(imagePath)

        # loading image to BGR
        image = cv2.imread(imagePath)

        # ocnverting image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(image, model=method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(image, boxes)

        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    return data

def cluster_face(data):
    encodings = [d["encoding"] for d in data]

    # cluster the embeddings
    print("[INFO] face clustering...")

    # creating DBSCAN object for clustering the encodings with the metric "euclidean"
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)

    # determine the total number of unique faces found in the dataset
    # clt.labels_ contains the label ID for all faces in our dataset (i.e., which cluster each face belongs to).
    # To find the unique faces/unique label IDs, used NumPy’s unique function.
    # The result is a list of unique labelIDs
    labelIDs = np.unique(clt.labels_)

    # we count the numUniqueFaces . There could potentially be a value of -1 in labelIDs — this value corresponds
    # to the “outlier” class where a 128-d embedding was too far away from any other clusters to be added to it.
    # “outliers” could either be worth examining or simply discarding based on the application of face clustering.
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    faceCluster = {l:[] for l in labelIDs}
    # loop over the unique face integers
    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)),
            replace=False)

        # loop over the sampled indexes
        for i in idxs:
            # load the input image and extract the face ROI
            image = cv2.imread(data[i]["imagePath"])
            (top, right, bottom, left) = data[i]["loc"]
            faceCluster[labelID].append(int(os.path.split(data[i]["imagePath"])[-1].split('.')[0]))
            # puting the image in the clustered folder
            move_image(image,i,labelID)
    return faceCluster
