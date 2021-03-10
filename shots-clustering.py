import shutil, sys, io, os
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from google.cloud import videointelligence_v1p3beta1 as videointelligence
from bisect import bisect
from face import encode_face, cluster_face
FACE_DATA_PATH = os.path.join(os.getcwd(),'faces')

if len(sys.argv) != 2:
    print('Illegal input. Useage: \n\tpython leading-casts.py target')
    exit()
file = sys.argv[1]
clip = mp.VideoFileClip(file)

shutil.rmtree(FACE_DATA_PATH, ignore_errors=True)
os.mkdir(FACE_DATA_PATH)

def detect_faces(local_file_path=None, gcs_uri=None):
    """Detects faces in a video from a local file."""

    client = videointelligence.VideoIntelligenceServiceClient()
    if gcs_uri:
        input_content = None
    else:
        with io.open(local_file_path, "rb") as f:
            input_content = f.read()

    # Configure the request
    config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True
    )
    context = videointelligence.VideoContext(face_detection_config=config)

    # Start the asynchronous request
    if input_content:
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION, videointelligence.Feature.SHOT_CHANGE_DETECTION],
                "input_content": input_content,
                "video_context": context,
            }
        )
    else:
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION, videointelligence.Feature.SHOT_CHANGE_DETECTION],
                "input_uri": gcs_uri,
                "video_context": context,
            }
        )

    print("\nProcessing video for shot change and face detection annotations.")
    result = operation.result(timeout=300)

    print("\nFinished processing.\n")

    # Retrieve the first result, because a single video was processed.
    shotAnnotation, faceAnnotation = result.annotation_results[0].shot_annotations, result.annotation_results[0].face_detection_annotations
    return (shotAnnotation, faceAnnotation)

(shotAnnotation, faceAnnotation) = detect_faces(local_file_path=file)

# chop each shot
shots = []
for i, shot in enumerate(shotAnnotation):
    start_time = (
        shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
    )
    end_time = (
        shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6
    )
    #ffmpeg_extract_subclip(file, start_time, end_time, targetname="{}-face-{}.mp4".format('.'.join(file.split('.')[:-1]), i))
    shots.append((start_time, end_time))
shots.sort()

faceThresh = 0.05 # ignore faces that below the size threshold
faceIdx = 0
faceTimestamp = {} # For each face track, assignment a timestamp to the face using average time offset of the track. This timestamp will be used to find shot that contains the face track
for annotation in faceAnnotation:
    for track in annotation.tracks:
        box0 = track.timestamped_objects[0].normalized_bounding_box
        if (box0.bottom - box0.top)*(box0.right - box0.left) < faceThresh:
            continue
        segStart = track.segment.start_time_offset.seconds + track.segment.start_time_offset.microseconds / 1e6
        segEnd = track.segment.end_time_offset.seconds + track.segment.end_time_offset.microseconds / 1e6
        faceTimestamp[faceIdx] = (segStart + segEnd) / 2
        with open(os.path.join(FACE_DATA_PATH, "{}.jpg".format(faceIdx)), "wb") as fh:
            fh.write(annotation.thumbnail)
        
        # #ffmpeg_extract_subclip(file, segStart, segEnd, targetname="{}-face-{}.mp4".format('.'.join(file.split('.')[:-1]), i))

        # for i, timedObj in enumerate(track.timestamped_objects):
        #     box = timedObj.normalized_bounding_box
        #     t = timedObj.time_offset.seconds + timedObj.time_offset.microseconds / 1e6
        #     frame = clip.get_frame(t)
        #     (h, w, _) = frame.shape
            
        #     face = Image.fromarray(frame[max(int(box.top*h) - 10, 0):min(int(box.bottom*h) + 10, h), max(int(box.left*w) - 10, 0):min(int(box.right*w)+10, w)])
        #     face.save("{}-face-{}-{}.png".format('.'.join(file.split('.')[:-1]), faceCnt, i))
        faceIdx += 1

faceData = encode_face(FACE_DATA_PATH)
faceCluster = cluster_face(faceData)
for faceID in faceCluster:
    for faceIdx in faceCluster[faceID]:
        timestamp = faceTimestamp[faceIdx]
        shotIdx = bisect(shots, (timestamp, timestamp)) - 1 # shorts is a list of (shot start time, shot end time)
        shot = shots[shotIdx] 
        
        # save the corresponding shot
        ffmpeg_extract_subclip(file, shot[0], shot[1], os.path.join('face{}'.format(faceID), 'shot-{}.mp4'.format(shotIdx)))
