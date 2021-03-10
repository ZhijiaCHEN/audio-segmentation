import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#from google.cloud import videointelligence
from google.cloud import videointelligence_v1p3beta1 as videointelligence
import sys, io
from PIL import Image

file = sys.argv[1]
clip = mp.VideoFileClip(file)

def detect_faces(local_file_path="path/to/your/video-file.mp4"):
    """Detects faces in a video from a local file."""

    client = videointelligence.VideoIntelligenceServiceClient()

    with io.open(local_file_path, "rb") as f:
        input_content = f.read()

    # Configure the request
    config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True
    )
    context = videointelligence.VideoContext(face_detection_config=config)

    # Start the asynchronous request
    operation = client.annotate_video(
        request={
            "features": [videointelligence.Feature.FACE_DETECTION],
            "input_content": input_content,
            "video_context": context,
        }
    )

    print("\nProcessing video for face detection annotations.")
    result = operation.result(timeout=300)

    print("\nFinished processing.\n")

    # Retrieve the first result, because a single video was processed.
    faceAnnotation = result.annotation_results[0].face_detection_annotations
    return faceAnnotation

faceAnnotation = detect_faces(file)
for i, annotation in enumerate(faceAnnotation):
    for track in annotation.tracks:
        segStart = track.segment.start_time_offset.seconds + track.segment.start_time_offset.microseconds / 1e6
        segEnd = track.segment.end_time_offset.seconds + track.segment.end_time_offset.microseconds / 1e6
        ffmpeg_extract_subclip(file, segStart, segEnd, targetname="{}-face-{}.mp4".format('.'.join(file.split('.')[:-1]), i))

        # Grab the first frame of the timestamped faces
        faceThresh = 0.01 # ignore faces that below the size threshold
        timedObj = track.timestamped_objects[0]
        box = timedObj.normalized_bounding_box
        t = timedObj.time_offset.seconds + timedObj.time_offset.microseconds / 1e6
        frame = clip.get_frame(t)
        (h, w, _) = frame.shape
        if (box.bottom - box.top)*(box.right - box.left) < faceThresh:
            continue
        face = Image.fromarray(frame[int(box.top*h):int(box.bottom*h), int(box.left*w):int(box.right*w)])
        face.save("{}-face-{}.png".format('.'.join(file.split('.')[:-1]), i))
