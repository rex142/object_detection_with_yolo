# => install opencv (cv2), numpy, pytorch, ultralytics
# =>Use open cv to read video frame by frame and display on a window
# =>learn to use pip freeze, generate dependency text file "requirements.txt"


# task1: Use YOLOv8s on (cpu) to "predict" object detection for all classes, display prediction on window (object detection of all classes)

# task2: Use YOLOv8s on (cpu) to "predict" object detection for "Person" class, display prediction on window (object detection of Person class only)


# task3: Track individual persons in the video, display prediction on window https://docs.ultralytics.com/modes/track/#persisting-tracks-loop

from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2
vid = 'vid.mp4'
model = YOLO("yolov8s.pt")  # pretrained model

# use open cv frame by frame as input for yolov8


def object_detection_all_clases(vid, model):  # use predic

    # used model.predict to to "predict the video"
    results = model.predict(source=vid, conf=0.4, device='cpu')
    # use open cv to display the results
    frame_count = 0
    for result in results:
        # access the each image plot it first
        annotated_frame = result.plot()
        frame_count = frame_count+1
        print(f"frame number : {frame_count}")
        cv2.imshow('object_detection', annotated_frame)
        cv2.waitKey(1000)

        # tried doing directly cv2.imshow(result) didnt work peraphs its not a numpy array
cv2.destroyAllWindows()


def object_detection_person(vid, model):

    # used model.predict to to "predict the video"
    results = model.predict(source=vid, conf=0.4, device='cpu', classes=[0])
    # use open cv to display the results
    frame_count = 0
    for result in results:
        # access the each image plot it first
        annotated_frame = result.plot()
        frame_count = frame_count+1
        print(f"frame number : {frame_count}")
        cv2.imshow('object_detection', annotated_frame)
        cv2.waitKey(1000)

        # tried doing directly cv2.imshow(result) didnt work peraphs its not a numpy array
cv2.destroyAllWindows()


def persistent_tracking(vid, model):
    captured_vid = cv2.VideoCapture(vid)  # capturing video
    # storing the track history
    track_history = defaultdict(lambda: [])
    while captured_vid.isOpened():
        success, frame = captured_vid.read()

        if success:
            # run yolo tracking on the frame, persisting tracks between frames to actually track a person
            result = model.track(frame, persist=True)[0]
            # Get the boxes and track ids
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

             # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 40:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(
                    230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


object_detection_person(vid, model)
