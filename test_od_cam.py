import cv2
from picamera2 import Picamera2
from tflite_object_detection import ObjectDetection

object_detection = ObjectDetection(
        model_name="spaghettinet_s",
        class_filter=["person"],
        min_score=0.40,
        trk_hits=2,
        threads=3,
        line_orientation="v",
        count_line=0.4,
    )

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
while True:
    frame= picam2.capture_array()
    prediction = object_detection.detect(frame=frame,track_objects=True)
    unique_objects,count = object_detection.get_unique_objects(prediction=prediction,store=True)
    result = object_detection.draw_results(prediction,count)
    cv2.imshow('Frame',result)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()