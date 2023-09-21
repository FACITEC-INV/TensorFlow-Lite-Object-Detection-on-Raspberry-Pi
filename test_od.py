import os
import cv2
from tflite_object_detection import ObjectDetection 
from resource_usage import ResourceUsage
resUsage = ResourceUsage()
resUsage.addInfo()

object_detection = ObjectDetection(
    model_name="mobilenet_quant",
    class_filter=["person"],
    min_score=0.2,
    threads=3,
    trk_hits=3,
    line_orientation="h",
    count_line=0.5,
)

curDir = os.getcwd()

cap = cv2.VideoCapture(os.path.join(curDir, "videos/people.mp4"))

i=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        prediction = object_detection.detect(
            frame=frame,
            track_objects=False,
            print_det_time=True
        )
        unique_objects,count = object_detection.get_unique_objects(
            prediction=prediction,
            store=False
        )
        resUsage.addInfo()
        #result = object_detection.draw_results(prediction,count)
        #cv2.imshow('Frame',result)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    else: 
        break
cap.release()
cv2.destroyAllWindows()
resUsage.printInfo()
