import os
import time
from datetime import datetime
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from sort.sort import *
from db import Detection


class ObjectDetection:
    #id modelo: [archivo modelo, bbox porcentual, offset clases, clases personalizadas (opcional)]
    models = {
        "mobilenet": ["lite-model_ssd_mobilenet_v1_100_320_fp32_nms_1.tflite",True,1],
        "mobilenet_quant": ["coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite",True,1],
        "spaghettinet_s": ["spaghettinet_edgetpu_s.tflite",True,1],
        "spaghettinet_l": ["spaghettinet_edgetpu_l.tflite",True,1],
        "retinanet": ["lite-model_qat_mobilenet_v2_retinanet_256_1.tflite",False,0],
        "object_localizer": ["lite-model_object_detection_mobile_object_localizer_v1_1_metadata_2.tflite",True,0,["Objeto"]],
        "efficientdet": ["lite-model_efficientdet_lite1_detection_default_1.tflite",True,1],
    }
    def __init__(self,model_name,class_filter=None,min_score=0.2,threads=3,trk_hits=3,line_orientation="h",count_line=0.25):
        curDir = os.getcwd()
        self.model_cfg = self.models[model_name]
        self.interpreter = tflite.Interpreter(model_path=os.path.join(curDir, "models/"+self.model_cfg[0]),num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5
        self.height = int(self.input_details[0]['shape'][1])
        self.width = int(self.input_details[0]['shape'][2])
        try:
            self.labels = self.model_cfg[3]
        except:
            with open(os.path.join(curDir, "metadata/coco-classes.txt")) as cl:
                self.labels = cl.read().split("\n")
        self.class_filter = class_filter
        self.started = time.time()
        self.last_update = time.time()
        self.frame_count = 0
        self.current_fps = ""
        self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2
        self.min_score = min_score
        self.mot_tracker = Sort(min_hits=trk_hits)
        self.trk_ids = []
        self.count_line = count_line
        self.line_orientation = line_orientation;
        self.start_records = []
        self.end_records = []
        self.detection_count = 0
        self.detection_time_sum = 0

    def __rgb_tensor(self,frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        return input_data

    def detect(self,frame,track_objects=True,print_det_time=False):
        start_time_millis = int(round(time.time() * 1000))
        frame = cv2.resize(frame,(1024,720))
        self.interpreter.set_tensor(self.input_details[0]['index'],self.__rgb_tensor(frame))
        self.interpreter.invoke()
        end_time_millis = int(round(time.time() * 1000))
        all_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] 
        all_label_ids = self.interpreter.get_tensor(self.output_details[1]['index'])[0].astype('int16')
        if type(all_label_ids) is not np.ndarray:
            all_label_ids = self.interpreter.get_tensor(self.output_details[3]['index'])[0].astype('int16')
        all_scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        imH = frame.shape[0]
        imW = frame.shape[1]
        boxes,det_labels,scores,detections = [],[],[],[]
        for score, (ymin,xmin,ymax,xmax), label_id in zip(all_scores, all_boxes, all_label_ids):
            label_idx = label_id+self.model_cfg[2]
            if label_idx<(0+self.model_cfg[2]) or label_idx >= len(self.labels):
                continue
            if (self.class_filter and self.labels[label_idx] not in self.class_filter) or score < self.min_score or score > 1:
                continue
            if self.model_cfg[1] and (ymin>1 or xmin>1 or ymax>1 or xmax>1):
                continue
            scores.append(score)
            det_labels.append(self.labels[label_idx])
            if self.model_cfg[1]:
                boxes.append((int(ymin*imH),int(xmin*imW),int(ymax*imH),int(xmax*imW)))
            else:
                boxes.append((int(ymin*(imH/self.height)),int(xmin*(imW/self.width)),int(ymax*(imH/self.height)),int(xmax*(imW/self.width))))
            y1,x1,y2,x2=boxes[-1]
            if (x2-x1)>0 or (y2-y1)>0:
                detections.append([x1,y1,x2,y2,score,label_idx])
        if(len(detections)>0 and print_det_time):
            self.detection_count += 1
            det_time_millis = end_time_millis - start_time_millis
            self.detection_time_sum += det_time_millis
            print(f"Ult. detección: {det_time_millis}ms")
            print(f"Tiempo promedio: {round(self.detection_time_sum / self.detection_count)}ms")
        if(len(detections)>0 and track_objects):
            tracks = self.mot_tracker.update(np.array(detections))
        else:
            tracks = np.empty((0, 5))
        self.frame_count += 1
        now = time.time()
        if now - self.last_update > 1:
            self.current_fps = f"{round(self.frame_count / (now-self.last_update),2)} fps"
            self.last_update = now
            self.frame_count = 0
        return [det_labels, boxes, scores], tracks, frame
    
    def get_unique_objects(self,prediction,store=True):
        unique_objects = []
        _,tracks,frame = prediction
        mult = frame.shape[0] if self.line_orientation == "h" else frame.shape[1]
        count_line = int(self.count_line*mult)
        for track in tracks:
            xmin,ymin,xmax,ymax,track_id,_,label_idx = track.astype(int)
            ctr = int(ymin+((ymax - ymin)/2)) if self.line_orientation == "h" else int(xmin+((xmax - xmin)/2))
            if(ctr < count_line and track_id not in self.start_records):
                self.start_records.append(track_id)
            if(ctr >= count_line and track_id in self.start_records and track_id not in self.end_records):
                self.end_records.append(track_id)
                unique_objects.append({"track_id":track_id,"label":self.labels[label_idx],"date":datetime.now()})
                if(store):
                    det = Detection({'label':self.labels[label_idx],'date':datetime.now()})
                    det.save()
        return unique_objects,len(self.end_records)
        
        

    def draw_results(self,prediction,count=None):
        (det_labels,boxes,scores),tracks,frame = prediction
        frame = self.__draw_lines(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, score, (ymin,xmin,ymax,xmax) in zip(det_labels, scores, boxes):
            score_txt = f'{round(100 * score,0)}'
            cv2.rectangle(frame,(xmin, ymax),(xmax, ymin),(0,255,0),1)
            cv2.putText(frame,label,(xmin, ymax-10), font,1, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(frame,score_txt,(xmin+20, ymin+20), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        for track in tracks:
            xmin,ymin,xmax,ymax,track_id,_,label_idx = track.astype(int)
            cX,cY = int(xmin+((xmax - xmin)/2)),int(ymin+((ymax - ymin)/2))
            if track_id not in self.trk_ids:
                self.trk_ids.append(track_id)
            cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
            cv2.putText(frame,str(track_id),(xmax-50, ymin-10), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame,self.current_fps,(frame.shape[1]-230, 100), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
        if count != None:
            cv2.putText(frame,"Cant.: "+str(count),(10, 100), font, 1.5, (0,255,255), 2, cv2.LINE_AA)
        return frame
    
    def print_results(self,prediction,count=None):
        print(self.current_fps)
        if count != None:
            print("Cant.: "+str(count))
    
    def __draw_lines(self,frame):
        line_w = frame.shape[1] if self.line_orientation == "h" else frame.shape[0]
        mult = frame.shape[0] if self.line_orientation == "h" else frame.shape[1]
        count_lineA,count_lineB = (0,int(self.count_line*mult)),(line_w,int(self.count_line*mult))
        if self.line_orientation == "v":
            count_lineA,count_lineB = count_lineA[::-1],count_lineB[::-1]
        frame = cv2.line(frame,count_lineA,count_lineB,(0,255,255),3)
        return frame
