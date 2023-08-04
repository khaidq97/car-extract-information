import json 
import pickle
from configs import MODEL_PATH, DATA_PATH, LIMIT_LINE_PATH
from model import Yolov5
from utils import is_rectangle_inside_polygon

class Controller:
    def __init__(self, 
                 model_path=MODEL_PATH,
                 data_path=DATA_PATH,
                 limit_line_path=LIMIT_LINE_PATH,):
        
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
        self.id_to_data = d['id_to_data']
        
        with open(limit_line_path, 'r') as f:
            self.limit_line = json.load(f)
        self.model = Yolov5(model_path=model_path)
        
    def run(self, img):
        boxes, labels, scores = self.model.run(img)
        if not len(boxes):
            return [], [], [] 
        data = []
        new_boxes = []
        for box, label in zip(boxes, labels):
            if is_rectangle_inside_polygon(box, self.limit_line):
                data.append(self.id_to_data[label])
                new_boxes.append(box)
        return data, new_boxes, boxes