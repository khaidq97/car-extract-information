import cv2 
import os
import yaml
from .config import DATABASE_PATH

def load_database(database_path):
    with open(database_path, 'r') as file:
        database = yaml.full_load(file)
    return database 

class ExtractInfoEngine:
    def __init__(self):
        self.database = load_database(DATABASE_PATH) 

    def run(self, image, cars_box, cars_id, debug_dir=None):
        cars_info = []
        debug_image = image.copy()
        for id, box in zip(cars_id, cars_box):
            info = self.database[id]
            info['box'] = box
            cars_info.append(info)
            if debug_dir:
                for i, (key, val) in enumerate(info.items()):
                    cv2.rectangle(debug_image, box[:2], box[2:], (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{str(key)}:{str(val)}", (box[0], box[1]+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "car_extract_info.png"), debug_image)
        return cars_info