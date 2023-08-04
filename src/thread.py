import cv2
import time
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from controller import Controller
from utils import draw_controller_result

controller = Controller()

class Thread(QThread):
    changePixmap = pyqtSignal(QImage, str, float, float, float) 
    cap = None
    
    def run(self):
        name = 'Car'
        height = 0.0
        width = 0.0
        length = 0.0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            img = frame.copy()
            start_time = time.time()
            data, new_boxes, boxes = controller.run(img)
            end_time = time.time()
            fps = 1/(end_time - start_time)
            img = draw_controller_result(img, data,new_boxes, boxes, controller.limit_line, fps)
            try:
                name_ = data[0]['model']
                height_ = int(data[0]['height'])
                width_ = int(data[0]['width'])
                length_ = int(data[0]['length'])
            except:
                name_ = None
                height_ = None
                width_ = None
                length_ = None 
            if name_ is not None:
                name = name_
                height = height_
                width = width_
                length = length_
            
            # Display the resulting frame
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat, name, height, width, length)
        self.cap.release()

    def stop(self):
        self.terminate()