import numpy as np
import cv2

def is_point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_rectangle_inside_polygon(rectangle, polygon):
    xmin, ymin, xmax, ymax = rectangle
    rectangle_points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
    for point in rectangle_points:
        x, y = point
        if not is_point_in_polygon(x, y, polygon):
            return False
    return True

def draw_limit_line(image,
                    limit_line,
                    color=(0, 0, 255)):
    polygon_points = np.array(limit_line, np.int64)
    cv2.polylines(image, [polygon_points], isClosed=True, color=color, thickness=2)
    return image

def draw_controller_result(img, 
                           data=None,
                           data_boxes=None, 
                           boxes=None,
                           limit_line=None,
                           fps=None):
    if fps is not None:
        cv2.putText(img, f"FPS:{round(fps,2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    if limit_line is not None and len(limit_line):
        img = draw_limit_line(img, limit_line, (255,0,0)) 
    
    if boxes is not None and len(boxes):
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            
    if data is not None and len(data):
        for d, box in zip(data, data_boxes):
            xmin, ymin, xmax, ymax = box
            ymin = ymax
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, d['model'], (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(img, f"Height:{d['height']}|Length:{d['length']}|Width:{d['width']}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return img