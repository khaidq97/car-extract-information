from .car_classifier import CarClassifier
from .car_detector import CardDetector
from .extract_info_engine import ExtractInfoEngine


class Controller:
    def __init__(self):
        self.car_classifier = CarClassifier()
        self.car_detector = CardDetector()
        self.extract_info_engine = ExtractInfoEngine()

    def run(self, image, debug_dir=None):
        cars_box = self.car_detector.run(image, debug_dir)
        cars_id = self.car_classifier.run(image, cars_box, debug_dir)
        cars_info = self.extract_info_engine.run(image, cars_box, cars_id, debug_dir)
        return cars_info