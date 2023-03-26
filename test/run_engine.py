import cv2
import os 
import random
from datetime import datetime
import argparse
from pathlib import Path 
import time
from lib.controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='/media/khai/data/CAR_VAN/')
parser.add_argument('--log_dir', default='logs')

if __name__ == '__main__':
    # Get parameters
    args = parser.parse_args()
    image_dir = Path(args.image_dir)
    log_dir = Path(args.log_dir)

    # save logs
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_debug_path = log_dir / timestamp
    os.makedirs(str(log_debug_path), exist_ok=True)
    
    # Get Engine
    controller = Controller()

    imfiles = [x for x in image_dir.glob('*/*') if x.suffix in ('.jpg', '.png', '.jpeg' ,'.JPG', '.PNG', '.JPEG')]
    random.shuffle(imfiles)
    for i,imfile in enumerate(imfiles[:200]):
        image = cv2.imread(str(imfile))
        os.makedirs(str(log_debug_path / imfile.stem), exist_ok=True)
        t1 = time.time()
        infos = controller.run(image, str(log_debug_path / imfile.stem))
        t2 = time.time()
        print("{}||{}||{}||{}".format(i, len(imfiles), str(imfile),t2-t1))