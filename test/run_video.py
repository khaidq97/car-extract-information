import cv2
import argparse 
import sys 
from pathlib import Path 
from tqdm import tqdm
from lib.controller import Controller

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

parser = argparse.ArgumentParser()
parser.add_argument('--video-path', type=str ,
                    default='/home/khai/Desktop/Projects/van_lvtp/data/videos/video35.mp4')
parser.add_argument('--save-path', type=str,
                    default='out.mp4')
parser.add_argument('--show', action='store_true', help='Show video, press q to exit')

def draw_infos(image, infos):
    for info in infos:
        box = info['box']
        model = info['model']
        height = info['height']
        length = info['length']
        width = info['width']

        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), 2)
        y = box[1] + 10
        x = (box[0]+box[2])//4
        cv2.putText(image, f"Name: {model}", (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
        cv2.putText(image, f"Length: {length} (mm)", (x, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
        cv2.putText(image, f"Width: {width} (mm)", (x, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
        cv2.putText(image, f"Height: {height} (mm)", (x, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
    return image


def count_total_frame_video(video_path):
    num_frame = 0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret: break 
        num_frame += 1
    cap.release()
    return num_frame


if __name__ == '__main__':
    # Get parameters
    args = parser.parse_args()
    video_path = args.video_path
    save_path = args.save_path
    show = args.show

    # Get Engine
    controller = Controller()

    # Cap inputvideo
    cap = cv2.VideoCapture(video_path)

    # Get the parameters of the video (size, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = count_total_frame_video(video_path)

    # Create a VideoWriter object to write the video to file
    fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    video_writer = cv2.VideoWriter(save_path, fourcc, 25, (width, height))

    i = 0
    tq = tqdm(total=frame_count)
    while cap.isOpened():
        tq.update(1)
        ret, frame = cap.read()
        if not ret: break 

        infos = controller.run(frame)
        image = draw_infos(frame.copy(), infos)
        video_writer.write(image)
        
        if show:
            cv2.imshow('video', image)
            if cv2.waitKey(5) & 0xff == 27:
                break 

    cv2.destroyAllWindows()
    cap.release()
    video_writer.release()


