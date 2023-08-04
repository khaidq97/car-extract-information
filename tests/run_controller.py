import cv2 
import time 
import shutil
from pathlib import Path 
from src.controller import Controller
from src.utils import draw_controller_result

def run_video(controller, video_path, video_process, save_path, save_draw, remove_draw):
    save_path = Path(save_path)
    save_path_video = save_path / 'videos'
    save_path_raw_video = save_path / 'raw_videos'
    save_path_video.mkdir(parents=True, exist_ok=True)
    if save_draw:
        save_path_raw_video.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(save_path_video / (Path(video_path).stem + '.avi')), fourcc, 20, (width, height))
    i = 0
    found_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = frame.copy()
            start_time = time.time()
            data, new_boxes, boxes = controller.run(img)
            end_time = time.time()
            fps = 1/(end_time - start_time)
            img = draw_controller_result(img, data,new_boxes, boxes, controller.limit_line, fps)
            if len(boxes):
                found_i += 1
            print(f"{i}:{total_frames}|{video_process}|found:{len(boxes)}|fps:{fps}")
            i+=1
            out.write(img)
            cv2.imshow('frame', cv2.resize(img, (1000,640)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        cap.release()
        out.release()
    cap.release()
    out.release()
    
    if save_draw and remove_draw and i !=0 and found_i/i > 0.4:
        shutil.copy2(str(video_path), str(save_path_raw_video / (Path(video_path).stem + '.avi')))


if __name__ == '__main__':
    import argparse
    import datetime 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', type=str, default='log path')
    parser.add_argument('--input-path', type=str, default='input path')
    parser.add_argument('--save-draw', action='store_true', default=False)
    parser.add_argument('--remove-draw', action='store_true', default=False)
    cfgs = parser.parse_args()
    
    log_path = cfgs.log_path
    input_path = Path(cfgs.input_path)
    
    # Log dict
    log_dir = Path(log_path)/ datetime.datetime.now().strftime('%Y%m%d_%H%M')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Test
    controller = Controller()
    
    if input_path.is_file():
        video_process = input_path.name
        run_video(controller, input_path, video_process, log_dir, save_draw=cfgs.save_draw, remove_draw=cfgs.remove_draw)
    elif input_path.is_dir():
        files = [x for x in input_path.rglob('*') if x.suffix in ['.mp4', '.avi']]
        for i, file in enumerate(files):
            video_process = f"{i}:{len(files)}|{file.name}"
            run_video(controller, file, video_process, log_dir, save_draw=cfgs.save_draw, remove_draw=cfgs.remove_draw)
    
    

