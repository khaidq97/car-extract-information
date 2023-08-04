import cv2 
import time
import numpy as np
import onnxruntime as ort

def clip_coords(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (numpy array of shape [N, 4]): N bounding boxes in (x1, y1, x2, y2) format.
        box2 (numpy array of shape [M, 4]): M bounding boxes in (x1, y1, x2, y2) format.
    Returns:
        iou (numpy array of shape [N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box1 and box2.
    """
    def box_area(box):
        # box = 4xN
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter_left_top = np.maximum(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    inter_right_bottom = np.minimum(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

    inter = np.prod(np.maximum(inter_right_bottom - inter_left_top, 0), axis=2)  # (N, M)

    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def letterbox(im,
            new_shape=(640, 640),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False, 
            scaleup=True,
            stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        return im, ratio, (dw, dh)
    
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[np.arange(len(l)), l[:, 0].astype(int) + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(float)), axis=1)
        else:  # best class only
            conf = x[:, 5:].max(1)
            j = x[:, 5:].argmax(1)
            x = np.concatenate((box, conf[:, None], j[:, None].astype(float)), axis=1)[conf > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)[:, None]).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]) / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        return output[0]

class Yolov5:
    def __init__(self,
                 model_path,
                 device='cpu',
                 conf_thres=0.6,
                 iou_thres=0.5,
                 max_det=100,
                 img_shape=(640,640)):
        self.img_shape = img_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        self.net = ort.InferenceSession(model_path, providers=providers)
        
    def run(self, img):
        shapeOrigin = img.shape[:2]
        img = self.__process_input(img)
        img_input = {self.net.get_inputs()[0].name: img}
        pred= self.net.run(None, img_input)[0]
        boxes, labels, scores = self.__process_output(
            pred=pred,
            shapeOrigin=shapeOrigin,
            shapeScale=self.img_shape,
        )
        return boxes, labels, scores
        
    def __process_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img, new_shape=self.img_shape)[0]
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img) / 255.0
        return img.astype(np.float32)
    
    def __process_output(self, pred, shapeOrigin, shapeScale):
        pred = non_max_suppression(
            prediction=pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )
        boxes, labels, scores = [], [], []
        if pred is not None:
            pred[:, :4] = scale_coords(shapeScale, pred[:, :4], shapeOrigin).round()  # xyxy format
            boxes, labels, scores = pred[:,:4].astype(int), pred[:,5].astype(int) , pred[:,4]
        return boxes, labels, scores
    
if __name__ == '__main__':
    model = Yolov5(model_path='../trained_models/best.onnx', device='cpu')
    
    fake_img = cv2.imread('../document/car_data_yolo/images/video01_28.png')
    boxes, labels, scores = model.run(fake_img)
    print(boxes, labels, scores)