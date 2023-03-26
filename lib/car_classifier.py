import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

ModelWeights = {
    'mobilenet_v2':'MobileNet_V2_Weights.IMAGENET1K_V1',
    'resnet18':'ResNet18_Weights.IMAGENET1K_V1',
    'resnet50':'ResNet50_Weights.IMAGENET1K_V1',
    'resnet101' : 'ResNet101_Weights.IMAGENET1K_V1',
    'swin_s': 'Swin_S_Weights.IMAGENET1K_V1',
    'swin_b': 'Swin_B_Weights.IMAGENET1K_V1',
    'vit_b_16': 'ViT_B_16_Weights.IMAGENET1K_V1',
    'vit_b_32' : 'ViT_B_32_Weights.IMAGENET1K_V1',
    'vit_l_16': 'ViT_L_16_Weights.IMAGENET1K_V1',
    'vit_l_32': 'ViT_L_32_Weights.IMAGENET1K_V1',
    'efficientnet_v2_m': 'EfficientNet_V2_M_Weights.IMAGENET1K_V1'
}

class ClsModel(nn.Module):
    def __init__(self, model_name, num_classes, is_pretrained=False):
        super(ClsModel, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        self.is_pretrained = is_pretrained

        if self.model_name not in ModelWeights:
            raise ValueError('Please confirm the name of model')

        if self.is_pretrained:
            self.base_model = getattr(torchvision.models, self.model_name)(weights=ModelWeights[self.model_name])
        else:
            self.base_model = getattr(torchvision.models, self.model_name)()

        if hasattr(self.base_model, 'classifier'):
            self.base_model.last_layer_name = 'classifier'
            feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'fc'):
            self.base_model.last_layer_name = 'fc'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.fc = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'head'):
            self.base_model.last_layer_name = 'head'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.head = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'heads'):
            self.base_model.last_layer_name = 'heads'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.heads = nn.Linear(feature_dim, self.num_class)
        else:
            raise ValueError('Please confirm the name of last')

    def forward(self, x):
        x = self.base_model(x)
        return x


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
def to_tensor(x):
    x = normalize(x)
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).float()

def normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD, value=255.0):
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image / value
    image = (image - mean) / std
    return image

################################################################################
from .config import CAR_CLASSIFICATION_MODEL_PATH
import os

class CarClassifier:
    def __init__(self, weight=CAR_CLASSIFICATION_MODEL_PATH,
                       model_name='efficientnet_v2_m',
                       num_classes=78,
                       device='cuda:0',
                       imgz=112):
        self.imgz, self.device = imgz, device
        self.model = ClsModel(model_name, num_classes, False).to(device)
        self.model.load_state_dict(torch.load(weight, map_location=device))
        self.model.eval()
    
    @torch.no_grad()
    def run(self, image, cars_box, debug_dir=None):
        cars_id = []
        cars_img = []
        for box in cars_box:
            car_img = image.copy()[box[1]:box[3],box[0]:box[2]]
            car_img = cv2.resize(car_img, (self.imgz, self.imgz))
            car_img = to_tensor(car_img).unsqueeze_(0)
            cars_img.append(car_img)
        if len(cars_img):
            cars_img = torch.concat(cars_img, dim=0)
            cars_img = cars_img.to(self.device)
            outs = self.model(cars_img)
            outs = torch.nn.functional.softmax(outs, dim=1)
            _, preds = torch.max(outs, dim=1)
            cars_id = preds.cpu().numpy()
            if debug_dir:
                debug_image = image.copy()
                for box, id in zip(cars_box, cars_id):
                    cv2.rectangle(debug_image, box[:2], box[2:], (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{id}", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 255, 255), 3)
                    cv2.imwrite(os.path.join(debug_dir, "car_classification.png"), debug_image)
        return cars_id