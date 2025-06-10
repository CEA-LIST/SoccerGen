import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import json
from copy import deepcopy
from pathlib import Path
from typing import Union, Tuple, Optional
import random

## Utils

FIELD_DIMS = { # length x width
    "football" : (115, 74), # 115 yd × 74 yd for soccer
}


def getSoccerFieldPoints(sport):
    points = []
    u0 = 7
    n = 6
    r = 2 / (n - 1) * (FIELD_DIMS[sport][1] / n - u0)
    u = u0
    s = 0
    for j in range(0, n + 1):
        for i in range(0, 13):
            points.append([i * FIELD_DIMS[sport][0] / 12, FIELD_DIMS[sport][1] - s])
        s += u
        u += r

    return np.array([points], dtype=float)

IMG_WIDTH = 1280
IMG_HEIGHT = 720
soccerFieldPoints = getSoccerFieldPoints('football')
tr = soccerFieldPoints[0, 90].astype(int)
bl = soccerFieldPoints[0, 0].astype(int)
FIELD_HEIGHT = bl[1]
FIELD_WIDTH = tr[0]
field_heatmap = np.zeros((FIELD_HEIGHT, FIELD_WIDTH, 3))

for x in range(FIELD_WIDTH):
    for y in range(FIELD_HEIGHT):
        field_heatmap[y, x, 0] = y / (FIELD_HEIGHT-1)
        field_heatmap[y, x, 1] = x / (FIELD_WIDTH-1)
        field_heatmap[y, x, 2] = 0.5

##

PROMPT = "A TV broacast image of a professional soccer game. Players of the two teams are scattered around the field on a green grass, some running, some kicking the ball. White lines mark the boundaries of the field. Players of the same team have the same jersey and the two teams' jerseys are different."

class SoccerDatasetBase(Dataset):
    def __init__(
        self,
        root:Union[str, Path],
        size:Optional[int]=256,
        p_crop:Optional[float]=0.5,
        p_uncond:Optional[float]=0.1,
        max_crop_ratio:Optional[Tuple[float, float]]=(0.2, 0.3) 
    ):
        super().__init__()
        assert isinstance(root, str) or isinstance(root, Path), f'Unsupported type for argument "root": {type(root)}'

        if isinstance(root, str):
            self.root = Path(root)
        else:
            self.root = root

        self.size = size
        self.p_crop = p_crop
        self.p_uncond = p_uncond
        self.max_crop_ratio = max_crop_ratio
        self.image_files = list(map(lambda path: path.as_posix(), self.root.glob('*.jpg')))
        self._length = len(self.image_files)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        prompt = PROMPT
        if random.random() < self.p_uncond:
            prompt = ""
        

        image = (image / 127.5 - 1.0).astype(np.float32)

        return dict(image=image, txt=prompt)
    
class SoccerDatasetTrain(SoccerDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/Soccer/train', **kwargs)


class SoccerDatasetValidation(SoccerDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/Soccer/val', **kwargs)


class SoccerDatasetBaseV2(Dataset):
    def __init__(
        self,
        root:Union[str, Path],
        size:Optional[int]=256,
        p_augment:Optional[float]=0.,
        p_uncond:Optional[float]=0.1
    ):
        super().__init__()
        assert isinstance(root, str) or isinstance(root, Path), f'Unsupported type for argument "root": {type(root)}'

        if isinstance(root, str):
            self.root = Path(root)
        else:
            self.root = root

        self.size = size
        self.p_augment = p_augment
        self.p_uncond = p_uncond
        self.image_files = list(self.root.glob('**/*.jpg'))
        self._length = len(self.image_files)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.array(Image.open(self.image_files[index]).convert('RGB'))
        H, W, _ = image.shape
        if random.random() < self.p_augment:
            crop_start = np.random.randint(0, W-H)
        else:
            crop_start = (W-H) // 2
        image = image[:, crop_start:crop_start+H, :]
        image = cv2.resize(image, (self.size, self.size))
        image = (image / 127.5 - 1.0).astype(np.float32)

        prompt = PROMPT
        if random.random() < self.p_uncond:
            prompt = ""

        return dict(image=image, txt=prompt)
    
class SoccerDatasetTrainV2(SoccerDatasetBaseV2):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/Soccer/train', **kwargs)


class SoccerDatasetValidationV2(SoccerDatasetBaseV2):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/Soccer/val', **kwargs)

class SoccerConditionalBase(Dataset):
    def __init__(
        self,
        root:Union[str, Path],
        size:Optional[int]=256,
        p_tuncond:Optional[float]=0.1,  # probability of dropping text prompt
        p_allconds: Optional[float]=0.8,    # probability of keeping all spatial conditions
        p_suncond: Optional[float]=0.5, # probability of dropping each spatial conditions
        p_augment:Optional[float]=0.,
    ):
        super().__init__()
        assert isinstance(root, str) or isinstance(root, Path), f'Unsupported type for argument "root": {type(root)}'

        if isinstance(root, str):
            self.root = Path(root)
        else:
            self.root = root

        self.size = size
        self.p_tuncond = p_tuncond
        self.p_suncond = p_suncond
        self.p_allconds = p_allconds
        self.p_augment = p_augment
        self.metadata = json.load((self.root / "metadata.json").open())
        self._length = len(self.metadata)

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx:int):
        sample = self.metadata[idx]
        image_path = self.root / sample["image_path"]
        image = np.array(Image.open(image_path.as_posix()).convert("RGB"))
        h, w, _ = image.shape
        if random.random() < self.p_augment:
            crop_start = np.random.randint(0, w-h)
        else:
            crop_start = (w-h)//2
        
        image = image[:, crop_start:crop_start+h, :]
        image = cv2.resize(image, (self.size, self.size))
        image = (image / 127.5 - 1.0).astype(np.float32)

        detections = sample["detections"]
        new_detections = []

        positionxcolor_control = np.zeros_like(image, dtype=np.uint8)
        for detection in detections:
            x1 = int(round(detection['x1'] * w))
            x2 = int(round(detection['x2'] * w))

            x1 = x1 - crop_start
            x2 = x2 - crop_start

            if (x1 < h) and (x2 > 0):
                x1 = max(0, x1) / h
                x2 = min(x2, h) / h
                x1 = int(round(x1 * self.size))
                y1 = int(round(detection['y1'] * self.size))
                x2 = int(round(x2 * self.size))
                y2 = int(round(detection['y2'] * self.size))
                if detection['cls'] != 'ball':
                    color = detection['color']
                    cv2.rectangle(positionxcolor_control, (x1, y1), (x2, y2), color, -1)
                else:
                    radius = max(x2-x1, y2-y1)
                    center = ((x1+x2)//2, (y1+y2)//2)
                    cv2.circle(positionxcolor_control, center, radius, (255, 255, 255), -1)

                new_detection = deepcopy(detection)
                new_detection['x1'] = x1 / self.size
                new_detection['x2'] = x2 /  self.size
                new_detections.append(new_detection)

        positionxcolor_control = positionxcolor_control.astype(np.float32) / 127.5 - 1.0

        homography_path = self.root / sample["homography_path"]
        H = np.load(homography_path)
        heatmap = cv2.warpPerspective(field_heatmap, H, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)
        heatmap = heatmap[:, crop_start:crop_start+h, :]
        heatmap = cv2.resize(heatmap, (self.size, self.size))
        heatmap = heatmap * 2.0 - 1.0

        prompt = PROMPT
        if np.random.RandomState().rand() < self.p_tuncond:
            prompt = ""

        if np.random.RandomState().rand() >= self.p_allconds:
            u = np.random.RandomState().rand(2)
            if u[0] < self.p_suncond:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            if u[1] < self.p_suncond:
                positionxcolor_control = np.zeros_like(positionxcolor_control, dtype=np.float32)
        
        return dict(jpg=image, txt=prompt, positionxcolor=positionxcolor_control, calibration=heatmap, detections=new_detections, homography=H)
      
class SoccerConditionalTrain(SoccerConditionalBase):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/SoccerConditional/train', **kwargs)

class SoccerConditionalVal(SoccerConditionalBase):
    def __init__(self, **kwargs):
        super().__init__(root='datasets/SoccerConditional/val', **kwargs)