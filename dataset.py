from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
from torchvision import transforms


"""
Custom COCO10S Dataset Class
"""


class COCO10SDataset(Dataset):
    def __init__(self, img_dir, set_type, json_file, modality, bbox=True, transforms=None):
        """
        :param img_dir: Directory with COCO-10S images (coco-10s-train)
        :param set_type: train or val
        :param json_file: Path to the JSON file with captions, category, and image IDs
        :param modality: vision, lang, or visionlang
        """
        self.img_dir = img_dir
        with open(json_file, 'r') as f:
            self.json_dict = json.load(f)
        self.transforms = transforms
        self.set_type = set_type
        self.modality = modality
        self.bbox = bbox
        self.categories = {'train': 10, 'bench': 1, 'dog': 2, 'umbrella': 3, 'skateboard': 4,
                           'pizza': 5, 'chair': 6, 'laptop': 7, 'sink': 8, 'clock': 9}

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, idx):
        """
        bbox code credit: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
        """
        img_id = list(self.json_dict.keys())[idx]
        
        # Gets category label
        label = self.categories[self.json_dict[img_id][0]]
        
        if self.bbox:
            boxes = self.json_dict[img_id][1][1]    
            num_bbox = len(boxes)
            target = {'labels': torch.tensor([label], dtype=torch.int64).repeat(num_bbox)}
        else:
            target = label

        if self.modality == 'vision':
            # Bounding box
            if self.bbox:
                boxes = self.json_dict[img_id][1][1]    
                target['boxes'] = boxes

            # Gets picture
            img_path = os.path.join(self.img_dir, 'COCO_' + self.set_type + '2014_' + img_id + '.jpg')
            item = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                if self.bbox:
                    item, bbox = self.transforms(item, target['boxes'])
                    target['boxes'] = bbox
                else:
                    item = self.transforms(item)

        elif self.modality == 'lang':
            # Gets caption
            item = self.json_dict[img_id][1][0]

        return item, target


"""
DataLoader
"""


def collate_fn(batch):
    """
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    """
    return tuple(zip(*batch))


"""
Custom Transforms
"""


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize(object):
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, img, bbox):
        w, h = img.size
        img = transforms.Resize((self.x_dim, self.y_dim))(img)
        
        scale_x = self.x_dim / w
        scale_y = self.y_dim / h

        for i in range(len(bbox)):
            # bbox: [xmin, ymin, xmax, ymax]
            # new_coord = old_coord * (new_size / old_size)
            bbox[i][0] = int(bbox[i][0] * scale_x) 
            bbox[i][1] = int(bbox[i][1] * scale_y)  
            bbox[i][2] = int(bbox[i][2] * scale_x)  
            bbox[i][3] = int(bbox[i][3] * scale_y)  

        return img, bbox


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, bbox):
        img = transforms.ToTensor()(img)
        bbox = torch.as_tensor(bbox, dtype=torch.int64)
        # bbox = torch.transpose(bbox, 0, 1) 
        
        return img, bbox


