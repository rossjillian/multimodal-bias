from torch.utils.data import Dataset
import json
import os
from PIL import Image
from pycococtools.coco import COCO


class COCO10SDataset(Dataset):
    def __init__(self, img_dir, set_type, json_file, annotation_file, modality, bbox=True, transforms=None):
        """
        :param img_dir: Directory with COCO-10S images (coco-10s-train)
        :param set_type: train or val
        :param json_file: Path to the JSON file with captions, category, and image IDs
        :param modality: vision, lang, or visionlang
        """
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        with open(json_file, 'r') as f:
            self.json_dict = json.load(f)
        self.transforms = transforms
        self.set_type = set_type
        self.modality = modality
        self.bbox = bbox
        self.categories = {'train': 0, 'bench': 1, 'dog': 2, 'umbrella': 3, 'skateboard': 4,
                           'pizza': 5, 'chair': 6, 'laptop': 7, 'sink': 8, 'clock': 9}

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, idx):
        """
        bbox code credit: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
        """
        img_id = list(self.json_dict.keys())[idx]
        if self.modality == 'vision':
            # Bounding box
            # TODO: move to create_dataset.py
            if self.bbox:
                ann_id = self.coco.getAnnIds(imgIds=img_id)
                coco_ann = self.coco.loadAnns(ann_id)
                boxes = []
                for i in range(len(coco_ann)):
                    xmin = coco_ann[i]['bbox'][0] 
                    ymin = coco_ann[i]['bbox'][1]
                    xmax = xmin + coco_ann[i]['bbox'][2]
                    ymax = ymin + coco_ann[i]['bbox'][3]
                    boxes.append([xmin, ymin, xmax, ymax])
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Gets picture
            img_path = os.path.join(self.img_dir, 'COCO_' + self.set_type + '2014_' + img_id + '.jpg')
            item = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                item = self.transforms(item)
        elif self.modality == 'lang':
            # Gets caption
            item = self.json_dict[img_id][1] 

        # Gets category label
        label = self.categories[self.json_dict[img_id][0]]
        return item, label


