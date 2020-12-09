from torch.utils.data import Dataset
import json
import os
from PIL import Image


class COCO10SDataset(Dataset):
    def __init__(self, img_dir, set_type, json_file, modality, transforms=None):
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
        self.categories = {'train': 0, 'bench': 1, 'dog': 2, 'umbrella': 3, 'skateboard': 4,
                           'pizza': 5, 'chair': 6, 'laptop': 7, 'sink': 8, 'clock': 9}

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, idx):
        if self.modality == 'vision':
            # Gets picture
            img_path = os.path.join(self.img_dir, 'COCO_' + self.set_type + '2014_' + list(self.json_dict.keys())[idx] + '.jpg')
            item = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                item = self.transforms(item)
        elif self.modality == 'lang':
            # Gets caption
            item = self.json_dict[(list(self.json_dict.keys())[idx])][1] 

        # Gets category label
        label = self.categories[self.json_dict[(list(self.json_dict.keys())[idx])][0]]
        return item, label


