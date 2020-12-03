from torch.utils.data import Dataset
import json
import os
from PIL import Image


class COCO10SDataset(Dataset):
    def __init__(self, img_dir, json_file, transforms=None):
        """
        :param img_dir: Directory with COCO-10S images (coco-10s-train)
        :param json_file: Path to the JSON file with captions, category, and image IDs
        """
        self.img_dir = img_dir
        with open(json_file, 'r') as f:
            self.json_dict = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, idx):
        # TODO: improve how data is stored for less hard-coding of file name
        # Gets picture
        img_path = os.path.join(self.img_dir, 'COCO_train2014_' + str(idx) + '.jpg')
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)

        # Gets category label
        label = self.json_dict[idx][0]
        return img, label
