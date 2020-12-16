from dataset import COCO10SDataset, Resize, Compose, ToTensor, collate_fn
from torchvision import transforms, models
from torch import utils
import torch
import argparse
from models import COCO10Classifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def confusion_matrix(predicted, labels, conf_matrix):
    """
    Credit: https://discuss.pytorch.org/t/how-to-check-and-read-confusion-matrix/41835/3
    """
    for p, t in zip(predicted, labels):
        # Row is true, Column is predicted
        conf_matrix[t, p] += 1
    return conf_matrix


def plot_confusion_matrix(conf_matrix, model):
    conf_matrix = conf_matrix.numpy()
    if model == 'resnet18' or model == 'resnet50':
        categories_ordered = ['train', 'bench', 'dog', 'umbrella', 'skateboard', 'pizza', 'chair', 'laptop', 
                'sink', 'clock']
    else:
        categories_ordered = ['background', 'bench', 'dog', 'umbrella', 'skateboard', 'pizza', 'chair', 'laptop'
                , 'sink', 'clock', 'train']

    df_conf = pd.DataFrame(conf_matrix, index=categories_ordered, columns=categories_ordered)
    sn.heatmap(df_conf)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('%s.png' % model)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet18' or args.model == 'resnet50':
        bbox = False
    
    elif args.model == 'faster-rcnn':
        bbox = True
    
    if args.modality == 'vision':
        if args.model == 'faster-rcnn':
            test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                set_type='val',
                img_dir='data/coco-10s-test/',
                modality=args.modality,
                bbox=bbox,
                transforms=Compose([
                    Resize(256, 256),
                    ToTensor()]))
            test_loader = utils.data.DataLoader(test_dataset, num_workers=4, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size)
        else:
            test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                    set_type='val',
                    img_dir='data/coco-10s-test/',
                    modality=args.modality, 
                    bbox=bbox,
                    transforms=transforms.Compose([                                                                                        transforms.Resize((256, 256)),
                        transforms.ToTensor()]))
            test_loader = utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    checkpoint = torch.load(args.pretrained)
    if args.model == 'resnet18':
        model = models.resnet18()
        model.fc = COCO10Classifier(in_size=512)

    elif args.model == 'resnet50':
        model = models.resnet50()
        model.fc = COCO10Classifier(in_size=2048)

    elif args.model == 'faster-rcnn':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if args.model == 'faster-rcnn':
        conf_matrix = torch.zeros(11, 11)
    else:
        conf_matrix = torch.zeros(10, 10)

    with tqdm(test_loader, unit='batch') as tepoch:
        for inputs, labels in tepoch:
            model.eval()
            with torch.no_grad():
                
                if args.model == 'faster-rcnn':  
                    inputs = list(image.to(device) for image in inputs)     
                    labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                else:                                                                                                                inputs, labels = inputs.to(device), labels.to(device)
                
                output = model.forward(inputs)
                
                if args.model == 'faster-rcnn':
                    modified_output = []
                    modified_labels = []
                    for i, entry in enumerate(output):
                        if entry['labels'].nelement() == 0:
                            # If predicts no objects, assume background
                            modified_output.append(0)
                        else:
                            value, index = entry['scores'].max(0)
                            modified_output.append(entry['labels'][index].item())
                        modified_labels.append(labels[i]['labels'][0])

                    predicted = torch.as_tensor(modified_output)
                    labels = torch.as_tensor(modified_labels)
                
                else:
                    predicted = torch.argmax(output, 1)
                
                conf_matrix = confusion_matrix(predicted, labels, conf_matrix)
    
    
    plot_confusion_matrix(conf_matrix, args.model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--modality', type=str, default='vision', help='vision, lang')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--pretrained', type=str, help='Data path to pretrained weights')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

