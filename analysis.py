from dataset import COCO10SDataset
from torchvision import transforms, models
from torch import utils
import torch
import argparse
from models import COCO10Classifier
from tqdm import tqdm


def confusion_matrix(predicted, labels, conf_matrix):
    """
    Credit: https://discuss.pytorch.org/t/how-to-check-and-read-confusion-matrix/41835/3
    """
    predicted = torch.argmax(predicted, 1)
    for p, t in zip(predicted, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def main(args):
    if args.modality == 'vision':
        test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                set_type='val',
                img_dir='data/coco-10s-test/',
                transforms=transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()]))
    
    test_loader = utils.data.DataLoader(test_dataset, batch_size=32)

    if args.model == 'resnet18':
        model = models.resnet18()
        model.fc = COCO10Classifier()
        model = torch.load(pretrained)

    conf_matrix = torch.zeros(10, 10)
    with tqdm(test_loader, unit='batch') as tepoch:
        for inputs, labels in tepoch:
            model.eval()
            with torch.no_grad():
                output = model.forward(inputs)
                conf_matrix = confusion_matrix(output, labels, conf_matrix)
    
    print(conf_matrix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='vision', help='vision, lang')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--pretrained', type=str, help='Data path to pretrained weights')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

