import torch
from torch import nn, optim, utils
from torch.utils.tensorboard import SummaryWriter
from dataset import COCO10SDataset, Resize, Compose, ToTensor, collate_fn
from torchvision import transforms, models
import argparse
from models import COCO10Classifier
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from train_utils import train_one_epoch, evaluate


def main(args):
    # Test dataset
    if args.test_grey and not args.test_B:
        json_file = 'data/coco-10s-test-A.json'
        img_dir = 'data/coco-10s-test-grey/'
    elif args.test_grey and args.test_B:
        json_file = 'data/coco-10s-test-B.json'
        img_dir = 'data/coco-10s-test-grey/'
    elif not args.test_grey and not args.test_B:
        json_file = 'data/coco-10s-test-A.json'
        img_dir = 'data/coco-10s-test/'
    elif not args.test_grey and args.test_B:
        json_file = 'data/coco-10s-test-B.json'
        img_dir = 'data/coco-10s-test/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet18':
        custom_collate_fn = None
        custom_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        bbox = False
        model = models.resnet18()
        model.fc = COCO10Classifier(in_size=512)
        criterion = nn.CrossEntropyLoss()

    elif args.model == 'resnet50':
        custom_collate_fn = None
        custom_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        bbox = False
        model = models.resnet50()
        model.fc = COCO10Classifier(in_size=2048)
        criterion = nn.CrossEntropyLoss()
    
    elif args.model == 'faster-rcnn':
        custom_collate_fn = collate_fn
        custom_transforms = Compose([Resize(256, 256), ToTensor()])
        bbox = True
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # 11 categories includes 10 COCO-10S categories + 1 FRCNN background category
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)

    # Initialize dataset and loaders
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
                                   set_type='train', img_dir='data/coco-10s-train/',
                                   modality=args.modality,
                                   bbox=bbox,
                                   transforms=custom_transforms)
    test_dataset = COCO10SDataset(json_file=json_file,
                                  set_type='val', img_dir=img_dir,
                                  modality=args.modality,
                                  bbox=bbox,
                                  transforms=custom_transforms)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=custom_collate_fn, num_workers=4)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn,
                                        num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    model.to(device)

    # Initialize Tensorboard
    writer = SummaryWriter()

    best_accuracy = 0
    # Train/test loop
    for epoch in range(args.epochs):
        if args.model == 'faster-rcnn':
            train_one_epoch(args, model, train_loader, epoch, device, writer, optimizer=optimizer, criterion=None)
            test_accuracy = evaluate(args, model, test_loader, epoch, device, writer=writer, criterion=None)
        else:
            train_one_epoch(args, model, train_loader, epoch, device, writer, criterion=criterion, optimizer=optimizer)
            test_accuracy = evaluate(args, model, test_loader, epoch, device, writer=writer, criterion=criterion)

        if test_accuracy > best_accuracy:
            # Save model weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, '%s_best.pt' % args.model)
            best_accuracy = test_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_grey', type=int, default=0)
    parser.add_argument('--test_B', type=int, default=0, help='Use name-B')
    parser.add_argument('--modality', type=str, default='vision', help='vision, lang')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

