import torch
from torch import nn, optim, utils
from torch.utils.tensorboard import SummaryWriter
from dataset import COCO10SDataset, Resize, Compose, ToTensor, collate_fn
from torchvision import transforms, models
import argparse
from models import COCO10Classifier
import statistics
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def main(args):
    # Initialize dataset and loaders

    # Train dataset
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
            set_type='train', img_dir='data/coco-10s-train/',
            modality = args.modality,
            # Custom Resize transform for bbox
            transforms=Compose([Resize(256, 256), ToTensor()]))

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

    # TODO: remove when language is integrated
    json_file = 'data/coco-10s-test.json'

    test_dataset = COCO10SDataset(json_file=json_file,
            set_type='val', img_dir=img_dir,
            modality=args.modality,
            transforms=Compose([Resize(256, 256), ToTensor()]))

    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet18':
        model = models.resnet18()
        model.fc = COCO10Classifier(in_size=512)
        criterion = nn.CrossEntropyLoss()

    elif args.model == 'resnet50':
        model = models.resnet50()
        model.fc = COCO10Classifier(in_size=2048)
        criterion = nn.CrossEntropyLoss()
    
    elif args.model == 'faster-rcnn':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # 11 categories includes 10 COCO-10S categories + 1 FRCNN background category
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)

    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    model.to(device)

    best_accuracy = 0

    # Initialize Tensorboard
    writer = SummaryWriter()

    # Train/test loop
    for epoch in range(args.epochs):
        model.train()
        print("Train")
        train_acc = []
        train_loss = []
        with tqdm(train_loader, unit='batch') as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
               
                if args.model == 'faster-rcnn': 
                    inputs = list(image.to(device) for image in inputs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                else:
                     inputs, labels = inputs.to(device), labels.to(device)

                if args.model == 'faster-rcnn':
                    loss_dict = model(inputs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    output = model.forward(inputs)
                    loss = criterion(output, labels)
                    losses = loss.item()

                if args.model == 'faster-rcnn':
                    train_loss.append(losses.item())
                else:
                    train_loss.append(losses)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if args.model != 'faster-rcnn':
                    # Calculate per batch accuracy
                    predicted = torch.argmax(output, dim=1)
                    correct = (targets == predicted).float().sum().item()
                    total = args.batch_size
                    accuracy = correct / total
                    train_acc.append(accuracy)

                    tepoch.set_postfix(loss = loss.item(), accuracy = accuracy)
                else:
                    tepoch.set_postfix(loss = losses.item())
 
        if args.model != 'faster-rcnn':
            print("Average train accuracy")
            print(statistics.mean(train_acc))
        
            writer.add_scalar("Accuracy/train", statistics.mean(train_acc), epoch)
        
        writer.add_scalar("Loss/train", statistics.mean(train_loss), epoch)

        print("Test")
        test_acc = []
        test_loss = []
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, targets in tepoch:
                model.eval()
                with torch.no_grad():
                    tepoch.set_description(f"Epoch {epoch}")
                        
                    if args.model == 'faster-rcnn': 
                        inputs = list(image.to(device) for image in inputs)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    else:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                    output = model.forward(inputs)
                    
                    total = args.batch_size
                    if args.model != 'faster-rcnn':
                        loss = criterion(output, targets)
                        test_loss.append(loss.item())

                        predicted = torch.argmax(output, dim=1)
                        correct = (targets == predicted).float().sum().item()
                    else:
                        correct = 0
                        # Iterate over predicted
                        for i, entry in enumerate(output):
                            # Check if predicts no boxes
                            if entry['labels'].nelement() == 0:
                                pass
                            else:
                                value, index = entry['scores'].max(0)
                                if entry['labels'][index].item() == int(targets[i]['labels'][0]):
                                    correct += 1

                    accuracy = correct / total
                    
                    test_acc.append(accuracy)
                    tepoch.set_postfix(accuracy = accuracy)

        
        print("Average test accuracy")
        print(statistics.mean(test_acc))

        writer.add_scalar("Accuracy/test", statistics.mean(test_acc), epoch)
        
        if args.model != 'faster-rcnn':
            writer.add_scalar("Loss/test", statistics.mean(test_loss), epoch)

        if statistics.mean(test_acc) > best_accuracy:
            # Save model weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'rcnn_best.pt')
            best_accuracy = statistics.mean(test_acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_grey', type=int, default=0)
    parser.add_argument('--test_B', type=int, default=0, help='Use name-B')
    parser.add_argument('--modality', type=str, default='vision', help='vision, lang')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

