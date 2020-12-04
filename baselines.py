import torch
from torch import nn, optim
from dataset import COCO10SDataset
from torchvision import transforms, models, utils
import argparse
from models import COCO10Classifier


def main(args):
    # Initialize dataset and loaders
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
                                         img_dir='data/coco-10s-train/',
                                         transforms=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.ToTensor()]))
    test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                                   img_dir='data/coco-10s-test/',
                                   transforms=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.ToTensor()]))

    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compile model
    model = models.resnet18()
    model.fc = COCO10Classifier()

    criterion = nn.CrossEntropyLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    running_loss = 0

    i = 0
    # Train/test loop
    for epoch in range(args.epochs):
        for inputs, labels in train_loader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predicted = model.forward(inputs)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        predicted = model.forward(inputs)
                        batch_loss = criterion(predicted, labels)
                        test_loss += batch_loss.item()

                running_loss = 0
                model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

