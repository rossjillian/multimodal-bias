import torch
from torch import nn, optim
from dataset import COCO10SDataset
from torchvision import transforms, models, utils
import argparse
from models import COCO10Classifier
import statistics


def main(args):
    # Initialize dataset and loaders
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
                                        set_type='train',
                                        img_dir='data/coco-10s-train/',
                                        transforms=transforms.Compose([
                                                   transforms.Resize(256),
                                                   transforms.ToTensor()]))
    if args.test_grey:
        test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                                      set_type='val',
                                      img_dir='data/val2014-grey/',
                                      transforms=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.ToTensor()]))
    else:
        test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                                       set_type='val',
                                       img_dir='data/val2014/',
                                       transforms=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.ToTensor()]))

    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compile model
    model = models.resnet18()
    # 10 way classification
    model.fc = COCO10Classifier()

    criterion = nn.CrossEntropyLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model.to(device)

    running_loss = 0
    best_accuracy = 0

    i = 0
    # Train/test loop
    for epoch in range(args.epochs):
        cur_accuracy = []
        for inputs, labels in train_loader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                test_loss = 0
                correct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        test_loss += batch_loss.item()

                        predicted = torch.argmax(output, dim=1)
                        correct += (predicted == labels).sum()
                        total += labels.size(0)
                        accuracy = correct / total
                        print('Accuracy: %f' % accuracy)
                        cur_accuracy.append(accuracy)

                running_loss = 0
                model.train()

        if statistics.mean(cur_accuracy) > best_accuracy:
            # Save model weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'resnet_best.pt')
            best_accuracy = statistics.mean(cur_accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_grey', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

