import torch
from torch import nn, optim, utils
from torch.utils.tensorboard import SummaryWriter
from dataset import COCO10SDataset
from torchvision import transforms, models
import argparse
from models import COCO10Classifier
import statistics
from tqdm import tqdm


def main(args):
    # Initialize dataset and loaders
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
                                        set_type='train',
                                        img_dir='data/coco-10s-train/',
                                        transforms=transforms.Compose([
                                                   transforms.Resize((256, 256)),
                                                   transforms.ToTensor()]))
    if args.test_grey:
        test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                                      set_type='val',
                                      img_dir='data/coco-10s-test-grey/',
                                      transforms=transforms.Compose([
                                          transforms.Resize((256, 256)),
                                          transforms.ToTensor()]))
    else:
        test_dataset = COCO10SDataset(json_file='data/coco-10s-test.json',
                                       set_type='val',
                                       img_dir='data/coco-10s-test/',
                                       transforms=transforms.Compose([
                                           transforms.Resize((256, 256)),
                                           transforms.ToTensor()]))

    train_loader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compile model
    model = models.resnet18()
    # 10 way classification
    model.fc = COCO10Classifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(device)

    best_accuracy = 0

    # Initialize Tensorboard
    writer = SummaryWriter()

    # Train/test loop
    for epoch in range(args.epochs):
        model.train()
        print("Train")
        train_acc = []
        with tqdm(train_loader, unit='batch') as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                # Calculate per batch accuracy
                predicted = torch.argmax(output, dim=1)
                correct = (labels == predicted).float().sum().item()
                total = args.batch_size
                accuracy = correct / total
                train_acc.append(accuracy)

                tepoch.set_postfix(loss = loss.item(), accuracy = accuracy)
                
        print("Average train accuracy")
        print(statistics.mean(train_acc))
        
        writer.add_scalar("Accuracy/train", statistics.mean(train_acc), epoch)

        print("Test")
        test_acc = []
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, labels in tepoch:
                model.eval()
                with torch.no_grad():
                    tepoch.set_description(f"Epoch {epoch}")
                        
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    loss = criterion(output, labels)

                    # Calculate per batch accuracy
                    predicted = torch.argmax(output, dim=1)
                    correct = (labels == predicted).float().sum().item()
                    total = args.batch_size
                    accuracy = correct / total
                    test_acc.append(accuracy)

                    tepoch.set_postfix(loss = loss.item(), accuracy = accuracy)

                model.train()
        
        print("Average test accuracy")
        print(statistics.mean(test_acc))

        writer.add_scalar("Accuracy/test", statistics.mean(test_acc), epoch)

        if statistics.mean(test_acc) > best_accuracy:
            # Save model weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'resnet_best.pt')
            best_accuracy = statistics.mean(test_acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_grey', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

