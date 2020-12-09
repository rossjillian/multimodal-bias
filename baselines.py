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

    # Train dataset
    train_dataset = COCO10SDataset(json_file='data/coco-10s-train.json',
            set_type='train', img_dir='data/coco-10s-train/',
            modality = args.modality,
            transforms=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

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
    elif not agrs.test_grey and args.test_B:
        json_file = 'data/coco-10s-test-B.json'
        img_dir = 'data/coco-10s-test/'

    test_dataset = COCO10SDataset(json_file=json_file,
            set_type='val', img_dir=img_dir,
            modality=args.modality, 
            transforms=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

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
        train_loss = []
        with tqdm(train_loader, unit='batch') as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)

                train_loss.append(loss.item())

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
        writer.add_scalar("Loss/train", statistics.mean(train_loss), epoch)

        print("Test")
        test_acc = []
        test_loss = []
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, labels in tepoch:
                model.eval()
                with torch.no_grad():
                    tepoch.set_description(f"Epoch {epoch}")
                        
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    loss = criterion(output, labels)

                    test_loss.append(loss.item())

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
        writer.add_scalar("Loss/test", statistics.mean(test_loss), epoch)

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
    parser.add_argument('--test_B', type=int, default=0, help='Use name-B')
    parser.add_argument('--modality', type=str, default='vision', help='vision, lang')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

