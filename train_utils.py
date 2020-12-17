from tqdm import tqdm
import torch
import statistics


def train_one_epoch(args, model, train_loader, epoch, device, writer=None, criterion=None, optimizer=None):
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
                inputs, targets = inputs.to(device), targets.to(device)

            if args.model == 'faster-rcnn':
                loss_dict = model(inputs, targets)
                losses = sum(loss for loss in loss_dict.values())
            else:
                output = model.forward(inputs)
                losses = criterion(output, targets)

            train_loss.append(losses.item())

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

                tepoch.set_postfix(loss=losses.item(), accuracy=accuracy)
            else:
                tepoch.set_postfix(loss=losses.item())

    if args.model != 'faster-rcnn':
        print("Average train accuracy")
        print(statistics.mean(train_acc))

        writer.add_scalar("Accuracy/train", statistics.mean(train_acc), epoch)

    writer.add_scalar("Loss/train", statistics.mean(train_loss), epoch)


def evaluate(args, model, test_loader, epoch, device, writer=None, criterion=None):
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
                    inputs, targets = inputs.to(device), targets.to(device)

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
                tepoch.set_postfix(accuracy=accuracy)

    print("Average test accuracy")
    print(statistics.mean(test_acc))

    writer.add_scalar("Accuracy/test", statistics.mean(test_acc), epoch)

    if args.model != 'faster-rcnn':
        writer.add_scalar("Loss/test", statistics.mean(test_loss), epoch)

    return statistics.mean(test_acc)

