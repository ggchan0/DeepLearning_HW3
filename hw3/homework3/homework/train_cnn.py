from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.nn as nn
import torch.optim as optim

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"
EARLY_STOP = 15

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(args):
    from os import path
    model = CNNClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = None
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loader = load_data(TRAIN_PATH, batch_size=args.batch_size)
    validation_accuracy = 0
    for epoch in range(0, args.epochs):
        model.train()
        running_loss = 0.0
        for index, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("loss at epoch ", epoch, running_loss)
        model.eval()
        confusion = ConfusionMatrix(6)
        for img, label in load_data(VALID_PATH):
            confusion.add(model(img.to(device)).argmax(1).cpu(), label)
        print("global accuracy: ", confusion.global_accuracy)
        if validation_accuracy > confusion.global_accuracy and epoch > EARLY_STOP:
            exit()
        else:
            validation_accuracy = confusion.global_accuracy
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default="adam")
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
