import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision
import torch.nn as nn
import torch.optim as optim

TRAIN_PATH = "dense_data/train"
VALID_PATH = "dense_data/valid"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(args):
    from os import path
    model = FCN().to(device)
    train_logger, valid_logger = None, None

    validation_accuracy = 0

    loss_function = nn.CrossEntropyLoss()
    optimizer = None
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    data_loader = load_dense_data(TRAIN_PATH, num_workers=4, batch_size=200)

    for epoch in range(15):
        model.train()
        running_loss = 0.0

        for index, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = inputs.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels.sum(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()

        confusion = ConfusionMatrix(6)
        for img, label in load_dense_data(VALID_PATH):
            confusion.add(model(img.to(device)).argmax(1).cpu(), label)

        print("loss at epoch ", epoch, running_loss, "global accuracy: ", confusion.global_accuracy)

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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=5000)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
