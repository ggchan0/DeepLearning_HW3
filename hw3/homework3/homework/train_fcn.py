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
EARLY_STOP = 10

def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None

    validation_accuracy = 0

    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loader = load_dense_data(TRAIN_PATH, num_workers=4, batch_size=200)

    for epoch in range(15):
        model.train()
        running_loss = 0.0

        for index, (inputs, label) in enumerate(data_loader):
            print("input shape", inputs.shape)
            print("label shape", label.shape)
            print(label[0])
            image_to_tensor = dense_transforms.ToTensor()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("loss at epoch ", epoch, running_loss)

        model.eval()

        confusion = ConfusionMatrix(6)
        for img, label in load_dense_data(VALID_PATH):
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
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
