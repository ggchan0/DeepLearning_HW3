from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.nn as nn
import torch.optim as optim

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"
EARLY_STOP = 10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(args):
    from os import path
    model = CNNClassifier()
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loader = load_data(TRAIN_PATH, num_workers=4, batch_size=200)

    """
    mean = 0.
    std = 0.
    total_num_batches = 0
    means = []
    stds = []
    for data, _ in data_loader:
        total_num_batches += 1
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        #print(data.shape)
        print(data.mean(2))
        print(data.std(2))
        exit()
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= total_num_batches
    std /= total_num_batches
    print(mean, std)
    exit()
    """
    validation_accuracy = 0

    for epoch in range(15):
        model.train()
        running_loss = 0.0

        for index, (inputs, label) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs)
            print(label)
            loss = loss_function(outputs, label)
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
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
