from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch
import cv2
from collections import deque
import os
import numpy as np
from matplotlib import pyplot as plt


class HealthReader(nn.Module):
    def __init__(self):
        super(HealthReader, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.maxpool1 = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(675, 256)
        self.lin2 = nn.Linear(256, 2)

    def forward(self, input):
        x = self.maxpool1(F.relu(self.conv1(input)))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        return x


class ElimDetector(nn.Module):
    def __init__(self):
        super(ElimDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.maxpool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.maxpool2 = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(64, 2)

    def forward(self, input):
        x = self.maxpool1(F.relu(self.conv1(input)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class HPImages(Dataset):
        def __init__(self):
            self.file = open('HP/labels.txt', 'r')
            self.filename = 'HP/hp'
            self.labels = self.file.read().split('\n')
            # print(self.labels)
            # print(self.labels.index(''))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, item):
            img = cv2.imread(self.filename + str(item) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(1, img.shape[0], img.shape[1])
            img = np.add(img, np.floor(np.random.random(size=img.shape) * 100.0), casting="unsafe")
            return torch.tensor([int(self.labels[item])], dtype=torch.long, device=device), \
                   torch.tensor(img, dtype=torch.float32, device=device)

    class ElimImages(Dataset):
        def __init__(self):
            self.file = open('Elim/labels.txt', 'r')
            self.filename = 'Elim/elim'
            self.labels = self.file.read().split('\n')
            # print(self.labels)
            # print(self.labels.index(''))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, item):
            img = cv2.imread(self.filename + str(item) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
            img = np.add(img, np.floor(np.random.random(size=img.shape) * 100.0), casting="unsafe")
            return torch.tensor([int(self.labels[item])], dtype=torch.long, device=device), \
                   torch.tensor(img, dtype=torch.float32, device=device)

    hp_dataset = HPImages()
    hp_dataloader = DataLoader(hp_dataset, batch_size=4, shuffle=True)
    elim_dataset = ElimImages()
    elim_dataloader = DataLoader(elim_dataset, batch_size=4, shuffle=True)

    def train_hp():
        hp = HealthReader().to(device)

        try:
            hp.load_state_dict(torch.load('HPReader.pt'))
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass

        hp_optim = torch.optim.Adam(hp.parameters(), lr=0.0001)

        epoch_loss = []
        losses = []
        criterion = nn.CrossEntropyLoss()
        loss_print = deque(maxlen=1)

        for i in range(1, 21):
            correct = 0
            if i % 10 == 0:
                epoch_loss.append(np.mean(losses))
                losses = []
            for labels, samples in hp_dataloader:
                x = hp(samples)
                hp_optim.zero_grad()
                loss = criterion(x, labels.squeeze(1))
                loss.backward()
                hp_optim.step()
                losses.append(loss.item())
                loss_print.append(loss.item())
                _, predicted = torch.max(x.data, 1)
                correct += (predicted == labels.squeeze(1)).sum().item()
            print('\r' + "Episode: {} Loss: {}, Acc: {}".format(i, np.mean(loss_print), 100 * correct / len(hp_dataset)),
                  end="")

        torch.save(hp.state_dict(), 'HPReader.pt')

        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.show()

    def train_elim(epochs=50):
        elim = ElimDetector().to(device)

        try:
            # elim.load_state_dict(torch.load('ElimDetector.pt'))
            pass
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass

        elim_optim = torch.optim.Adam(elim.parameters(), lr=0.0001)

        epoch_loss = []
        losses = []
        criterion = nn.CrossEntropyLoss()
        loss_print = deque(maxlen=1)
        acc_plot = []
        acc = []

        for i in range(1, epochs+1):
            correct = 0
            if i % 10 == 0:
                epoch_loss.append(np.mean(losses))
                acc_plot.append(np.mean(acc))
                losses = []
                acc = []

            for labels, samples in elim_dataloader:
                x = elim(samples)
                elim_optim.zero_grad()
                loss = criterion(x, labels.squeeze(1))
                loss.backward()
                elim_optim.step()
                losses.append(loss.item())
                loss_print.append(loss.item())
                _, predicted = torch.max(x.data, 1)
                correct += (predicted == labels.squeeze(1)).sum().item()
            print(
                '\r' + "Episode: {} Loss: {:.5f}, Acc: {:.2f}".format(i, np.mean(loss_print), 100 * correct / len(elim_dataset)),
                end="")
            acc.append(correct / len(elim_dataset))

        torch.save(elim.state_dict(), 'ElimDetector.pt')

        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.plot(range(len(acc_plot)), acc_plot)
        plt.show()

    train_elim()
