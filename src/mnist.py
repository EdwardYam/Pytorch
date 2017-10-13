#coding: utf-8

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn


learning_rate = 1e-3
batch_size = 100
epoches = 10

trans_img = transforms.Compose([
        transforms.ToTensor()
    ])

trainset = MNIST("../data/", train=True, transform=trans_img, download=True)
testset = MNIST("../data/", train=False, transform=trans_img, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
'''

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''
if __name__ == "__main__":
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoches):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    testloss = 0.
    testacc = 0.
    for (img, label) in testloader:
        img = Variable(img)
        label = Variable(label)

        output = model.forward(img)
        loss = criterion(output, label)
        testloss += loss.data[0]
        predict  = output.max(1)[1]
        #_, predict = torch.max(output, 1)
        num_correct = predict.eq(label).sum()
        testacc += num_correct.data[0]

    #print testacc
    testloss /= len(testset)
    testacc /= len(testset)

    print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))