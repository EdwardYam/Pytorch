import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


learning_rate = 1e-3
batch_size = 100
epoches = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trans_img = transforms.Compose([
        transforms.ToTensor()
    ])

trainset = MNIST("../data/", train=True, transform=trans_img, download=True)
testset = MNIST("../data/", train=False, transform=trans_img, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

for epoch in range(epoches):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')



testloss = 0.
testacc = 0.
for (img, label) in testloader:
    img = Variable(img)
    label = Variable(label)

    output = net.forward(img)
    loss = criterion(output, label)
    testloss += loss.data[0]
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.data[0]

#print testacc
testloss /= len(testset)
testacc /= len(testset)

print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))