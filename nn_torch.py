import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

class neural_network(nn.Module):
    def __init__(self, dimensions, random_weights=True):
        super().__init__()
        self.fc1 = nn.Linear(dimensions[0], dimensions[1])
        self.fc2 = nn.Linear(dimensions[1], dimensions[-1])

        if not random_weights:
            self.fc1.weight.data = torch.full((dimensions[1], dimensions[0]), 0.0)
            self.fc2.weight.data = torch.full((dimensions[-1], dimensions[1]), 0.0)
        else:
            self.fc1.weight.data = torch.normal(0, 0.1, (dimensions[1], dimensions[0]))
            self.fc2.weight.data = torch.normal(0, 0.1, (dimensions[-1], dimensions[1]))

    def forward(self, x):
        z_prime = self.fc1(x)
        a = torch.sigmoid(z_prime)
        z = self.fc2(a)
        hat_y = nn.functional.log_softmax(z, dim=1)

        return hat_y, torch.argmax(hat_y)

if __name__ == '__main__':
    train = datasets.MNIST('', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('', train=False, download=True, transform=transforms.ToTensor())

    train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=1)

    net = neural_network([784, 300, 10], random_weights=True)
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(10):
        print('Epoch -', epoch+1)
        count = 0
        for data in train:
            X, y = data
            net.zero_grad()
            hat_y, _ = net(X.view(-1, (28*28)))
            loss = loss_function(hat_y, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            count += 1
        print('-------------------')
    
    print('\n\nTesting Started!')
    correct = 0
    for data in test:
        X, y = data
        hat_y, pred = net(X.view(-1, 784))
        if pred == y: correct += 1
    print('Accuracy -', correct/100, '%')
    print('Error -', (10000 - correct)/100, '%')
    
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Learning Curve for Learning Rate = 0.05, Batch Size = 32, Epochs = 10')
    plt.show()