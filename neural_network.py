from torch.utils.data import TensorDataset, DataLoader
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# This is giving my ~ 95% accuracy with 10 epochs. It also runs in ~ 2 min 20 sec.
# With zero weights I get 78.78% accuracy which can be improved with more epochs.
# So, I will consider this as my from scratch implementation.

class neural_network:
    def __init__(self, dimensions, random_weights = True, learning_rate = 0.05, batch_size = 32):
        if random_weights:
            self.weights_1 = torch.normal(0, 0.1, (dimensions[1], dimensions[0]))
            self.weights_2 = torch.normal(0, 0.1, (dimensions[-1], dimensions[1]))
            self.bias_1 = torch.normal(0, 0.1, (dimensions[1], 1))
            self.bias_2 = torch.normal(0, 0.1, (dimensions[-1], 1))
            
        else:
            self.weights_1 = torch.zeros(dimensions[1], dimensions[0])
            self.weights_2 = torch.zeros(dimensions[-1], dimensions[1])
            self.bias_1 = torch.zeros(dimensions[1], 1)
            self.bias_2 = torch.zeros(dimensions[-1], 1)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def train(self, train_data):
        data_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, drop_last=False)

        count = 0
        losses = []
        for x, y in data_loader:
            x = torch.reshape(x, (self.batch_size, 784, 1))
            target_vector = torch.zeros((self.batch_size, 10, 1))
            for i in range(self.batch_size):
                target_vector[i][y[i]] = 1
            y = target_vector


            cum_grad_1 = torch.zeros(self.weights_1.shape)
            cum_grad_2 = torch.zeros(self.weights_2.shape)
            cum_bias_1 = torch.zeros(self.bias_1.shape)
            cum_bias_2 = torch.zeros(self.bias_2.shape)

            loss_ = [0]*self.batch_size

            for i in range(self.batch_size):
                hat_y, a = self.forward_pass(x[i])
                new_grad_2, new_bias_2 = self.grad_2(y[i], hat_y, a)
                new_grad_1, new_bias_1 = self.grad_1(y[i], hat_y, a, x[i])
                cum_grad_1 += new_grad_1
                cum_bias_1 += new_bias_1
                cum_grad_2 += new_grad_2
                cum_bias_2 += new_bias_2
                count += 1
                print(f'{count}/60000 Done!', end='\r')

                loss_[i] = self.cross_entropy_loss(hat_y, y[i]).item()
            
            losses.append(np.mean(loss_))
            
            self.weights_1 -= (self.learning_rate / self.batch_size) * cum_grad_1
            self.weights_2 -= (self.learning_rate / self.batch_size) * cum_grad_2
            self.bias_1 -= (self.learning_rate / self.batch_size) * cum_bias_1
            self.bias_2 -= (self.learning_rate / self.batch_size) * cum_bias_2
        
        return losses

    def evaluate(self, x):
        hat_y, _= self.forward_pass(x)
        return hat_y, torch.argmax(hat_y)

    def forward_pass(self, x):
        z_prime = torch.matmul(self.weights_1, x) + self.bias_1
        a = torch.sigmoid(z_prime) # neuron activations
        z = torch.matmul(self.weights_2, a) + self.bias_2
        hat_y = torch.softmax(z, 0)

        return hat_y, a

    def grad_1(self, y, hat_y, a, x):
        return torch.matmul(torch.matmul(self.weights_2.T, (hat_y - y)) * (a * (1 - a)), torch.transpose(x, 0, 1)), \
              torch.matmul(self.weights_2.T, (hat_y - y)) * (a * (1 - a))

    def grad_2(self, y, hat_y, a):
        return torch.matmul(hat_y - y, torch.transpose(a, 0, 1)), hat_y - y

    def cross_entropy_loss(self, hat_y, y):
        return -torch.matmul(torch.transpose(y, 0, 1), torch.log(hat_y))

if __name__ == '__main__':
    train = datasets.MNIST('', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('', train=False, download=True, transform=transforms.ToTensor())

    classifier = neural_network([784, 300, 10])

    losses = []
    print('Training Started!')
    print('-----------------')
    for epoch in range(10):
        print('Epoch -', epoch+1)
        losses_ = classifier.train(train)
        losses += losses_
        print('------------------')
    print('Training Finished!')

    print('Testing Started!')
    correct = 0
    data_loader = DataLoader(test, batch_size=1, shuffle=True)
    for x, y in data_loader:
        x = torch.reshape(x, (784, 1))
        hat_y, pred = classifier.evaluate(x)
        if pred == y: correct += 1
    
    print('Accuracy -', correct/100, '%')
    print('Error -', (10000 - correct)/100, '%')

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Learning Curve for Learning Rate = 0.05, Batch Size = 32, Epochs = 10')
    plt.show()