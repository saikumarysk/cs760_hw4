import numpy as np
from torchvision import datasets, transforms

# This is only giving me ~ 88% accuracy no matter how much I change stuff. Moreover, it takes ~ 9 mins to run.
# With zero weights, it is even worse, I get same class everytime because weights_1 is so sparse. I get ~ 10% accuracy.

class neural_network:
    def __init__(self, dimensions, random_weights = True, learning_rate = 0.05, batch_size = 32):
        if random_weights:
            self.weights_1 = np.random.normal(0, 0.1, size=(dimensions[1], dimensions[0])).astype(np.float64)
            self.weights_2 = np.random.normal(0, 0.1, size=(dimensions[-1], dimensions[1])).astype(np.float64)
            self.bias_1 = np.random.normal(0, 0.1, size=(dimensions[1], 1)).astype(np.float64)
            self.bias_2 = np.random.normal(0, 0.1, size=(dimensions[-1], 1)).astype(np.float64)
            
        else:
            self.weights_1 = np.zeros((dimensions[1], dimensions[0]), dtype=np.float64)
            self.weights_2 = np.zeros((dimensions[-1], dimensions[1]), dtype=np.float64)
            self.bias_1 = np.zeros((dimensions[1], 1), dtype=np.float64)
            self.bias_2 = np.zeros((dimensions[-1], 1), dtype=np.float64)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def train(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y

        count = 0
        cum_grad_1 = np.zeros(self.weights_1.shape, dtype=np.float64)
        cum_grad_2 = np.zeros(self.weights_2.shape, dtype=np.float64)
        cum_bias_1 = np.zeros(self.bias_1.shape, dtype=np.float64)
        cum_bias_2 = np.zeros(self.bias_2.shape, dtype=np.float64)
        remainder = train_x.shape[0] % self.batch_size
        for i in range(train_x.shape[0]):
            hat_y, a = self.forward_pass(train_x[i])
            new_grad_2, new_grad_bias_2 = self.grad_2(train_y[i], hat_y, a)
            new_grad_1, new_grad_bias_1 = self.grad_1(train_y[i], hat_y, a, train_x[i])

            cum_grad_2 = cum_grad_2 + new_grad_2
            cum_grad_1 = cum_grad_1 + new_grad_1
            cum_bias_1 = cum_bias_1 + new_grad_bias_1
            cum_bias_2 = cum_bias_2 + new_grad_bias_2
            count += 1
            print(f'{i}/60000 Done!', end='\r')
            if count == self.batch_size:
                self.weights_1 = self.weights_1 - ((self.learning_rate / self.batch_size) * cum_grad_1)
                self.weights_2 = self.weights_2 - ((self.learning_rate / self.batch_size) * cum_grad_2)
                self.bias_1 = self.bias_1 - ((self.learning_rate / self.batch_size) * cum_bias_1)
                self.bias_2 = self.bias_2 - ((self.learning_rate / self.batch_size) * cum_bias_2)
                count = 0
                cum_grad_1 = np.zeros(self.weights_1.shape, dtype=np.float64)
                cum_grad_2 = np.zeros(self.weights_2.shape, dtype=np.float64)
                cum_bias_1 = np.zeros(self.bias_1.shape, dtype=np.float64)
                cum_bias_2 = np.zeros(self.bias_2.shape, dtype=np.float64)
        
        if remainder != 0 :
            self.weights_1 = self.weights_1 - ((self.learning_rate / remainder) * cum_grad_1)
            self.weights_2 = self.weights_2 - ((self.learning_rate / remainder) * cum_grad_2)

    def evaluate(self, x):
        hat_y, _= self.forward_pass(x)
        return hat_y, np.argmax(hat_y, axis=0)[0]

    def forward_pass(self, x):
        z_prime = np.matmul(self.weights_1, x) + self.bias_1
        a = self.sigmoid(z_prime) # neuron activations
        z = np.matmul(self.weights_2, a) + self.bias_2
        hat_y = self.softmax(z)

        return hat_y, a

    def grad_1(self, y, hat_y, a, x):
        return np.outer(np.matmul(self.weights_2.T, (hat_y - y)) * (a * (1 - a)), x), np.matmul(self.weights_2.T, (hat_y - y)) * (a * (1 - a))

    def grad_2(self, y, hat_y, a):
        return np.outer(hat_y - y, a), hat_y - y

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def softmax(self, z):
        denom = np.sum(np.exp(z), axis=0)
        return np.exp(z)/denom

    def cross_entropy_loss(self, pred, label):
        return -1*np.log(pred[label])

if __name__ == '__main__':
    train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    train_x = train.data.numpy()
    train_y = train.targets.numpy()
    train_x_mod = np.zeros((train_x.shape[0], 784, 1), dtype=np.float64)
    train_y_mod = np.zeros((train_y.shape[0], 10, 1), dtype=np.float64)
    for i in range(train_x.shape[0]):
        train_x_mod[i] = train_x[i].flatten().reshape((784, 1))
        target_vector = np.zeros((10, 1), dtype=np.float64)
        target_vector[train_y[i]] = 1
        train_y_mod[i] = target_vector # one-hot vector
    
    # p = np.random.permutation(train_x_mod.shape[0])
    # train_x_mod, train_y_mod = train_x_mod[p], train_y_mod[p]
    
    test_x = test.data.numpy()
    test_y = test.targets.numpy()
    test_x_mod = np.zeros((test_x.shape[0], 784, 1), dtype=np.float64)
    test_y_mod = np.zeros((test_y.shape[0], 10, 1), dtype=np.float64)
    for i in range(test_x.shape[0]):
        test_x_mod[i] = test_x[i].flatten().reshape((784, 1))
        target_vector = np.zeros((10, 1), dtype=np.float64)
        target_vector[test_y[i]] = 1
        test_y_mod[i] = target_vector
    
    dimensions = [784, 300, 10]

    classifier = neural_network(dimensions, random_weights=False)

    for epoch in range(1):
        print('Epoch -', epoch)
        print('Training Started!')
        classifier.train(train_x_mod, train_y_mod)
        print('Training Finished!')
        print('---------------------')

    print('\n\nTesting Started!')
    correct = 0
    cum_loss = 0
    for i in range(test_x_mod.shape[0]):
        y_pred, pred = classifier.evaluate(test_x_mod[i])
        y_label = test_y_mod[i]
        loss = classifier.cross_entropy_loss(y_pred, test_y[i])
        cum_loss += float(loss[0])
        correct += 1 if pred == test_y[i] else 0
    
    print('Accuracy -', round(correct/100, 2), '%')
    print('Cumulative Loss -', cum_loss)
