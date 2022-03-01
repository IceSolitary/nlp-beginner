import numpy as np
import copy


class model():

    def __init__(self, dim, batch_size, data_length):
        # w (dim,features) b (1,features)
        self.w = np.ones((dim, 5))
        self.b = np.ones((1, 5))
        self.bath_size = batch_size
        self.data_length = data_length
        self.features = 5

    def softmax(self, z):
        shift_z = z - np.max(z)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def to_one_hot(self, y, features_num=5):
        return np.eye(features_num)[y]

    def propagate(self, x, y):
        # x (batchsize , dim) y(batchsize, features)
        logits = x.dot(self.w) + self.b
        y_pred = self.softmax(logits)

        cost = -np.sum(np.sum(np.multiply(y, np.log(y_pred)), axis=1)/self.features, axis=0)/self.bath_size
        dw = (x.T.dot((logits - y))) / self.bath_size
        db = np.mean(logits - y, axis=0)


        grad = {
            'dw': dw,
            'db': db
        }

        return cost, grad

    def optimize(self, x, y, num_iterations=10000, learning_rate=0.1, print_cost=False):
        # y (batch_size,features) x (batch_size,dim)
        # y = x * w  + b
        costs = []

        for i in range(num_iterations):
            cost, grads = self.propagate(x, y)
            dw = grads['dw']
            db = grads['db']

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)

        cost = np.array(costs).mean()
        return cost

    def eval(self, x, y):

        logits = x.dot(self.w) + self.b
        pred = np.argmax(self.softmax(logits), axis=1)
        y = np.argmax(y,axis=1)
        accuracy = np.mean(pred == y)

        return accuracy




