import numpy as np
import random


class Nureal():
    lerning_rate = 0.4

    def __init__(self, size, id=None,weight= None,baises=None):
        if weight == None :
            self.weight = np.array([np.random.rand(size[i + 1], size[i]) for i in range(len(size) - 1)],dtype=np.ndarray)
        else:
            self.weight = weight
        if  baises == None:
            self.biases = np.array([np.random.rand(j, 1) for j in size[1:]],dtype=np.ndarray)
        else:
            self.biases = baises
        # self.weight = [np.ones((size[i + 1], size[i])) for i in range(len(size) - 1)]
        # self.biases = [np.ones((j, 1)) for j in size[1:]]
        self.changed_weight = []
        self._output = []
        self._input = []
        self.point = 0
        self.id = id
        self.hidden_layers = []
        self.all_layers = []
        self.hidden_layers_error = []

    def forward(self, input):
        self._output = []
        self.hidden_layers = []
        self.all_layers = []
        self.hidden_layers_error = []
        input = [[i] for i in input]
        self._input = np.array(input, dtype=np.ndarray)
        self.all_layers = [self._input]
        input = np.array(input)
        for l in range(len(self.weight)):
            if l == len(self.weight) - 1:
                self._output = np.add(np.matmul(self.weight[l], input), self.biases[l])
                continue
            input = np.add(np.matmul(self.weight[l], input), self.biases[l])
            input = self.activation_sigmoid(input)
            self.hidden_layers.append(input)
            self.all_layers.append(input)
        self._output = self.activation_sigmoid(self._output)
        # self._output = self.activation_relue(self._output)
        self.all_layers.append(self._output)
        return self._output

    def activation_relue(self, layer):
        layer = np.array(layer, dtype=np.ndarray)
        k = layer.argmax()
        for i in range(len(layer)):
            if i == k:
                for j in range(len(layer[i])):
                    layer[i][j] = float(1.0)
            else:
                for j in range(len(layer[i])):
                    layer[i][j] = float(0.0)
        return layer

    def activation_softmax(self, layer):
        m = (np.exp(layer))
        return m / m.sum()

    def activation_sigmoid(self, layer):
        layer = layer.astype(float)
        try:
            np.exp(-layer)
        except TypeError:
            print(layer)
        return 1 / (1 + np.exp(-layer))

    @staticmethod
    def sigmoiddelta(error, output):
        ones = [[1] for i in range(len(output))]
        return np.multiply(error, np.multiply(output, np.subtract(ones, output)))

    @staticmethod
    def error_calculation(result, output):
        error = np.subtract(output, result)
        return error

    @staticmethod
    def delta(error):
        return (np.multiply(Nureal.lerning_rate, error))

    @staticmethod
    def transpose(m):
        return np.matrix.transpose(m)

    def backward(self, result):
        result = [[i] for i in result]
        error = Nureal.error_calculation(result, self._output)
        self.hidden_layers_error.append(error)
        for i in range(len(self.hidden_layers)):
            temp = np.matmul(Nureal.transpose(self.weight[-1 - i]), self.hidden_layers_error[0])
            # temp = np.matmul(Nureal.transpose(self.weight[-1 - i]),
            #                  Nureal.sigmoiddelta(self.hidden_layers_error[0],self.all_layers[-1-i]))
            self.hidden_layers_error.insert(0, temp)

        for i in range(len(self.weight)):
            temp = np.matmul(Nureal.sigmoiddelta(self.hidden_layers_error[i], self.all_layers[i + 1]),
                             Nureal.transpose(self.all_layers[i]))

            self.weight[i] = np.subtract(self.weight[i], Nureal.delta(temp))
            self.biases[i] = np.subtract(self.biases[i],
                                         Nureal.delta(
                                              Nureal.sigmoiddelta(self.hidden_layers_error[i], self.all_layers[i + 1])))

        self._output = []
        return error
a = Nureal([2,4,1])
b = [[1,1],[1,0],[0,0],[0,1]]

for j in range(600):
    for i in b:
        output = a.forward(i)
        if i == [0,0] or i == [1,1]:
            a.backward([0])
        else:
            a.backward([1])
    print(j)
for i in b:
    print(i,a.forward(i))
