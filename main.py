import numpy as np

class DenseLayer():
    '''
    init layer with input / output size
    layer should be able to 
    '''
    def __init__(self, dims: tuple, is_final=False) -> None:
        self.weights = np.random.normal(0, 1, dims)
        self.biases = np.random.normal(0, 1, dims[1])
        self.activation = self.softmax if is_final else self.leaky_relu

    def call(self, input):
        out = np.matmul(input, self.weights)
        out = np.add(out, self.biases)
        out = self.activation(out)
        return out

    def leaky_relu(self, input):
        out = [val if val > 0 else 0.1*val for val in input]
        return out
    
    def softmax(self, input):
        out = np.exp(input)
        denom = np.sum(out)
        out = out / denom
        return out

class Model():

    def __init__(self) -> None:
        self.layers = [
            DenseLayer((28*28, 128)),
            DenseLayer((128, 32)),
            DenseLayer((32, 10), True)
        ]
    
    def call(self, inputs):
        '''inputs: batch size X 28*28'''
        
        pass