import numpy as np
import tensorflow as tf

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
        out = np.array(self.activation(out))
        return out

    def leaky_relu(self, input):
        out = [[val if val > 0 else 0.1*val for val in row] for row in input]
        return out
    
    def softmax(self, input):
        out = np.exp(input)
        denom = np.reshape(np.sum(out, 1), (-1, 1))
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
        next = inputs
        for layer in self.layers:
            next = layer.call(next)

        pass
        preds = np.argmax(next, 1)
        return preds

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28*28))
    x_train = x_train / 256
    x_test = np.reshape(x_test, (10000, 28*28))
    x_test = x_test / 256


    model = Model()
    test_out = model.call(x_train)
    pass
    

main()