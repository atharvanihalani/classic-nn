import numpy as np
import tensorflow as tf

class DenseLayer():
    '''
    init layer with input / output size
    '''
    def __init__(self, dims: tuple, is_final=False) -> None:
        self.weights = np.random.normal(0, np.sqrt(1/dims[0]), dims)
        self.biases = np.random.normal(0, np.sqrt(1/dims[0]), dims[1])
        self.activation = self.softmax if is_final else self.leaky_relu

    def call(self, inputs):
        self.inputs = inputs
        out = np.matmul(inputs, self.weights)
        out = np.add(out, self.biases)
        out = self.activation(out)

        return out
    
    def update_grads(self, downstream, nabla):
        '''
        updates the weights and biases
        downstream: dL / d(layer_output); the self.___ variables are d(layer_output) / d(___)
        nabla: learning rate
        returns dL / d(layer_input)
        '''
        gradients = []
        for i in range(len(downstream)):
            gradients.append(np.matmul(downstream[i], self.activation_jacobian[i]))
        gradients = np.array(gradients) 

        self.biases -= nabla * np.mean(gradients, 0)

        weight_grads = []
        for i in range(len(gradients)):
            weight_grads.append([[input * dw for dw in gradients[i]] for input in self.inputs[i]])
        self.weights -= nabla * np.mean(weight_grads, 0)

        return gradients @ self.weights.T 

    def leaky_relu(self, input):
        out = np.array([[val if val > 0 else 0.1*val for val in row] for row in input])

        self.activation_jacobian = []
        for row in out:
            zeros = np.zeros((len(row), len(row)))
            np.fill_diagonal(zeros, [1 if i > 0 else 0.1 for i in row])
            self.activation_jacobian.append(zeros)
        self.activation_jacobian = np.array(self.activation_jacobian)
        return out
    
    def softmax(self, input):
        out = np.exp(input)
        denom = np.reshape(np.sum(out, 1), (-1, 1))
        out = out / denom

        self.activation_jacobian = []
        for row in out:
            diag = np.zeros((len(row), len(row)), dtype=np.float64)
            np.fill_diagonal(diag, row)
            temp = np.array([[-row[i]*row[j] for j in range(len(row))] for i in range(len(row))]) + diag
            self.activation_jacobian.append(temp)
        self.activation_jacobian = np.array(self.activation_jacobian)
        return out
    

class LossLayer():
    '''
    RMS or CE w/ logits
    '''
    def __init__(self, CE=True) -> None:
        self.loss = self.ce_logits if CE else self.mse
    
    def call(self, inputs, fax):
        return self.loss(inputs, fax)
    
    def ce_logits(self, inputs, fax):
        out_probmax = [inputs[i][fax[i]] for i in range(len(inputs))]
        out = -1 * np.mean(np.log(out_probmax))
        
        self.input_grads = np.zeros_like(inputs)
        for i in range(len(inputs)):
            self.input_grads[i][fax[i]] = -1 / out_probmax[i]

        return out

    def rms(self, inputs, fax):
        oh_fax = np.zeros((len(fax), 10))
        oh_fax[np.arange(fax.size), fax] = 1

        out_points = np.sqrt(np.mean(np.square(inputs - oh_fax), 1, keepdims=True))
        out = np.mean(out_points) # takes the average loss over the batch

        self.input_grads = (inputs - oh_fax) / (inputs.shape[0] * inputs.shape[1] * out_points) 
        return out
    
    def mse(self, inputs, fax):
        oh_fax = np.zeros((len(fax), 10))
        oh_fax[np.arange(fax.size), fax] = 1

        out = np.mean(np.square(inputs - oh_fax))

        self.input_grads = (inputs - oh_fax) / np.prod(inputs.shape)
        return out
    
    def get_input_grads(self):
        return self.input_grads

class Model():
    def __init__(self) -> None:
        self.layers = [
            DenseLayer((28*28, 64)),
            DenseLayer((64, 10), True),
        ]
        self.loss_layer = LossLayer(True)
    
    def train(self, inputs, fax):
        for i in range(30):
            next = inputs[i*2000 : (i+1)*2000]
            bfax = fax[i*2000 : (i+1)*2000]
            
            for layer in self.layers:
                next = layer.call(next)
            loss = self.loss_layer.call(next, bfax)

            preds = np.argmax(next, 1) 
            accuracy = np.count_nonzero(preds == bfax)

            current_grads = self.loss_layer.get_input_grads()
            for j in range(len(self.layers)-1, -1, -1):
                current_grads = self.layers[j].update_grads(current_grads, 0.2)
                
            # calc / update gradients
            print(f'batch: {i+1}/30 \n loss: {loss} \n accuracy = {accuracy}/{len(inputs) / 30}')
    
    def test(self, inputs, fax):
        next = inputs
        
        for layer in self.layers:
            next = layer.call(next)
        loss = self.loss_layer.call(next, fax)
        
        preds = np.argmax(next, 1)
        accuracy = np.count_nonzero(preds == fax)

        print(f'\n\ntesting metrics:\n loss: {loss} \n accuracy = {accuracy}/{len(inputs)}')

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28*28))
    x_train = x_train / 256
    x_test = np.reshape(x_test, (10000, 28*28))
    x_test = x_test / 256
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train_small = x_train[:1000]
    y_train_small = y_train[:1000]

    model = Model()
    model.train(x_train, y_train)
    model.test(x_test, y_test)

main()