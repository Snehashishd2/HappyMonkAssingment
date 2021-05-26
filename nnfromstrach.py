import numpy as np
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

EPOCHS = 10
BATCH_SIZE = 128

#Reading the train dataset
data = pd.read_csv('train.csv')

#checking unique numbers in train label column
unique = data['label'].unique()
print("Unique Numbers :", unique)

#countine the unique number of digits for classification
n_classes = len(unique)
print("Number of classes :", n_classes)

#Filtering the dataset between X and Y
x = data.drop(labels=["label"], axis=1)
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y)
# # normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.to_numpy().shape, x_test.to_numpy().shape)


class Layers_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # self.biases = np.full((1, n_neurons), 0.001)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class customActivation:
    def __init__(self, n_inputs):
        self.weights = 0.01 * np.random.randn(1, n_inputs)
        self.biases = 0.01 * np.random.randn(1, n_inputs)

    def forward(self, inputs):
        self.output = np.multiply(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dinputs = np.multiply(dvalues, self.weights)
        self.dweights = np.mean(self.dinputs, axis=0, keepdims=True)
        self.dbiases = np.mean(dvalues, axis=0, keepdims=True)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probalities = exp_values / np.sum(exp_values , axis = 1, keepdims=True)
        self.output = probalities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategorialCrossrntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis = 1)
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        lables = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(lables)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategorialCrossrntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategorialCrossrntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate= 1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
    
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iteration += 1


dense1 = Layers_Dense(784, 392)
activation1 = Activation_ReLU()
dense2 = Layers_Dense(392, 10)
loss_activation = Activation_Softmax_Loss_CategorialCrossrntropy()
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)  

steps_train = x_train.to_numpy().shape[0] // BATCH_SIZE
info = {'train_loss': [] , 'test_loss': [] , 'train_acc':[], 'test_acc': []}
dense1.forward(x_train.to_numpy())
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_train.to_numpy())
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y_train.to_numpy(), axis=1)
accuracy = np.mean(predictions == y_train.to_numpy())
info['train_loss'].append(loss)
info['train_acc'].append(accuracy)
print(f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')

dense1.forward(x_test.to_numpy())
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test.to_numpy())
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y_test.to_numpy(), axis=1)
accuracy = np.mean(predictions == y_test.to_numpy())
info['test_loss'].append(loss)
info['test_acc'].append(accuracy)
print(f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' +
        f'lr: {optimizer.current_learning_rate}')

for epoch in range(EPOCHS):
    for step in range(steps_train):
        x_batch = x_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        y_batch = y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

        dense1.forward(x_batch.to_numpy())
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y_batch.to_numpy())

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y_batch.to_numpy(), axis=1)
        accuracy = np.mean(predictions == y_batch.to_numpy())

        # if not epoch % 100:
        #     print(f'epoch:{epoch}, ' +
        #         f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' +
        #         f'lr: {optimizer.current_learning_rate}')

        loss_activation.backward(loss_activation.output, y_batch.to_numpy())
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        # optimizer.update_params(activation1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    
    dense1.forward(x_train.to_numpy())
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_train.to_numpy())
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y_train.to_numpy(), axis=1)
    accuracy = np.mean(predictions == y_train.to_numpy())
    print(f'epoch:{epoch}, ' +
          f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' +
          f'lr: {optimizer.current_learning_rate}')
    info['train_loss'].append(loss)
    info['train_acc'].append(accuracy)

    dense1.forward(x_test.to_numpy())
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test.to_numpy())
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y_test.to_numpy(), axis=1)
    accuracy = np.mean(predictions == y_test.to_numpy())
    print(f'test - epoch:{epoch}, ' +
          f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' +
          f'lr: {optimizer.current_learning_rate}')
    info['test_loss'].append(loss)
    info['test_acc'].append(accuracy)

print(info)
    

