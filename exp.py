import nnfs
from nnfs.datasets import spiral_data, vertical_data, sine_data
nnfs.init()
class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        print(self.layers)
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss

    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in range(1, epochs+1):
            output = self.forward(X)
            print(output)
            exit()

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in Self.layers:
            layer.forward(layer.prev.output)
        return layer.output


X, y = sine_data()
model = Model()
model.add(Layers_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layers_Dense(64, 3))
model.add(Activation_Softmax())

model.set(loss=Loss_CategorialCrossrntropy(),
          optimizer=Optimizer_SGD(decay=1e-3, momentum=0.9))
model.finalize()
model.train(X, y, epochs=10000, print_every=100)


# X, y = spiral_data(samples=100, classes=3)

# dense1 = Layers_Dense(2, 64)
# activation1 = Activation_ReLU()
# dense2 = Layers_Dense(64, 3)
# loss_activation = Activation_Softmax_Loss_CategorialCrossrntropy()
# optimizer = Optimizer_SGD(decay = 1e-3, momentum=0.9)

# print(type(X), X.shape ,type(y), y.shape)
# for epoch in range(10001):

#     dense1.forward(X)
#     activation1.forward(dense1.output)

#     dense2.forward(activation1.output)
#     loss = loss_activation.forward(dense2.output, y)
#     # print(loss_activation.output[:5])
#     # loss = loss_function.calculate(activation2.output, y)
#     # print('loss:',loss)

#     predictions = np.argmax(loss_activation.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions == y)
#     # print("acc:", accuracy)

#     if not epoch % 100:
#         print(f'epoch:{epoch}, ' +
#               f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, '+
#               f'lr: {optimizer.current_learning_rate}')

#     loss_activation.backward(loss_activation.output, y)
#     dense2.backward(loss_activation.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)

#     optimizer.pre_update_params()
#     optimizer.update_params(dense1)
#     # optimizer.update_params(activation1)
#     optimizer.update_params(dense2)
#     optimizer.post_update_params()
