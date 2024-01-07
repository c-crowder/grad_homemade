import numpy as np
import math
import random


class Value():
    def __init__(self, data: int|float, children: tuple = (), operation='', label=''):
        self.data = data
        self._backward = lambda: None
        self.grad = 0
        self._children = set(children)
        self._operation = operation
        self._label = label

    def __str__(self):
        return (f"Value object: self.data={self.data}")

    def __rpr__(self):
        return (f"Value object: self.data={self.data}")

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        x = Value(self.data + other.data, (self, other), operation='+')

        def _backward():
            self.grad += 1.0 * x.grad
            other.grad += 1.0 * x.grad
        x._backward = _backward

        return x

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other    

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        x = Value(self.data * other.data, (self, other), operation='*')

        def _backward():
            self.grad += other.data * x.grad
            other.grad += self.data * x.grad
        x._backward = _backward

        return x

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        x = Value(self.data ** other.data, (self, other), operation='**')

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * x.grad
        x._backward = _backward

        return x

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        self.grad = 1
        topo = []
        visited = set()

        def unroll(visit):
            if visit not in visited:
                visited.add(visit)
                for child in visit._children:
                    unroll(child)
                topo.append(visit)

        unroll(self)
        for node in reversed(topo):
            node._backward()

class Neuron():
    def __init__(self, number_inputs):
        self.ws = [Value(random.uniform(-1, 1)) for _ in range(number_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, xs):
        forward = sum([wi * xi for wi, xi in zip(self.ws, xs)], self.b)
        activation = forward.tanh()
        return activation

    def parameters(self):
        return self.ws + [self.b]


class Layer():
    def __init__(self, number_inputs, number_neurons):
        self.neurons = [Neuron(number_inputs) for _ in range(number_neurons)]

    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class Network():
    def __init__(self, neurons_in, neurons_out):
        sz = [neurons_in] + neurons_out
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(neurons_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, xs, y_true, epochs=20, lr=.01, verbose=False):
        for e in range(epochs):
            self.zero_grad()
            losses = self.update_params(xs, y_true, lr)

            if verbose:
                self.print_epoch(e, losses)

            for p in self.parameters():
                p.data += -lr * p.grad

    def print_epoch(self, e, losses):
        print(f"""
Epoch: {e}
Loss: {losses.data}""")

    def update_params(self, xs, y_true, lr):
        y_pred = [self(x) for x in xs]

        losses = self.loss(y_true, y_pred)
        losses.backward()
        for p in self.parameters():
            p.data += -lr * p.grad

        return losses

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


    def loss(self, y_true, y_pred, method='mse'):
        mapping = {'mse': self.mse}
        try:
            loss = mapping[method]
        except KeyError:
            print('loss function not valid, using mse instead')
            loss = self.mse
        return loss(y_true, y_pred)


    def mse(self, y_true, y_pred):
        loss = sum((y_out - y_begin)**2 for y_begin, y_out in zip(y_true, y_pred))
        return loss


if __name__ == "__main__":
    a = Value(3)
    b = Value(4)
    c = a*b
    print(1+a)
    print(a.grad, b.grad, c.grad)
    c.backward()
    print(a.grad, b.grad, c.grad)
    xs = [
        [2, 3, -1],
        [3, -1, .5],
        [.5, 1, 1],
        [1, 1, -1]
    ]

    ys = [1, -1, -1, 1]

    n = Network(3, [2, 3, 1])
    n.train(xs, ys, epochs=400, lr=.01, verbose=True)
    [print(n(x)) for x in xs]
