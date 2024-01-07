import numpy as np
import math


class Value():
    def __init__(self, data: int, children: tuple = (), operation='', label=''):
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


if __name__ == "__main__":
    from draw import draw_dot
    a = Value(3)
    b = Value(4)
    c = a*b
    print(a.grad, b.grad, c.grad)
    c.backward()
    print(a.grad, b.grad, c.grad)
    print(a.tanh())

