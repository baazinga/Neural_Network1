import math
import numpy as np
import matplotlib.pyplot as plt
# Visualize the expression code
from graphviz import Digraph
import random

class Value:
    #Default initialize
    def __init__(self,data, _children=(), _op='', label=''):
        self.data = data
        #changing this variable is not going to influence the loss function
        self.grad = 0.0 
        # By default, lambda is a function that does nothing
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op #maintain
        self.label= label
    
    #wrapper: providing a way to print the results looking micier in Python
    def __repr__(self):
       return f"Value(data={self.data})" 
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other),'+')

        def _backward():
            # Derivation of the operation: when it comes to addation, the deravitive is only the distribution of the former result (*1.0)
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
    
    def __neg__(self): #-self
        return self*-1
    
    def __sub__(self,other): #self - other
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other),'*')

        def _backward():
            '''with multi variables that might exist in more than one monomial(单项式),
            so we need to accumulate the gradients (add them)'''
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad #got this right!!! BUT why don't we need '.data'  ?
        out._backward = _backward   
    
    def __rmul__(self, other): #other * self :if it can't do other*self then it will check if it can do the other way round.
                                #---> what if it is dot product or are we just assuming it is only single number doing multiplicationi?
        return self*other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self*(other**-1)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
         x = self.data
         out = Value(math.exp(x), (self, ), 'exp')

         def _backward():
             self.grad += out.data*out.grad
         out._backward = _backward   

         return out 

    def backward(self):
        #make a topological graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()


def trace(root):
    #builds a set of all nodes and edges of a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'}) # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        #for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" %(n.label, n.data, n.grad), shape='record')
        if n._op:
            # if thid value is a result of some operation, create an op node fot it
            dot.node(name = uid + n._op, label = n._op)
            #and connect thus node to it
            dot. edge(uid + n._op, uid)

    for n1, n2 in edges:
        #connect n1 to n2 in edges
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    #save the graph into a file
    dot.render('graph10', format='png', view=True)
        
    return dot

#using Pytorch API --> BP
# tensors should be single elements
import torch
"""Tensors are n dimensional arrays of scalars """
#inputs
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
#algorithm
n = w1*x1 + w2*x2 + b
#tanh
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2',x2.grad.item())
print('w2',w2.grad.item())
print('x1',x1.grad.item())
print('w1',w1.grad.item())

#Building the MLP now neural network

class Neuron:

    def __init__(self, nin):
        #nin : number of inputs to  the neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range (nin)] # weight choose a number randomly from -1 to 1 
        self.b = Value(random.uniform(-1, 1))   #bias : control the overall trigger happiness of this neuron

    def __call__(self, x):
        #w * x + b (dot product)
        #n(x) means initiate call function
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) #sum()has two inouts and the 2nd one is 0.0 by default 
        out = act.tanh() 
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
"""
zip() function is used to aggregate elements from multiple iterables (such as lists, tuples, strings, etc.) 
by pairing corresponding elements together into tuples, and then returning an iterator over these tuples. 
Essentially, zip() can pair elements from multiple sequences.
"""
        
class Layer:
    
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len (outs) == 1 else outs #return outs if it is exactly a single element else return the full list 
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
       
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)

#draw_dot(n(x))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] #desired targets --->ground truth

for k in range (20):
    #forward pass
    ypred = [n(x) for x in xs] #---> prediction
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    #unsupported operand type(s) for +: 'int' and 'NoneType'
    '''
    we want the loss to be small, indicating that the network is perforing well
    -> the prediction is close to the desired results, aka ground truth
    '''

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)