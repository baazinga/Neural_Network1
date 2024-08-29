import math
import numpy as np
import matplotlib.pyplot as plt
# Visualize the expression code
from graphviz import Digraph

# Data Structer(how each value came to be  by word expression and from what other values)
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
        return self*(other**(-1))
    
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

#inputs x1, x2
x1 = Value(2.0, label = 'x1')
x2 = Value(0.0, label = 'x2')
#weights w1, w2
w1 = Value(-3.0, label = 'w1')
w2 = Value(1.0, label = 'w2')
#bias of the neuron
b = Value(6.8814, label = 'b')
#w1*x1 + w2*x2 + b
w1x1 = w1*x1; w1x1.label = 'w1*x1'
w2x2 = w2*x2; w2x2.label = 'w2*x2'
w1x1w2x2 = w1x1 + w2x2; w1x1w2x2.label = 'w1*x1 + w2*x2'
n = w1x1w2x2 + b; n.label = 'n'
e = (2*n).exp(); e.label = 'e' #probably here... the evpotential function didn't work...
o = (e-1)/(e+1); o.label = 'o'

o.backward()

draw_dot(o)
