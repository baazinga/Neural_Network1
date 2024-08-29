import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
'''
def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
#plt.show() #print the graoh

#h = 0.0000001
#x = -3.0

#print((f(x+h)-f(x))/h)

a = 2.0
b = -3.0
c = 10.0
h = 0.0001
  
d = a*b + c

d1 = a*b + c
c+= h
d2 = a*b + c

#print('d1',d1)
#print('d2',d2)
#print('slope',(d2-d1)/h)
'''

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
        out = Value(self.data + other.data, (self, other),'+')

        def _backward():
            # Derivation of the operation: when it comes to addation, the deravitive is only the distribution of the former result (*1.0)
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other),'*')

        def _backward():
            '''with multi variables that might exist in more than one monomial(单项式),
            so we need to accumulate the gradients (add them)'''
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
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
    
a = Value(2.0, label = 'a')
b = Value(-3.0, label = 'b')
c = Value(10.0, label = 'c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label = 'f')
L = d*f; L.label = 'L'
#print(a*b)

# Visualize the expression code
from graphviz import Digraph

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
    dot.render('graph9', format='png', view=True)
        
    return dot
"""
L.grad = 1.0
f.grad = 4.0
d.grad = -2.0
c.grad = -2.0
e.grad = -2.0
a.grad = 6.0
b.grad = -4.0
"""

"""
#nudge the input tp try make L go up
a.data += 0.01*a.data 
b.data += 0.01*b.data 
c.data += 0.01*c.data 
f.data += 0.01*f.data 

e = a*b
d = e+c
L = d*f

print(L.data)
 """
# we expect to see L to go up 

#draw_dot(L)
'''
What we have sone so far:
* build out mathamatical expressions using only + ann *
* scalar value along the way
* forward pass 
* multi-input

--> back propogation: 
* reverse the process and calculate the gradients of all the intermediate values
* calculate the derivative of the L with the repect to each input variables 
* L --> loss function ; variables --> weights (leaf notes)
* process is literally chain rule!! *
'''


""" 
--- testing the derivatives only ---
def lol():
    h = 0.0001

    a = Value(2.0, label = 'a')
    b = Value(-3.0, label = 'b')
    c = Value(10.0, label = 'c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label = 'f')
    L = d*f; L.label = 'L'
    L1 = L.data

    a = Value(2.0 + h, label = 'a')
    b = Value(-3.0, label = 'b')
    c = Value(10.0, label = 'c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label = 'f')
    L = d*f; L.label = 'L'
    L2 = L.data

    print((L2-L1)/h)

lol()
"""

# The gardient give us some insight on how to influence the final outcome.  

#plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid()

'''
These are only the examples for manual backprop...  ---> ridiculous and tolerating!!!
'''

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
o = n.tanh(); o.label = 'o'

o.backward()

draw_dot(o)
"""
o._backward()
n._backward()
b._backward()
w1x1w2x2._backward()
w1x1._backward()
w2x2._backward()
"""
 

'''

o.grad = 1.0
n.grad = 0.5

b.grad = 0.5
w1x1w2x2.grad = 0.5

w1x1.grad = 0.5
w2x2.grad = 0.5

x2.grad = 0.5
w2.grad = 0.0
x1.grad = -1.5
w1.grad = 1.0

'''



