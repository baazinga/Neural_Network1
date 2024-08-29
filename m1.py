#read on eveything into a massive string & split it into a list of name strings
words = open('names.txt','r').read().splitlines() 
#print(words[:10])
#print(len(words))
# print(min(len(w) for w in words))
# print(max(len(w) for w in words))

b = {}
for w in words:
    #adding an 'S' and an 'E' at the begnning & the bottom to print out the the first & last character twice
    chs = ['<S>'] + list(w) + ['<E>'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        #print(ch1,ch2) #  eg. 'emma' --> '<S>' e , e m, m m, m a, a '<E>'

''' How to predict the next character isbasically COUNT the times that a character appears after another'''
#print(b) #  ('a', '<E>'): 3 --> the name ends with an 'a' happened 3 times


#listed from the most likely to the least (times of the two character appears)
sorted(b.items(), key = lambda kv : -kv[1]) 


'''store the information in a 2D array'''
import torch
#create a tensor:
N = torch.zeros((27,27), dtype=torch.int32)
#create a look-up table: 
#list of the lower key words from 'a' to 'z' (26)
chars = sorted(list(set(''.join(words)))) 
#stoi: mapping of string(character) to integers   --->eg.'a':0  'b':1   'c':2  ...  'z':25
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
#itos: reverse the mapping integers to string(character)
itos = {i:s for s,i in stoi.items()} #read all the items & reverse(i:s)

#the 2D array:
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1] #interpret the characters into integers
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 #count the times that two characters are aside --->eg.'a' follow 'a' 556 times



import matplotlib.pyplot as plt
import numpy as np  # If you need to create an image or manipulate arrays
# '%matplotlib inline' ---> magic command is specific to IPython environments, such as Jupyter notebooks,
#plt.imshow(N) ---> a better way to do so:
plt.figure(figsize=(16,16))
plt.imshow(N,cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.show() #makesure the figure could be shown

'''a simple bigram model'''
#follow the probabilities ---> sampling from the row, sample the Nth token '.'
'''
#extract the first row & int --> float
p = N[0].float()
#normalization: get the probability distribution
p = p/p.sum()
#Generator: (fxed)
g = torch.Generator().manual_seed(2147483647)
#ix: index
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
'''

#Initialize the Network
#a fixed generator
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27),generator=g,requires_grad=True)

P = (N+1).float() # model smoothing (avoid  -infin such outputs)
#normalize by *rows*:  
P /= P.sum(1, keepdim=True) #(27,1) --- copy the sums to strectch --->  (27,27)


for i in range(50):
    #create a list
    out = []
    ix = 0
    while True:
        p = P[ix]
        #p = N[ix].float() #extract & convert 
        #p = p/p.sum()
        # p = torch.ones(27) / 27.0 
        # ---> an untrained model, TITS each character is eauqlly likely to be appeared in the output
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix]) #output added to the list
        if ix==0:
            break
    print(''.join(out))

'''evaluate the quality of the model ---> training loss ---> (log)likelihood'''
log_likelihood = 0.0
n = 0

#log(a*b*c) = log(a) + log(b) + log(c)
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob) #use the log function in pytorch
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood #nll: negative log likelihood -- we want the loss to present the lower the better
print(f'{nll=}')
print(f'{nll/n}')

'''Training: find the parameters that minimize the nll '''
#create the training set of bigrams(x,y)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        #print(ch1,ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

import torch.nn.functional as F
 
#gradient descent:
for k in range (100):
    #forward pass
    #create a 1x27 tensor that only include 0,1 ---> 1 highlights where the character is
    #xenc : xencoded --> encode integers into vectors 
    xenc = F.one_hot(xs, num_classes=27).float() 
    #we know how to backpropagate the operations below:
    # + & *
    logits = xenc @ W # log-counts   # btw : @ is matrix multipilication mark in PyTorch 
    # expo
    counts = logits.exp() #counts, equivalent to  N  
    probs = counts/ counts.sum(1,keepdims=True) #probabilities for next character
    # the last two lines ---> 大名鼎鼎的'softmax'
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean() #regularization : 
    print(loss.item())

    #backward pass
    W.grad = None #set to zero the gradient
    loss.backward() 

    #update the weights
    W.data += -50 * W.grad

'''
#an example of the model
nlls = torch.zeros(5)
for i in range(5):
    #i-th bigram
    x = xs[i].item() #input character index
    y = ys[i].item() #label character index
    print('---------------')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes{x},{y})')
    print('input to the neural net:', x)
    print('output probabilities from the neural net:',probs[i])
    print('label (actual next character):', y)
    p = probs[i, y]
    print('probability assigned by the net to the correct character:', p.item())
    logp = torch.log(p)
    print('log likeihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print('=============')
print('average negative log likelihood, i.e. loss = ', nlls.mean().item())

'''

