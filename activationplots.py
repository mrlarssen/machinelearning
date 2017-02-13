import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
sig = sigmoid(x)
t = np.tanh(x)
relu = np.maximum(x,0)


plt.plot(x,sig, label=r'$f(x)=\frac{1}{1+e^{-x}}$') 
plt.grid(True)
plt.legend(loc='upper left', frameon=False, prop={'size':18})
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,-0.5,1.5))

plt.show()
#plt.plot(x,t, label=r'$f(x)=\frac{1}{1+e^{-x}}$')
