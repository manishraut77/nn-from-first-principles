
from scalar import Scalar
import math
import random
class Neuron:

    def __init__(self,nin,nout,nonlin=True):

        if nonlin:
            # Hidden layers: ReLU + He initialization
            std = math.sqrt(2 / nin)
            self.weights = [Scalar(random.gauss(0, std)) for _ in range(nin)]
        else:
                # Output layer: linear + Xavier initialization
                limit = math.sqrt(6 / (nin + nout))
                self.weights = [Scalar(random.uniform(-limit, limit)) for _ in range(nin) ]

        self.bias = Scalar(0.0)
        self.nonlin=nonlin

    def __call__(self,x):
        act=sum((w*x for w,x in zip(self.weights,x) ), self.bias)
        return act.tanh() if self.nonlin else act

    def parametres(self):
       return   self.weights + [self.bias]

class Layer:
    def __init__(self,nin,nouts,nonlin=True):
        self.neurons=[Neuron(nin,nouts,nonlin) for i in range(nouts)]

    def __call__(self,x):
        outs=[n(x) for n in self.neurons]
        return outs

    def parametres(self): 
            parameters = []
            for neuron in self.neurons:
               for p in neuron.parametres():
                  parameters.append(p)
            return parameters

class MLP:

    def __init__(self,nin,nouts):
        block = [nin] + nouts
        self.layers =[Layer(block[i],block[i+1],nonlin=(i != len(nouts)-1)) for i in range(len(nouts)) ]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parametres(self):
            parametres=[] 
            for layer in self.layers:
                for p in layer.parametres():
                    parametres.append(p)
            return parametres