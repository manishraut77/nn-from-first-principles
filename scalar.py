import math
import random


class Scalar:

    def __init__(self,value,_children=(),op='',label=''):

        self.value=value
        self.grad=0.0
        self._children=set(_children)
        self._op=op
        self.label=label
        self._backward=lambda:None

    def __repr__(self):
        return f"Scalar: {self.value}"

    def __add__(self,other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out= Scalar(self.value+other.value,(self,other),'+',"+")
        def _backward():
             self.grad+=1.0*out.grad
             other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value * other.value, (self, other), '*', '*')
        def _backward():
             self.grad+=other.value*out.grad
             other.grad+=self.value*out.grad
        out._backward=_backward         
        return out

    def __rmul__(self,other):
        return self*other

    def __pow__(self,other):
        assert isinstance(other, (int, float))
        out = Scalar(self.value**other,(self,),f'**{other}')
        def _backward():
             self.grad+=other*self.value**(other-1)*out.grad
        out._backward=_backward
        return out
    
    def __truediv__(self,other):     #self/other
             return self * other**-1

    def tanh(self):
        t = math.tanh(self.value)
        out=Scalar(t,(self,),'tanh','tanh')
        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out

    def relu(self):
        out=Scalar(max(0.0,self.value),(self,),"relu","relu")
        def _backward():
            if self.value >0:
              self.grad += 1.0*out.grad
            else:
                self.grad+=0
        out._backward=_backward
        return out
    
    
    def exp(self):
        out=Scalar(math.exp(self.value),(self,),'exp','exp')
        def _backward():
            self.grad+=out.value * out.grad
        out._backward=_backward
        return out

    def log(self):
        out=Scalar(math.log(self.value),(self,),'log','log')
        def _backward():
            self.grad+=(1/self.value)*out.grad
        out._backward=_backward
        return out

    def __neg__(self):
        return self*-1

    def __sub__(self, other):
      other = other if isinstance(other, Scalar) else Scalar(other)
      return self + (-other)
    

    def backward(self):

        topo=[]
        visited=set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
            
# what else should it do?? umm ..............

