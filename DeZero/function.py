from variable import Variable
import numpy as np
import weakref
from  config import Config


class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            # 设置辈分
            self.generation=max([x.generation for x in inputs])
            # 设置连接
            for output in outputs:
                 output.set_creator(self)

        self.inputs=inputs #保存输入的变量
        self.outputs=[weakref.ref(output) for output in outputs ]

        return outputs if len(outputs)>1 else outputs[0]
        
    def forward(self,xs):
        raise NotImplementedError()

    def backward(self,gys):
        raise NotImplementedError()



class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    def backward(self, gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx
    

class Exp(Function):
    def forward(self, x):
        y=np.exp(x)
        return y
    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx
    
def square(x):
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)


    

def numberical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/2*eps

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x