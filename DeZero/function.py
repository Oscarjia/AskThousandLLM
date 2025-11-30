import numpy as np
import weakref

try:
    from .variable import Variable
    from .config import Config
except ImportError:  # pragma: no cover - fallback for direct script usage
    from variable import Variable
    from config import Config


class Function:

    def __call__(self, *inputs):
        inputs=[as_variable(x) for x in inputs]
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
        x=self.inputs[0].data
        gx=np.exp(x)*gy
        return gx

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        gx0 = gy * x1
        gx1 = gy * x0
        return gx0, gx1
    
def square(x):
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)

def mul(x0, x1):
    x0 = as_variable(x0)
    x1 = as_variable(x1)
    f = Mul()
    return f(x0, x1)


    

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

def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(as_array(x))

def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(as_array(obj))