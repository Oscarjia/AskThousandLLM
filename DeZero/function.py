from variable import Variable
import numpy as np

class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs=inputs #保存输入的变量
        self.outputs=outputs

        return outputs if len(outputs)>1 else outputs[0]
        
    def forward(self,xs):
        raise NotImplementedError()

    def backward(self,gys):
        raise NotImplementedError()

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