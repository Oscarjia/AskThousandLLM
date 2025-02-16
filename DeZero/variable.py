import numpy as np

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f'{type(data)} is not the supported data format ')
        self.data=data
        self.grad=None
        self.creator=None # link with function

    def set_creator(self,func):
        self.creator=func
    
    def cleargrad(self):
        self.grad=None

    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    funcs.append(x.creator)
            # x,y=f.input,f.output
            # x.grad=f.backward(y.grad)
            # if x.creator is not None:
            #     funcs.append(x.creator)
        