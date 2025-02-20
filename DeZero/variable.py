import numpy as np

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f'{type(data)} is not the supported data format ')
        self.data=data
        self.grad=None
        self.creator=None # link with function
        self.generation=0 # parent

    def set_creator(self,func):
        self.creator=func
        self.generation=func.generation+1
    
    def cleargrad(self):
        self.grad=None

    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[]
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        add_func(self.creator)
        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad=None
    
        