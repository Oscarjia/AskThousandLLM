import numpy as np
class Variable:
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f'{type(data)} is not the supported data format ')
        self.data=data  # 持有底层 numpy 数据
        self.name=name
        self.grad=None
        self.creator=None # link with function
        self.generation=0 # parent

    def set_creator(self,func):
        # 记录生成该 Variable 的 Function，并根据 Function 的世代推算自身的世代
        self.creator=func
        self.generation=func.generation+1
    
    def cleargrad(self):
        self.grad=None

    def backward(self,retain_grad=False):
        # 初始化梯度为 1（针对标量输出）以触发反向传播
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[]
        seen_set=set()
        # 维护一个集合，确保同一个 Function 只会被压入栈一次
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        add_func(self.creator)
        # 采用拓扑顺序从后往前遍历计算图
        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            # 将本函数的梯度贡献累积到输入变量上
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                # 如果不保留中间梯度，节省内存
                for y in f.outputs:
                    y().grad=None
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'
    
    def __mul__(self,other):
        mul = _resolve_mul()
        if not isinstance(other, Variable):
            other = Variable(np.array(other))
        return mul(self,other)


def _resolve_mul():
    """Import mul lazily to avoid circular imports."""
    try:
        from .function import mul as mul_op
    except ImportError:  # pragma: no cover - fallback for direct script usage
        from function import mul as mul_op
    return mul_op
        