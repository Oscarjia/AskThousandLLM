from function import Function,numberical_diff
from variable import Variable
import numpy as np

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
def add(x0,x1):
    return Add()(x0,x1)

class Add(Function):
    def forward(self, x0,x1):
        y=x0+x1
        return y
    def backward(self, gys):
        return gys,gys
    
   


# x0=Variable(np.array(2))
# x1=Variable(np.array(3))

# ys=add(square(x0),square(x1))
# ys.backward()
# print(ys.data)
# print(x0.grad)
# print(x1.grad)
# x=Variable(np.array(0.5))
# a=square(x)
# b=exp(a)
# y=square(b)

# #反向传播计算梯度
# y.grad=np.array(1.0)
# y.backward()

# print(f"X backward {x.grad}")
# # x_num_diff=numberical_diff(C(B(A(x))),x)
# # print(f"X numberical diff{x_num_diff}")
