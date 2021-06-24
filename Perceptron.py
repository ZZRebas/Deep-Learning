#单层感知器
#参考：https://www.jianshu.com/p/d7189cbd0983?from=groupmessage
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X=np.array([[1,4,3],
            [1,5,4],
            [1,4,5],
            [1,1,1],
            [1,2,1],
            [1,3,2]])
Y=np.array([1,1,1,-1,-1,-1])
W=(np.random.random(3)-0.5)*2   #设定权值向量(w0,w1,w2),权值范围为-1,1
lr=0.3  #学习率
n=0     #迭代次数
Out=0     #神经网络输出

def update():
    global X,Y,W,lr,n
    n+=1
    Out=np.sign(np.dot(X,W.T))
    W_Tmp=lr*(np.dot((Y-Out),X))
    W=W+W_Tmp

if __name__ == '__main__':
    for index in range(100):
        update()
        Out=np.sign(np.dot(X,W.T))
        print('Out:'.ljust(5),Out)
        print('Y:'.ljust(5),Y)
        if (Out==Y).all():
            print('Finished')
            print('epoch:',n)
            break

x1=[3,4]
y1=[3,3]
x2=[1]
y2=[1]

k=-W[1]/W[2]
b=-W[0]/W[2]
print('k=',k)
print('b=',b)
x=np.linspace(0,5)
plt.figure()
plt.plot(x,k*x+b,'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()
