#用感知器训练权重，实现逻辑与
'''参考：https://github.com/hanbt/learn_dl/blob/master/perceptron.py'''

from functools import reduce

class VectorOp():
    @staticmethod
    def element_multiply(x,y):
        '''向量按元素相乘'''
        return list(map(lambda xy:xy[0]*xy[1],zip(x,y)))

    @staticmethod
    def dot(x,y):
        '''向量内积，元素相乘再求和'''
        return reduce(lambda x,y:x+y,VectorOp.element_multiply(x,y))

    @staticmethod
    def element_add(x,y):
        '''向量按元素相加'''
        return list(map(lambda xy:xy[0]+xy[1],zip(x,y)))

    @staticmethod
    def scala_multiply(v,s):
        '''向量v中的每个元素和标量s相乘'''
        return list(map(lambda i:i*s,v))

# print(VectorOp.element_multiply([1,2,3],[4,5,6]))
# print(VectorOp.element_add([1,2,3],[4,5,6]))
# print(VectorOp.dot([1,2,3],[4,5,6]))
# print(VectorOp.scala_multiply([1,2,3],3))

class Perceptron():
    def __init__(self,input_num,activator):
        self.activator=activator
        self.weights=[0.0]*input_num
        self.bias=0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n'%(self.weights,self.bias)

    def predict(self,input_vec):
        '''计算输入向量和权重的内积，再加bias，y=wx+b'''
        return self.activator(VectorOp.dot(input_vec,self.weights)+self.bias)

    def train(self,input_vecs,labels,iteration,rate):
        '''输入训练数据：一组向量、对应的label；及迭代次数、学习率'''
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def _one_iteration(self,input_vecs,labels,rate):
        '''一次迭代，把所有训练数据过一遍'''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples=zip(input_vecs,labels)
        for (input_vec,label) in samples:
            output=self.predict(input_vec)
            self._updata_weight(input_vec,output,label,rate)

    def _updata_weight(self,input_vec,output,label,rate):
        '''按感知器规则更新权重'''
        # 首先计算本次更新的delta
        # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
        delta=label-output
        self.weights=VectorOp.element_add(
            self.weights,VectorOp.scala_multiply(input_vec,rate*delta))
        self.bias += rate*delta

def f(x):
    '''定义激活函数'''
    return 1 if x>0 else 0

def get_training_dataset():
    #输入向量列表
    input_vecs=[[1, 1], [0, 0], [1, 0], [0, 1]]
    #对应的labels，要与输入向量对应
    labels=[1,0,0,0]
    return input_vecs,labels

def train_and_preceptron():
    '''训练感知器,实现逻辑与'''
    #创建感知器，输入参数为2个，；因为逻辑与是二元函数，激活函数为f
    p=Perceptron(2,f)
    #训练，迭代10次，学习率为0.1
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == '__main__':
    and_preceptron=train_and_preceptron()
    print(and_preceptron)   #打印训练获得的weights，bias

    #test
    print('1 and 1 = %d'%and_preceptron.predict([1,1]))
    print('1 and 0 = %d'%and_preceptron.predict([1,0]))
    print('0 and 1 = %d'%and_preceptron.predict([0,1]))
    print('0 and 0 = %d'%and_preceptron.predict([0,0]))

    
    
