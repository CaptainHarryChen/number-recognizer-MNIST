import numpy as np
import random
from progress.bar import Bar

def sigmoid(z):
    '''逻辑函数'''
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    '''逻辑函数导数'''
    return sigmoid(z) * (1 - sigmoid(z))



class Network:
    '''
    神经网络类
        sizes:一个list，每层神经元数量

        num_layer:神经元层数
        weights:二维array，每个神经元每个输入端的权重，不包括输入神经元
        biases:竖向量，每个神经元的偏置，不包括输入神经元
    '''
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes) ]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    def feedforward(self,a):
        '''
        将a作为神经网络输入值，返回输出值
            a为列向量
        '''
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def evaluate(self,test_data):
        '''
        用test_data测试神经网络，输出正确率
            test_data为(X,Y)列向量组的list
        '''
        sum = 0
        '''print(test_data[0][0])
        print(self.feedforward(test_data[0][0]))
        print(test_data[0][1])'''
        for X,Y in test_data:
            a = self.feedforward(X)
            sum+=int(Y[np.argmax(a)])
        print("Accuracy %d / %d" % (sum,len(test_data)))

    def cost_derivative(self,a,y):
        '''代价函数的导数'''
        return a - y

    def back_prop(self,X,Y):
        '''反向传播算法，计算输入为X，期望输出为Y时，权重w和偏置b的修改值'''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        tmp = X
        a = [X]
        z = []
        for w,b in zip(self.weights,self.biases):
            tmp = np.dot(w,tmp) + b
            z.append(tmp)
            tmp = sigmoid(tmp)
            a.append(tmp)
        delta = sigmoid_derivative(z[-1]) * self.cost_derivative(a[-1],Y)
        nabla_w[-1] = np.dot(delta,a[-2].T)
        nabla_b[-1] = delta
        for l in range(2,self.num_layers):
            delta = np.dot(self.weights[-l + 1].T,delta) * sigmoid_derivative(z[-l])
            nabla_w[-l] = np.dot(delta,a[-l - 1].T)
            nabla_b[-l] = delta
        return nabla_w,nabla_b

    def update_mini_batch(self,mini_batch,eta):
        '''
        用mini_batch数据更新网络
            mini_batch为(X,Y)的list
            eta为学习速率
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for X,Y in mini_batch:
            delta_nabla_w,delta_nabla_b = self.back_prop(X,Y)
            nabla_w = [w + dw for w,dw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [b + db for b,db in zip(nabla_b,delta_nabla_b)]
        m = len(mini_batch)
        self.weights = [w - eta / m * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - eta / m * nb for b,nb in zip(self.biases,nabla_b)]
    
    def SGD(self,train_data,epochs=30,mini_batch_size=10,eta=3.0,test_data=None):
        '''
        随机梯度下降算法，训练网络
            train_data:二元向量组(X,Y)的list，X为输入向量，Y为期望输出向量
            epochs:迭代期数量，即训练次数
            mini_batch_size:一次训练采样的数据量
            eta:学习速率
            test_data:测试数据，可用于评估网络
        '''
        for i in range(epochs):
           random.shuffle(train_data)
           n = len(train_data)
           print("Trainning %d/%d" % (i,epochs))
           bar = Bar("Trainning",max=1.0 * n / mini_batch_size, fill='@', suffix='%(percent)d%%')
           for j in range(0,n,mini_batch_size):
               self.update_mini_batch(train_data[j:j + mini_batch_size],eta)
               bar.next()
           bar.finish()
           if test_data:
               self.evaluate(test_data)
        print("Finish trainning.")
