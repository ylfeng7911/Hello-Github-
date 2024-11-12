import numpy as np

result_tanh = []
result_sigmoid = []
result_relu = []

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0-tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return 1 * (x>0)

class NeuralNetwork:
    def __init__(self,layers,activation = 'relu'):
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else :
            self.activation = relu
            self.activation_derivative = relu_derivative
        self.weights = []

        #初始化输入层与隐含层的权重
        for i in range(1,len(layers) - 1):# i = 1
            tmp =  2 * np.random.random([layers[i - 1] + 1,layers[i] + 1]) - 1  # (2+1) * (2+1),  [-1,1]
            self.weights.append(tmp)

        #初始化输出层的权重
        tmp = 2 * np.random.random([layers[-2] + 1,layers[-1]]) - 1
        self.weights.append(tmp)


    def run(self,x):
        z = np.hstack([np.ones(1),x])
        a = z
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i])
            a = self.activation(z)
        return a
        #a是行向量


    def train(self,X,Y,learning_rate = 0.6,epochs = 10000):
        #训练之前数据预处理
        X = np.hstack([np.ones([X.shape[0],1]),X])        #X原来是4x2的矩阵，在X左边拼上一列1，作为偏置b，然后X变为4x3

        #迭代训练
        for k in range(1,epochs+1):
            # 把X中每个行向量看成一组数据,从X中随机取一个样本作为一轮的初值
            n = np.random.randint(X.shape[0])
            a = [X[n]]
            z = [X[n]]

            # 把所有激励值放入列表a
            for i in range(len(self.weights)):
                # print('a[i]:',a[i].shape)
                # print(self.weights[i].shape)
                z.append(np.dot(a[i],self.weights[i])) #z是行向量
                a_tmp = self.activation(z[-1])
                a.append(a_tmp)         #a由[3 3 1]维的行向量组成

            #反向传播过程
            #均方误差函数 E = 1/2 * (Y-a)^2
            error = Y[n] - a[-1]
            deltas = [-1 * error * self.activation_derivative(z[-1])]

            for j in range(len(z) - 2, 0, -1):#j = 1
                deltas.append(np.dot(deltas[-1],self.weights[j].T) * self.activation_derivative(z[j]))

            # 排序逆转为delta2 ,delta3
            # 为什么没有delta1？ 因为第一层没有权重
            deltas.reverse()

            #更新权重
            for i in range(len(self.weights)):
                a_prior = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] -= learning_rate * np.dot(a_prior.T,delta)

            if k % 100 == 0:
                res = []
                for x in X[:,1:3]:
                    res.append(self.run(x))
                error = [0,1,1,0] - np.reshape(res,-1)
                error_abs = np.abs(error)
                mean_error = np.mean(error_abs)
                # print("mean_error=",mean_error)
                if mean_error < 0.008:
                    print("平均误差小于0.008，停止迭代，迭代次数为%d"%k)
                    break
            if k==epochs:
                print("迭代次数达到上限，为%d"%k)
        if self.activation == tanh:
            result_tanh.append(k)
        elif self.activation == sigmoid:
            result_sigmoid.append(k)
        else:
            result_relu.append(k)

    def reset_weights(self):
        new_weights = []

        #初始化输入层和隐含层权重
        for i in range(1,len(layers) - 1):#i = 1
            r = self.weights[i-1].shape[0]
            c = self.weights[i-1].shape[1]
            new_weights.append(2 * np.random.random([r,c]) - 1)

        #初始化输出层
        r = self.weights[-1].shape[0]
        c = self.weights[-1].shape[1]
        new_weights.append(2 * np.random.random([r,c]) - 1)

        self.weights = new_weights



if __name__ == '__main__':
    layers = [2,2,1]
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    Y = np.array([0,1,1,0])

    nn1 = NeuralNetwork(layers,activation='tanh')
    for i in range(30):
        nn1.train(X,Y)
        nn1.reset_weights()
    print('采用tanh函数作为激励函数时，30次实验的平均迭代次数为',np.mean(result_tanh))
