#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def evaluate_gradient(loss_function, data, params):
    """计算loss_function对params的梯度，根据data的大小有不同策略。
    """
    pass


#batch gradient descent
#目标函数是凸函数，保证收敛到全局最优点
#目标函数是非凸函数，保证收敛到局部最优点
#重复计算，每次都计算全部样本的梯度，在迭代轮次之间，可能会重复计算相同的梯度
#无法在线更新
#for i in range(nb_epochs):
#  params_grad = evaluate_gradient(loss_function, data, params)
#  params = params - learning_rate * params_grad


#stochastic gradient descent
#每次选择1个样本，计算梯度。
#导致每次的梯度变化比较大，梯度的方差比较大。loss下降的不是很平滑。可以指数降低lr
#有机会跳出鞍点
#计算简单，可以支持在线更新
#for i in range(nb_epochs):
#  np.random.shuffle(data)
#  for example in data:
#    params_grad = evaluate_gradient(loss_function, example, params)
#    params = params - learning_rate * params_grad


#mini-batch gradient descent
#综合利用了batch和stochastic的优点
#降低stochastic的梯度方差，loss下降更平滑。
#能利用矩阵计算达到降低计算复杂度的问题
#for i in range(nb_epochs):
#  np.random.shuffle(data)
#  for batch in get_batches(data, batch_size=50):
#    params_grad = evaluate_gradient(loss_function, batch, params)
#    params = params - learning_rate * params_grad


#梯度下降法的缺点：
#1.学习速率很难确定，需要交叉验证(CV)
#2.学习速率调度问题。某些场景下，需要对学习速率进行衰减。预定义衰减策略没有利用到数据的信息
#3.所有参数的学习速率是一致的。特征稀疏程度不一样，需要不同的学习速率，对于较少出现的特征，需要增加学习速率，对于较多的特征，需要降低学习速率
#跳出鞍点


#momentum
#冲量算法，物理意义表示力F作用一段时间t后的效果：F*t = Mv
#v_now = rho * v_prev + lr * d_now(params)
#params = params - v_now
#v_prev = v_now
#收敛速度更快，loss的变化的方差更小

#nesterov accelerated gradient
#v_now = rho * v_prev + lr * d_now(params - rho * v_prev)
#params = params - v_now

#上述方法解决的是，学习方向的问题，可以动态调整学习速率
#下面介绍解决动态调整学习速率的问题

#adagrad jeff dean
#g_t_i = d(theta_t-1_i)
#theta_t_i = theta_t-1_i - lr / (math.sqrt(G_t-1_ii) + epsilon) * g_t_i
#G_t_ii = sigma_k=0^t g_k_i * g_k_i
#G是对角矩阵，每个对角线上的元素是过去梯度的平方和
#为什么要除以根号G，目前暂时不明朗。试验结果不加根号的话，效果反而很差
#学习速率越来越小，最后出现参数不更新的情况

#RMSprop, hinton
#对G矩阵做指数加权平均，G_t = gamma * G_t-1 + (1 - gamma) * g_t * g_t

#Adam
#将冲量法和自适应学习速率法结合
#方向g_t = beta1 * g_t-1 + (1 - beta1) * d_now(theta)
#学习速率衰减银子G_t = beta2 * G_t-1 + (1 - beta2) * d_now(theta) * d_now(theta)
#g_t = g_t / (1 - beta1 ** t) 偏差修正，原因：因为如果beta取值较大的情况下，g和G会取趋向于0的值，除以1-beta**t是为了使得g_t是梯度的无偏估计，对G同理
#G_t = G_t / (1 - beta2 ** t)
#theta_t = theta_t-1 - lr / (math.sqrt(G_t-1_ii) + epsilon) * g_t
#推荐值：lr=0.002 beta1 = 0.9 beta2=0.999, eps=1e-8

#AdaMax
#将学习速率衰减的2次方换成p次方，p越大，数值会越不稳定,why?
#p->无穷大时，数值稳定
#G_t = beta2 ** p * G_t-1 + (1 - beta2 ** p) * d_now(theta) ** p
#p->无穷大时,上式等于下式
#G_t = max(beta2 * G_t-1, math.abs(d_now(theta)))
#theta_t = theta_t-1 - lr / G_t * g_t
#推荐值：lr=0.002 beta1 = 0.9 beta2=0.999

#

def SGD(X, y, params, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params)
    params -= lr * params_gradient
    return params

def momentum(X, y, params, velocity=0.0, gamma=0.9, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params)
    velocity = gamma * velocity + learning_rate * params_gradient
    params -= velocity
    return params, velocity

def nag(X, y, params, velocity=0.0, gamma=0.9, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params - gamma * velocity)
    velocity = gamma * velocity + learning_rate * params_gradient
    params -= velocity
    return params, velocity

def adagrad(X, y, params, G=0, eps=1e-8, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params)
    G += params_gradient ** 2
    params -= learning_rate / np.sqrt(G + eps) * params_gradient
    return params, G

def RMSprop(X, y, params, G=0, gamma=0.9, eps=1e-8, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params)
    G = gamma * G + (1 - gamma) * params_gradient ** 2
    params -= learning_Rate / np.sqrt(G + eps) * params_gradient
    return params, G

def adam(X, y, params, correction=False, iter_num=1, velocity=0, beta1=0.9, G=0, beta2=0.99, eps=1e-8, learning_rate=0.1):
    params_gradient = eval_gradient(X, y, params)
    velocity = beta1 * velocity + (1 - beta1) * params_gradient
    G = beta2 * G + (1 - beta2) * params_gradient ** 2
    if correction:
        velocity /= (1 - beta1 ** iter_num)
        G /= (1 - beta2 ** iter_num)
    params -= learning_rate / (np.sqrt(G) + eps) * velocity
    return params, velocity, G

import numpy as np
import matplotlib.pyplot as plt


# 目标函数:y=x^2
def func(x):
    return np.square(x)


# 目标函数一阶导数:dy/dx=2*x
def dfunc(x):
    return 2 * x

def GD_momentum(x_start, df, epochs, lr, momentum):
    """
    带有冲量的梯度下降法。
    :param x_start: x的起始点
    :param df: 目标函数的一阶导函数
    :param epochs: 迭代周期
    :param lr: 学习率
    :param momentum: 冲量
    :return: x在每次迭代后的位置（包括起始点），长度为epochs+1
    """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr + momentum * v
        x += v
        xs[i+1] = x
    return xs

def demo2_GD_momentum():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate, Momentum', figsize=(20, 20))

    x_start = -5
    epochs = 6

    lr = [0.01, 0.1, 0.6, 0.9]
    momentum = [0.0, 0.1, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    row = len(lr)
    col = len(momentum)
    size = np.ones(epochs+1) * 10
    size[-1] = 70
    for i in range(row):
        for j in range(col):
            x = GD_momentum(x_start, dfunc, epochs, lr=lr[i], momentum=momentum[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, func(x), c=color[i], label='lr={}, mo={}'.format(lr[i], momentum[j]))
            plt.scatter(x, func(x), c=color[i], s=size)
            plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    demo2_GD_momentum()
