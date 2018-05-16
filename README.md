# 环境安装
集成安装环境Anaconda：https://www.anaconda.com/download/ (miniconda)

安装anacode后，自带jupyter notebook

测试jupyter是否成功安装:
> jupyter notebook

* 安装paddlepaddle，Baidu

pip install paddlepaddle # paddlepaddle是仅有的在macOS上面会出现问题的deep learning框架

问题描述：
```python
Fatal Python error: PyThreadState_Get: no current thread
Abort trap: 6
```
解决方案：https://www.cnblogs.com/charlotte77/p/8270710.html

本教程paddle版本：0.11.0

* 安装mxnet，DMLC->Amazon

pip install mxnet

本教程mxnet版本：1.1.0

* 安装tensorflow，Google

pip install tensorflow

本教程tensorflow版本：1.8.0

# 启动notebook
在deep-learning-101目录下，执行jupyter notebook

浏览器中输入http://localhost:8888

# 项目介绍
* lesson0：环境搭建以及numpy介绍，cifar10和mnist数据集
* lesson1：线性回归python版本，以及mxnet版本
* lesson2：多分类模型，python版本以及mxnet版本
* lesson3：向量化加速计算过程
* lesson4：2层网络
* lesson5：常用最优化算法
* lesson6：paddlepaddle线性回归入门
* utils：读取cifar10数据的库文件
* datasets：cifar10数据目录

# TroubleShooting
miaoyiyun@baidu.com
