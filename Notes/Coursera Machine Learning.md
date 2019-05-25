# Coursera Machine Learning by AndrewNg



## Week 1

### 1.Introduction

**1.1 机器学习**：在进行特定编程的情况下，给予计算机学习能力的领域（Arthur Samuel）

​		          一个好的学习问题定义如下，一个程序被认为能从经验E中学习，解决任务T，达到性能度量值P，

​                          当且仅当，有了经验E后，经过P评判，程序在处理T时的性能有所提升（Tom Mitchell） 

**1.2 监督学习**：是指教计算机如何完成任务，给学习算法一个数据集，这个数据集由“正确答案”组成

​                          包括回归问题（连续值）和分类问题（离散值）

**1.3 无监督学习**：是指让计算机自己去学习，数据集没有任何标签或相同的标签

​                             常见的有聚类算法(cluster)，典型例子如鸡尾酒会问题（分离音频）、新闻事件分类、细分市场

### 2.Linear Regression with One Variable

**2.1 假设函数**：给学习算法一个训练集得到假设函数，用假设函数去预测

​                          一种可能的表达式：$h_\theta \left( x \right)=\theta_{0} + \theta_{1}x$（单变量线性回归）

**2.2 代价函数**：即平方误差函数（平方误差代价函数），是解决大多数问题特别是回归问题最常用的手段

​                          使代价函数的最小的模型参数确定的假设函数的建模误差最小

![代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/代价函数.png)

**2.3 梯度下降**：用来求函数最小值的算法，其思想为：开始随机选择一个参数的组合，计算代价函数

​                         然后寻找一个能让代价函数值下降最多的参数组合；选择不同的初始参数组合，

​                         可能会找到不同的局部最小值；

​                         批量梯度下降（batch gradient descent）算法的公式如下

![梯度下降](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/梯度下降.png)

​                         其中α为学习率，决定能让代价函数下降程度最大的方向向下迈出的步子有多大

​                         α太小下降会太慢，α太大可能会越过最低点导致无法收敛

​                         梯度下降法中，接近局部最低点时，会自动采取更小的幅度，所以没有必要再另外减小α

​                         其实就是不断求每个参数的偏导，要注意的是所有参数要同时更新

**2.4 梯度下降的线性回归**：将梯度下降和代价函数相结合，应用于拟合直线的线性回归算法

​	                                     ① 梯度下降算法和线性回归算法比较：

![梯度下降的线性回归](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/梯度下降的线性回归.png)

​                                             ② 对线性回归问题运用梯度下降算法，求代价函数的导数如下：

![求导](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/求导.png)

​                                           ③ 算法改写为：

![改写](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/改写.png)

​                                            之所以叫“批量”梯度下降，是因为梯度下降的每一步中都用到了所有的训练样本，

​                                            在计算微分求导项时需要进行m个训练样本的求和运算

### 3.Linear Algebra Review

线性代数基础知识，考过研的都会，保研的肯定更会了，不赘述



## Week 2

### 1.Linear Regression with Multiple Variables

**1.1 多维特征**：在单变量回归模型中引入多个变量（增加多个特征），假设函数变为：

![多元假设函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多元假设函数.png)

​                          引入x0=1，公式化简为：

![引入x0](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/引入x0.png)

​                         模型参数为n+1维向量，训练实例也是n+1维向量，特征矩阵X的维度为m*(n+1)，公式化简为：

![简化多元假设函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/简化多元假设函数.png)

**1.2 多变量梯度下降**：与单变量线性回归类似，构造代价函数为所有建模误差的平方和，如下：

![多变量代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多变量代价函数.png)

​                                     与单变量线性回归相同，要找出使代价函数最小的一系列的参数，应用梯度下降算法如下：

![多变量梯度下降](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多变量梯度下降.png)

​                                     求导后得：

![多变量梯度下降化简](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多变量梯度下降化简.png)

​                                     即特征数n≥1时，如下：

![多变量梯度下降的线性回归](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多变量梯度下降的线性回归.png)

**1.3 特征缩放**：使收敛所需的迭代次数更少，梯度下降的速度更快，常用的方法是均值归一化，即

​                          令![均值归一化](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/均值归一化.png)

​                           Sn有时直接用最大值减最小值代替，即归一化特征方程

**1.4 学习率**：绘制迭代次数和代价函数的图来观测算法在何时趋于收敛，梯度下降算法的每次迭代受学习率影响

​                      如果学习率α过小，达到收敛所需的迭代次数会很高，收敛速度会慢；

​                      如果学习率α过大，代价函数J(θ)可能不会在每次迭代中都下降，甚至可能越过局部最小值导致不收敛

**1.5 特征和多项式回归**：可以自由选择特征，并通过设计不同的特征，使用更复杂的函数拟合数据，如多项式函数

​                                         若采用多项式回归模型，在运行梯度下降算法前，要注意特征缩放

**1.6 正规方程**：通过求偏导令导数值为0求解使代价函数最小的参数，用向量表示即为：

![正规方程](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正规方程.png)

​                          在数值计算中，若矩阵不可逆，用伪逆(pinv)来计算，一般都是可逆的

​                          而且使用正规方程不需要特征缩放

梯度下降与正规方程优劣比较如下：

![梯度下降和正规方程的比较](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/梯度下降和正规方程的比较.png)

### 2.Octave/Matlab Tutorial

略



## Week 3

### 1.Logistic Regression

**1.1 假设函数**：需要用到Sigmoid函数，公式如下：

![逻辑函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/逻辑函数.png)

​                         图像如下：

![逻辑函数图像](C:\Users\guohouxiao\Desktop\ML Note\逻辑函数图像.png)

​                         当h(x)≥0.5时，预测y=1；当h(x)＜0.5时，预测y=0

​                         所以假设函数的作用是，对于给定的输入变量，根据选择的参数计算输出变量为1的可能性

**1.2 决策边界**：即模型分类的分界线，要注意决策边界不是训练集的属性，而是假设本身及其参数的属性，如下：

![决策边界](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/决策边界.png)

**1.3 代价函数**：不能用线性回归的代价函数，因为最后会得到非凸函数，逻辑回归的代价函数如下：

![逻辑回归代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/逻辑回归代价函数.png)

​                          两个对数函数的图像如下：

![对数函数图像](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/对数函数图像.png)

​                          简化版本如下：

![逻辑回归代价函数的简化](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/逻辑回归代价函数的简化.png)

​                          向量化表示如下：

![逻辑回归代价函数向量化](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/逻辑回归代价函数向量化.png)

​                          最小化代价函数的方法仍然是梯度下降法，如下：

![逻辑回归的梯度下降](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/逻辑回归的梯度下降.png)

**1.4 高级优化**：比梯度下降更高级的计算代价函数和偏导数项的算法

​                          如共轭梯度法BFGS（变尺度法）和L-BFGS（限制变尺度法）

​                          使用这些算法不需要手动选择学习率α，自带智能的内部循环（线性搜索算法）

​                          暂时不用了解细节，记住Octave函数fminunc的使用，即无约束最小化函数

**1.5 多元分类**：可分成n个二元分类问题，如下图所示：

![多元分类](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/多元分类.png)

### 2.Regularization

**2.1 过拟合**：过于强调拟合原始数据，丢失了算法的本质，不能很好的预测新数据

​                      即可以很好地适应性训练集数据但在新输入变量进行预测时效果不好

​                      解决方法有两种：一是舍弃一些特征，使用模型选择算法如PCA

​                                                     二是正则化，保留所有特征，但是减少参数大小

**2.2 正则化代价函数**：引入正则化参数λ，惩罚部分特征，如下：

![正则化代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化代价函数.png)

​                                     模型对比如下：

![模型对比](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/模型对比.png)

**2.3 正则化线性回归**：梯度下降法如下：

![正则化线性梯度下降](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化线性梯度下降.png)

​                                     调整后如下：

![正则化线性梯度下降调整](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化线性梯度下降调整.png)

​                                     正规方程法如下：

![正则化正规方程](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化正规方程.png)

**2.4 正则化逻辑回归**：注意假设函数与线性回归不同，新的代价函数如下：

![正则化逻辑回归代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化逻辑回归代价函数.png)

​                                     梯度下降如下：

![正则化逻辑梯度下降](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/正则化逻辑梯度下降.png)

​                                     注意θ0不参与正则化



## Week 4

### 1.Neural Networks: Representation

**1.1 神经网络模型**：结合大脑中的神经网络来理解

​                                 模型建立在很多神经元上，每个神经元又是一个学习模型

​                                 神经元（激活单元）采纳一些特征做输出，并且根据本身的模型提供一个输出

​                                 模型中的参数又称为权重

​                               ![神经网络模型](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/神经网络模型.jpg)

​                                 如上图所示，第一层为输入层，最后一层为输出层，中间的为隐含层

​                                 每一层都增加了一个偏差单位

​                                 激活单元和输出表达如下：

![神经网络模型表达](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/神经网络模型表达.png)

​                                 每一个a都是由上一层所有的x和每一个x所对应的决定的

​                                 这样从左到右的算法称为前向传播算法

**1.2 前向传播算法的向量化**：

![前向传播算法的向量化](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/前向传播算法的向量化.png)

**1.3 和逻辑回归的比较**：其实神经网络就像是logistic regression

​                                         只不过改变了logistic regression中的输入向量

​                                         特征值更加高级，由x和θ决定

**1.4 多元分类**：在输出层用n个神经元分别表示n类



## Week 5

### 1.Neural Networks: Learning

**1.1 代价函数**：在神经网络中可以有多输出量，代价函数比逻辑回归复杂，如下：

![神经网络的代价函数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/神经网络的代价函数.png)

**1.2 反向传播算法**：用于计算代价函数的偏导数

​                                 首先计算最后一层的误差，然后逐层反向求出各层误差，直到倒数第二层，如下：

![反向传播算法](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/反向传播算法.png)

​                                 即首先用正向传播方法计算出每一层的激活单元

​                                 利用训练集的结果与神经网络预测的结果求出最后一层的误差

​                                 然后利用该误差运用反向传播法计算出直至第二层的所有误差

**1.3 展开参数**：把参数从矩阵展开成向量的方法，如下：

![展开参数](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/展开参数.png)

**1.4 梯度检测**：为了避免神经网络在使用梯度下降算法时可能会产生的一些不易察觉的错误，获得最优解，如下：

![梯度检测](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/梯度检测.png)

​                         然后和反向传播算法计算的偏导值进行比较即可

​                         注意检测完毕进行训练之前，要及时关闭检测代码

**1.5 随机初始化**：初试参数为0，对于逻辑回归可行，但对神经网络不可行

​                             所以通常初始化为参数为正负ε之间的随机值

**1.6 小结**：训练神经网络

![训练神经网络的步骤](https://github.com/ErisRolo/MachineLearningNotes/blob/master/Images/训练神经网络的步骤.png)

神经网络部分建议参考3Blue1Brown的视频讲解



## Week 6

### 1.Advice for Applying Machine Learning

### 2.Machine Learning System Design



## Week 7

### 1.Support Vector Machines



## Week 8

### 1.Unsupervised Learning

### 2.Dimensionality Reduction



## Week 9

### 1.Anomaly Detection

### 2.Recommender Systems



## Week 10

### 1.Large Scale Machine Learning



## Week 11

### 1.Application Example: Photo OCR









