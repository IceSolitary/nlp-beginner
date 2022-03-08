# NLP-Beginner report

# Task 1

---

将数据集以8：2的比例划分为训练集和验证集，用如下配置在训练集上训练20w步，得到以下结果：

1. Bow
    1. batch_size = 16, lr = 0.01, acc = 0.5543
    2. batch_size = 16, lr = 0.1, acc = 0.6049
    3. batch_size = 16, lr = 1, acc = 0.6258
    4. batch_size = 1,   lr = 1, acc = 0.5965
    5. batch_size = 1,   lr = 0.1, acc = 0.5475
    6. 当lr >1 时出现nan ， 此时权重w更新过快，导致数值过小，送入softmax中计算出nan。
2. ngram
    1. batch_size = 16, lr =0.1, acc = 0.6459
    2. batch_size = 16, lr = 1, acc = 0.6519

实验结论：MBGD结果好于BGD，可能是因为MBGD在训练过程中引入了更多噪声，但相比于SGD又不易收敛到局部最优，从而使模型的泛化能力更强，在验证集的表现上更好。在本次实验中，学习率较小的情况下出现了不收敛的情况，并未出现学习率过大导致损失震荡或者增大的情况。在特征选择上，ngram显然优于词袋模型，ngram考虑了token之间的位置关系，但依旧存在着特征矩阵过大的问题，在实际实验中，手写的ngram发生了爆内存的情况，使用sklearn中的ngram模型，其模型使用scipy库存储稀疏矩阵。

# Task 2

---

## 实验设置：

实验使用glove.6B.50d.txt预训练词向量，分别使用了两个模型：LSTM和TextCNN，实验设置如下：

LSTM设置：embedding_size=50, hidden_size=128,dropout=0.5，损失函数：交叉熵，优化器：Adam，lr=0.001。

TextCNN设置: embedding_size=50, 卷积核大小采用2，3，4三种大小的卷积核，卷积核数量=64， dropout=0.5，将三种卷积提取出的特征拼接起来。损失函数：交叉熵，优化器：Adam，lr=0.0001。

## 实验结果：

![accuracy.png](NLP-Beginn%20a6454/accuracy.png)

![loss.png](NLP-Beginn%20a6454/loss.png)

在TextCNN中学习率设置为0.001时，出现学习率过大，损失上升的情况，对于不同网络需要设置不同的超参数。可以看出，Lstm在序列建模的效果要优于TextCNN。

# Task 3

---

## 实验设置：

batch_size = 1024

embedding_size=300

hidden_size=100

损失函数：交叉熵

优化器：Adam,lr = 0.004

epochs =20

梯度裁剪的max_grad_norm = 10.0 

动态学习率调整的的patience=5，factor=0.5

使用词向量glove.6B.300d.txt

在数据上做了一些预处理，包括词形还原，缩写展开，数字替换，标点符号替换，符号替换等，使用unk处理oov问题

## 实验结果：

![loss.png](NLP-Beginn%20a6454/loss%201.png)

![accuracy.png](NLP-Beginn%20a6454/accuracy%201.png)

在数据处理前，acc只有0.68左右，在处理数据后，acc上升到0.78左右，对于原论文中得到注意力打分后得矩阵拼接操作，因其维度过大，曾尝试改成sum操作，发现效果不好，acc下降了十个百分点，思考原因可能是因为网络能力不够强，sum后一些语义叠加起来，网络难以识别。

# Task4

---

## 实验设置：

batch_size=256

embedding_size=50

hidden_size=50

优化器：Adam, lr =0.01

epochs =20

max_grad_norm=5.0

max_len=30

使用词向量glove.6B.50d.txt

使用自己手写的线性链CRF,使用动态规划计算配分函数，使用维特比算法进行解码

## 实验结果：

![F1-score.png](NLP-Beginn%20a6454/F1-score.png)

训练集上达到了0.95，验证集上只有0.8，存在一些过拟合的情况。

# Task5

---

## 实验设置：

因数据集较小batch_size设置为1，embedding_size=50, hidden_size=32，max_len=30，损失函数：交叉熵

## 实验结果：

![Untitled](NLP-Beginn%20a6454/Untitled.png)

生成结果存在生成句子过短的问题，原因可能是网络训练时间不够，二是采用贪心解码效果有限，三是数据集过小。