### 一个简单的深度学习练手项目：MNIST手写识别

这是一个基于pytorch的手写识别小项目，尽可能多的写了注释，用的两层卷积（3x3）和三层的全连接，总的准确度是97.1%，每一类的准确度如下：

| 0     |   1   | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
| ----- | :---: | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 99.2% | 98.8% | 95.7% | 97.8% | 98.2% | 97.2% | 96.7% | 98.9% | 94.3% | 94.3% |

训练中的loss和acc图如下（不知道为啥开始的时候loss那么大）：

![](https://gitee.com/programmerhyy/Image_Repo/raw/master/result_loss.png)

![](https://gitee.com/programmerhyy/Image_Repo/raw/master/result_acc.png)

