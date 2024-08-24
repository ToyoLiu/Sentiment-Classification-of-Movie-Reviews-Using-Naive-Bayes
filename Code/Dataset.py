import numpy as np
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# 加载IMDB数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# 数据集统计
print('Training samples: {}'.format(len(x_train)))
print('Test samples: {}'.format(len(x_test)))

# 观察样本格式
print('Sample shape:', np.array(x_train[0]).shape)
print('Sample:', x_train[0])

# 统计每个样本的单词数量分布
lengths = [len(s) for s in x_train]
print('Minimum sample length:', min(lengths))
print('Maximum sample length:', max(lengths))
print('Mean sample length:', sum(lengths) / len(lengths))

# 绘制单词数量分布图
plt.hist(lengths)
plt.xlabel('Sample length')
plt.ylabel('Number of samples')
plt.show()
