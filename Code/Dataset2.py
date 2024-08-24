import tensorflow as tf
from tensorflow import keras
import numpy as np

# 加载IMDB数据集
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 映射字典
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 整数样本转文本
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 打印10个样本的文本
for i in range(10):
    print(decode_review(train_data[i]))