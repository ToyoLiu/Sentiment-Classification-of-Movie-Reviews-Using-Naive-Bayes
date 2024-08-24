from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score


# 定义训练集所占比例train_ratio以及词袋模型的参数 max_features
def preprocessing(train_ratio, max_features):
    num_words = 10000
    max_length = 256

    # 加载IMDB电影评论数据集
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    # 加载数据后打印一个样本
    # print(train_data[0])

    # 合并训练集和测试集以进行预处理
    data = np.concatenate((train_data, test_data), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    # 将整数序列转换为字符串
    data = [' '.join([str(word) for word in sample]) for sample in data]

    # 对文本进行分词
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    # 分词后打印
    # print(sequences[0])

    # 显示分词后的单词数量
    word_count = [len(s.split()) for s in data]

    # 对序列进行填充和截断
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # 填充后打印
    # print(padded_sequences[0])

    # 显示填充前后的序列长度
    seq_len = [len(s) for s in sequences]

    # 划分训练集和测试集
    split = int(50000 * train_ratio / 100)
    train_data = padded_sequences[:split]
    test_data = padded_sequences[split:]

    # 划分后打印
    # print(train_data[0])

    train_labels = labels[:split]
    test_labels = labels[split:]

    # 将整数序列转换回文本序列
    train_text = tokenizer.sequences_to_texts(train_data)
    test_text = tokenizer.sequences_to_texts(test_data)

    # 转换文本后打印
    # print(train_text[0])

    # 将文本序列转换为字符串数组
    train_text = np.array(train_text)
    test_text = np.array(test_text)

    # 创建一个词袋模型
    vectorizer = CountVectorizer(max_features=max_features)

    # 将整数序列转换为词袋向量表示
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)

    # 词袋表示后打印
    # print(train_features)

    # 使用词袋特征进行分类
    return train_features, train_labels, test_features, test_labels


# 计算分类准确度等指标
def calculate_indicators(train_features, train_labels, test_features, test_labels):
    class NaiveBayesClassifier:
        def __init__(self, alpha=1):
            self.alpha = alpha
            self.class_probabilities = None
            self.feature_probabilities = None

        def fit(self, X_train, y_train):
            # 计算先验概率,概率可输出观察
            self.class_probabilities = self.calculate_class_probabilities(y_train)

            # 计算条件概率
            self.feature_probabilities = self.calculate_feature_probabilities(X_train, y_train)

        # 计算先验概率
        def calculate_class_probabilities(self, y_train):
            class_counts = np.bincount(y_train)
            total_samples = y_train.shape[0]
            class_probabilities = (class_counts + self.alpha) / (total_samples + self.alpha * len(class_counts))
            return class_probabilities

        # 计算条件概率
        def calculate_feature_probabilities(self, X_train, y_train):
            num_classes = len(np.unique(y_train))
            num_features = X_train.shape[1]
            feature_probabilities = np.zeros((num_classes, num_features))

            for class_label in range(num_classes):
                class_samples = X_train[y_train == class_label]
                total_count = class_samples.sum()
                feature_probabilities[class_label] = (class_samples.sum(axis=0) + self.alpha) / (
                        total_count + self.alpha * num_features)

            return feature_probabilities

        # 利用已训练好的朴素贝叶斯模型进行预测
        def predict(self, X_test):
            # 根据类的个数和测试样本数,初始化预测结果数组
            num_classes = len(self.class_probabilities)
            num_samples = X_test.shape[0]
            predictions = np.zeros(num_samples)

            # 计算该样本在每个类中的后验概率分数
            for i in range(num_samples):
                sample = X_test[i]
                class_scores = np.log(self.class_probabilities) + np.log(
                    self.feature_probabilities[:, sample.nonzero()[1]]).sum(axis=1)
                predictions[i] = np.argmax(class_scores)

            return predictions

    NB = NaiveBayesClassifier()
    NB.fit(train_features, train_labels)
    y_predict = NB.predict(test_features)
    accuracy = accuracy_score(test_labels, y_predict)
    precision = precision_score(test_labels, y_predict)
    f1 = f1_score(test_labels, y_predict)
    return accuracy


# 数据集划分对结果影响
train_ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
accuracies = []
for i in range(10):
    train_features, train_labels, test_features, test_labels = preprocessing(train_ratio[i], 512)
    accuracy = calculate_indicators(train_features, train_labels, test_features, test_labels)
    accuracies.append(accuracy)
    print("Train Ratio: ", train_ratio[i], "Accuracy: ", accuracy)
plt.plot(train_ratio, accuracies)
plt.xlabel("train ratio %")
plt.ylabel("accuracy")
plt.title("Training_Set_Ratio_Accuracy")
plt.show()



# 词袋最大特征数目对结果影响
max_features = [256, 512, 1024, 2048, 4096, 8192, 10000, 15000, 20000, 30000]
accuracies = []
for i in range(10):
    train_features, train_labels, test_features, test_labels = preprocessing(50, max_features[i])
    accuracy = calculate_indicators(train_features, train_labels, test_features, test_labels)
    accuracies.append(accuracy)
    print("Max Features: ", max_features[i], "Accuracy: ", accuracy)
plt.plot(max_features[:5], accuracies[:5])
plt.title("Maximum_Features_Accuracy1")
plt.xlabel("max features")
plt.ylabel("accuracy")
plt.show()
plt.plot(max_features, accuracies)
plt.title("Maximum_Features_Accuracy2")
plt.xlabel("max features")
plt.ylabel("accuracy")
plt.show()



