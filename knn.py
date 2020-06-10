
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

path = r'C:\Users\z7512\Desktop\classification\data.txt'
data = np.loadtxt(path, dtype=float, delimiter=',')
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签




#划分数据集
train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.8,test_size=0.2)
neighbors = np.arange(1, 20)
test_accuracy = np.empty(len(neighbors))

scaler = StandardScaler()  # 标准化转换
train_data = scaler.fit_transform(train_data)  # 训练标准化对象
test_data = scaler.transform(test_data)

#均等权重
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    knn.fit(train_data, train_label.ravel())
    test_accuracy[i] = knn.score(test_data, test_label.ravel())  # 使用测试集测试准确率
print('均等权重')
print('k从1到9')
print('test_accuracy\n',test_accuracy)
print('best_test_accuracy=',max(test_accuracy))
print('k_best=',np.where(test_accuracy==max(test_accuracy)))

#不均等权重
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance')
    knn.fit(train_data, train_label.ravel())
    test_accuracy[i] = knn.score(test_data, test_label)  # 使用测试集测试准确率
print('\n')
print('不均等权重')
print('k从1到9')
print('test_accuracy\n',test_accuracy)
print('best_test_accuracy=',max(test_accuracy))
print('k_best=',np.where(test_accuracy==max(test_accuracy)))