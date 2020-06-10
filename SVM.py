
from sklearn import svm
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler             # 数据的标准化
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV  # 网格搜索和随机搜索
# 1.读取数据集
path = r'C:\Users\z7512\Desktop\classification\data.txt'
data = np.loadtxt(path, dtype=float, delimiter=',')
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签
#2划分数据集
train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.8,test_size=0.2)
scaler = StandardScaler()  # 标准化转换
train_data = scaler.fit_transform(train_data)  # 训练标准化对象
test_data = scaler.transform(test_data)

#3.训练svm线性核分类器
classifier=svm.SVC(C=2,kernel='linear',decision_function_shape='ovr') # ovr:一对多策略 gamma高斯核精度
C=range(1,5)

param_grid = {'C':C}  # 定义优化参数字典
grid = GridSearchCV(estimator = classifier, param_grid = param_grid, cv=10, scoring='accuracy',n_jobs=-1) #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid.fit(train_data, train_label.ravel())


print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典


classifier=svm.SVC(C=grid.best_params_['C'],kernel='linear',decision_function_shape='ovr') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优
print("SVM线性核")
print("训练集：",classifier.score(train_data,train_label.ravel()))
print("测试集：",classifier.score(test_data,test_label.ravel()))


#4.训练svm高斯核分类器
classifier=svm.SVC(kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略 gamma高斯核精度
C=range(1,5)
gamma=np.arange(0,2,0.5)

param_grid = {'C':C,'gamma':gamma}  # 定义优化参数字典
grid = GridSearchCV(estimator = classifier, param_grid = param_grid, cv=10, scoring='accuracy',n_jobs=-1) #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid.fit(train_data, train_label.ravel())


print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典


classifier=svm.SVC(C=grid.best_params_['C'],kernel='rbf',decision_function_shape='ovr',gamma=grid.best_params_['gamma']) # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优
print("SVM高斯核")
print("训练集高斯核：",classifier.score(train_data,train_label.ravel()))
print("测试集线性核：",classifier.score(test_data,test_label.ravel()))