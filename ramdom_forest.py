

# ### 导入随面森林的相关库文件.
from sklearn.ensemble import RandomForestClassifier          # 导入随机森林的包
# from sklearn.model_selection import train_test_split         # 这个用于后台数据的分割
from sklearn.preprocessing import StandardScaler             # 数据的标准化
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV  # 网格搜索和随机搜索
path = r'C:\Users\z7512\Desktop\classification\data.txt'
data = np.loadtxt(path, dtype=float, delimiter=',')
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签

train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.8,test_size=0.2)
#sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
#n_estimators:树的数目 integer, optional (default=10)  整数，可选择(默认值为10)
#criterion：衡量分裂质量的性能（函数）。 受支持的标准是基尼不纯度的"gini",和信息增益的"entropy"（熵）
#
# 数据训练
scaler = StandardScaler()  # 标准化转换
train_data = scaler.fit_transform(train_data)  # 训练标准化对象
test_data = scaler.transform(test_data)


# criterion="gini"
clf = RandomForestClassifier(n_estimators=200,criterion="gini")
param_grid = {'max_depth':np.arange(30, 50, 2),'min_samples_leaf':np.arange(1, 3, 1)}  # 定义优化参数字典
grid = GridSearchCV(estimator = clf, param_grid = param_grid, cv=10, scoring='accuracy',n_jobs=-1) #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid.fit(train_data, train_label.ravel())
print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
clf = RandomForestClassifier(criterion="gini",max_depth = grid.best_params_['max_depth'],min_samples_leaf=grid.best_params_['min_samples_leaf'],min_samples_split = 2)
clf.fit(train_data, train_label.ravel())
predict_results = clf.predict(test_data)
print('criterion="gini')
print('训练集',accuracy_score(train_label.ravel(),  clf.predict(train_data)))
print('测试集',accuracy_score(test_label.ravel(), predict_results))

# criterion="entropy"
clf = RandomForestClassifier(n_estimators=200,criterion="entropy")
param_grid = {'max_depth':np.arange(10, 50, 2),'min_samples_leaf':np.arange(1, 3, 1)}  # 定义优化参数字典
grid = GridSearchCV(estimator = clf, param_grid = param_grid, cv=10, scoring='accuracy',n_jobs=-1) #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid.fit(train_data, train_label.ravel())
print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
clf = RandomForestClassifier(criterion="entropy",max_depth = grid.best_params_['max_depth'],min_samples_leaf=grid.best_params_['min_samples_leaf'],min_samples_split = 2)
clf.fit(train_data, train_label.ravel())
predict_results = clf.predict(test_data)
print('criterion="entropy')
print('训练集',accuracy_score(train_label.ravel(),  clf.predict(train_data)))
print('测试集',accuracy_score(test_label.ravel(), predict_results))