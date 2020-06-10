
from sklearn.model_selection import KFold
import numpy as np


kf = KFold(n_splits=10,shuffle=True)
path = r'C:\Users\z7512\Desktop\KNN\train.txt'
data = np.loadtxt(path, dtype=float, delimiter=',')
for train, test in kf.split(data):
    print(train)
    print(test)
