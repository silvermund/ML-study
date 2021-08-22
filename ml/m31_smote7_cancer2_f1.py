# 실습
# cancer로 만들 것
# 지표는 _f1
# 라벨 0을 112개 삭제

from imblearn.over_sampling import SMOTE
from numpy.lib.function_base import average
from pandas.core.algorithms import value_counts
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

# datasets = datasets.values

x = datasets.data
y = datasets.target

# print(pd.Series(y).value_counts())

# print(x.shape, y.shape) # (569, 30) (569,)
y = np.array(y).reshape(569,1)
print(y.shape)

union = np.concatenate((x,y), axis=1)
print(union)

union = union[union[:, 30].argsort()]
print(union)

x = union[112:,0:-1] 
y = union[112:,-1]

print(x.shape, y.shape)
print(y)



print(x.shape, y.shape) # (569, 30) (569,)

print(pd.Series(y).value_counts())
# 1    357
# 0    212

print(y)

print(pd.Series(y).value_counts())


##########################################################
##### 라벨 대통합!!!
##########################################################
print("====================================")
# for i in range(y.shape[0]):
#     if y[i] == 9.0:
#         y[i] = 8.0

# for index, value in enumerate(y):
#     if value == 9:
#          y[index]= 2
#     elif value == 8:
#          y[index]= 2
#     elif value == 7:    
#          y[index]= 1
#     elif value == 6:
#          y[index]= 1
#     elif value == 5:    
#          y[index]= 1
#     elif value == 4:
#          y[index]= 0
#     elif value == 3:
#          y[index]= 0


print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)#, stratify=y)

print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score : ", score)


y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score ", f1)

# model.score :  0.9456521739130435
# f1_score  0.9264588329336532


###################################### smote 적용 #################################
print("=============================== smote 적용 ===============================")

smote = SMOTE(random_state=66, k_neighbors=3)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
end_time = time.time() - start_time

print(pd.Series(y_smote_train).value_counts())


print(x_smote_train.shape, y_smote_train.shape)
#(159, 13) (159,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", (x_smote_train.shape, y_smote_train.shape))
print("smote 전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote 후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())
print("smote 경과시간 ", end_time)

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("model2.score : ", score)

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score ", f1)

# smote 경과시간  0.0019991397857666016
# model2.score :  0.9565217391304348
# f1_score  0.9420289855072463
