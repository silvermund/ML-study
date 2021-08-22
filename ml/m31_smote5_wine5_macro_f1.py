# 실습1

# 라벨의 범위를 바꿀 것

# 기존
# 3,4 -> 0
# 5,6,7 -> 1
# 8,9 ->2

# 범위 변경
# 3,4,5 -> 0
# 6      -> 1
# 7,8,9, -->2

# 스모트하기 전 후 비교해볼 것 
# 더좋아하지는 기준은 macro-f1



from imblearn.over_sampling import SMOTE
from numpy.lib.function_base import average
from pandas.core.algorithms import value_counts
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

##########################################################
##### 라벨 대통합!!!
##########################################################
print("====================================")
# for i in range(y.shape[0]):
#     if y[i] == 9.0:
#         y[i] = 8.0

for index, value in enumerate(y):
    if value == 9:
         y[index]= 2
    elif value == 8:
         y[index]= 2
    elif value == 7:    
         y[index]= 2
    elif value == 6:
         y[index]= 1
    elif value == 5:    
         y[index]= 0
    elif value == 4:
         y[index]= 0
    elif value == 3:
         y[index]= 0


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
# model.score :  0.643265306122449

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score ", f1)


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
# model2.score :  0.6228571428571429

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score ", f1)