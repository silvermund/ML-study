# 실습
# 데이터는 diabetes 

# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
#  최적의 R2 값과 피처임포턴스 구할 것

# 2. 위 스레드 값으로 SelectFromModel 돌려서
# 최적의 피쳐 갯수 구할 것

# 3. 위 피처 갯수로 피처 갯수를 조정한 뒤
# 그걸로 다시 랜덤서치 그리드서치해서
# 최적의 R2 구할 것

# 1번값과 3번값 비교 # 0.47 이상

from numpy.lib.function_base import select
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=66)

#2. 모델


# n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parameters =[
#     {'n_jobs' : [-1], 'n_estimators' : [100,200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
#     {'n_jobs' : [-1], 'max_depth' : [5,6,7], 'min_samples_leaf' : [6,7,11], 'min_samples_split' : [3,4,5]},
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7], 'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1], 'min_samples_split' : [2,3,5,10]},
# ]

model = XGBRegressor(n_jobs=-1)
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)



#3. 훈련
start_time = time.time()

model.fit(x_train, y_train)

#4. 평가, 예측
# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_score_ : ", model.best_score_)

# print("model.score: ", model.score(x_test, y_test))

# y_predict = model.predict(x_test)
# print("정답률 : ", accuracy_score(y_test, y_predict))

# print("걸린시간 : ", time.time() - start_time)

score = model.score(x_test, y_test)
print("model.score : ", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)


# for thresh in thresholds:
#     # print(thresh)
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     # print(selection)

#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)

#     print(select_x_train.shape, select_x_test.shape)

#     selection_model = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)

#     y_predict = selection_model.predict(select_x_test)

#     score = r2_score(y_test, y_predict)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
#             score*100))

selection = SelectFromModel(model, threshold=0.09, prefit=True)
print(selection)

select_x_train = selection.transform(x_train)
select_x_test = selection.transform(x_test)

print(select_x_train.shape, select_x_test.shape)

selection_model = XGBRegressor(n_jobs=-1)
selection_model.fit(select_x_train, y_train)

y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print("Thresh=%.3f, n=%d, R2: %.2f%%" %(0.09, select_x_train.shape[1],
        score*100))