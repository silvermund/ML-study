from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from xgboost.callback import LearningRateScheduler
import matplotlib.pyplot as plt
import pickle


#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

# print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = XGBRegressor(n_estimators=2000, learning_rate=0.05, n_jobs=1)

#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=10
)

#4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# results :  0.9348959737066168
# r2 :  0.9348959737066168


print("========================================")
hist = model.evals_result()
print(hist)

# 저장
pickle.dump(model, open('./_save/xgb_save/m21_pickle.dat', 'wb'))

