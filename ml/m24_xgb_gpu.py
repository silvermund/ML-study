from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from xgboost.callback import LearningRateScheduler
import matplotlib.pyplot as plt
import time



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
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, #n_jobs=16
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0
)

#3. 훈련

start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)])
print("걸린시간 : ", time.time() - start_time)

#4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# results :  0.9348959737066168
# r2 :  0.9348959737066168


# print("========================================")
# hist = model.evals_result()
# print(hist)

#n_jobs=1
# 걸린시간 :  9.76607346534729
# results :  0.9290692082181989
# r2 :  0.9290692082181989

#n_jobs=4
# 걸린시간 :  7.362780570983887
# results :  0.9290692082181989
# r2 :  0.9290692082181989

#n_jobs=8
# 걸린시간 :  8.34700345993042
# results :  0.9290692082181989
# r2 :  0.9290692082181989

#n_jobs=16
# 걸린시간 :  11.533358097076416
# results :  0.9290692082181989
# r2 :  0.9290692082181989

#gpu_hist
# 걸린시간 :  46.91599917411804
# results :  0.9235823029360168
# r2 :  0.9235823029360168

#gpu_hist gpu_id=0
# 걸린시간 :  37.295005559921265
# results :  0.9235823029360168
# r2 :  0.9235823029360168
