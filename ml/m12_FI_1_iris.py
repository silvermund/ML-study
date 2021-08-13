# 실습!!
# 피쳐임포턴스가 전체 중요도에서 25% 미만인 컬럼들을 제거하여 데이터셋을 재 구성 후 
# 각 모델별로 돌려서 결과 도출

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_iris()

Iris_Data = pd.DataFrame(data= np.c_[datasets['data'], datasets['target']], columns= datasets['feature_names'] + ['target'])
# Iris_Data['target'] = Iris_Data['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})
Iris_Data = Iris_Data.drop(columns=['sepal width (cm)'])

datasets.data = Iris_Data.iloc[:,:-1]
datasets.target = Iris_Data.iloc[:,[-1]]


# print(X_Data)

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

# print(model.feature_importances_)

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

# 결과비교
# 1.DecisionTreeClassifier
# acc :  0.9111111111111111
# 컬럼 삭제 후
# acc :  0.8888888888888888

# 2. RandomForestClassifie
# 기존
# acc :  0.8888888888888888
# 컬럼 삭제 후
# acc :  0.9111111111111111

# 3. XGBClassifier
# 기존
# acc :  0.9111111111111111
# 컬럼 삭제 후
# acc :  0.9111111111111111

