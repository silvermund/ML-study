from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
# datasets = load_iris()
datasets = load_diabetes()




x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

# model = DecisionTreeRegressor(max_depth=4)
# acc :  0.33339660919782466
# [0.03400704 0.         0.26623557 0.11279298 0.00124721 0.
#  0.01272153 0.         0.51986371 0.05313196]

# model = RandomForestRegressor()
# acc :  0.37893115243249
# [0.0629251  0.01119334 0.27628181 0.11047968 0.04422974 0.05977197
#  0.04442983 0.02113721 0.30180362 0.06774771]

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
