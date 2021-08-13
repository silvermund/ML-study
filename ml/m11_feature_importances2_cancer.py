from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_breast_cancer()
print(dir(datasets))
'''
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

# model = DecisionTreeClassifier(max_depth=5)
# acc :  0.935672514619883
# [0.         0.0277196  0.         0.         0.         0.
#  0.         0.         0.         0.03666318 0.01431236 0.
#  0.00816738 0.         0.         0.         0.         0.
#  0.         0.         0.         0.04456092 0.         0.74944965
#  0.         0.         0.         0.1191269  0.         0.        ]

# model = RandomForestClassifier()
# acc :  0.9590643274853801
# [0.05541974 0.01723874 0.03549087 0.06401381 0.00526332 0.00379171
#  0.03856692 0.06925499 0.00340312 0.00458139 0.01678861 0.00526418
#  0.00688406 0.02778267 0.0039101  0.00275444 0.00423796 0.00537834
#  0.00442209 0.00420928 0.1366897  0.02061638 0.15903878 0.12539588
#  0.01482351 0.01508264 0.03864349 0.09428097 0.00756835 0.00920398]

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
'''