from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)



# model = DecisionTreeClassifier(max_depth=5)
# acc :  0.9629629629629629
# [0.0055201  0.         0.         0.01830348 0.         0.
#  0.04628889 0.         0.         0.06885097 0.07968982 0.37054514
#  0.41080159]

# model = RandomForestClassifier()
# acc :  1.0
# [0.09536133 0.04163091 0.01723338 0.03834708 0.03185662 0.04538997
#  0.14111778 0.01557433 0.02685925 0.15028402 0.11594965 0.10518215
#  0.17521353]

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