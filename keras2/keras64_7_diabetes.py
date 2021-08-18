import numpy as np
import pandas as pd
from sklearn.utils import validation
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


#1. 데이터
datasets = load_diabetes()


x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


scaler = StandardScaler()
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],10,1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],10,1).astype('float32')/255.

#2. 모델
def build_model(drop=0.2, optimizer='adam'):
    inputs = Input(shape=(10, 1), name='input')
    x = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu', name='hidden1')(inputs)
    x = Conv1D(16, 2, padding='same', activation='relu')(x)
    x = Conv1D(64, 2, padding='same', activation='relu')(x)
    x = Conv1D(64, 2, padding='same', activation='relu')(x)
    x = MaxPool1D()(x)
    x = Conv1D(256, 2, padding='same', activation='relu')(x)
    x = Conv1D(256, 2, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='mse')
    return model

def create_hyperparameter():
    batches = [1000, 2000]
    optimizers = Adam(learning_rate=0.001)
    dropout = [0.3, 0.4, 0,5]
    return {"batch_size" : batches, #"optimizer" : optimizers,
            "drop" : dropout}

hyperparameters = create_hyperparameter()
# print(hyperparameters)
# model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose=1) #, validation_split=0.2, epochs=2)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyperparameters, cv=5)

model = RandomizedSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)


print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)


# 1/1 [==============================] - 2s 2s/step - loss: 10546.0166 - acc: 0.0000e+00
# 최종 스코어 :  0.0