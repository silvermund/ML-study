# 실습 : 
# CNN으로 변경하고
# 파라미터 변경해보고
# 노드의 갯수, activation도 추가
# epochs = [1,2,3]
# learning_rate 추가

# 나중 과제 : 레이어도 파라미터로 만들어봐!! (Dense, Conv1D)

import numpy as np
from sklearn.utils import validation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255


#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(filters=100, kernel_size=(2,2), padding='same', activation='relu', name='hidden1')(inputs)
    x = Conv2D(20, (2,2), activation='relu')(x)
    x = Conv2D(30, (2,2), padding='valid')(x)
    x = MaxPool2D()(x)
    x = Conv2D(15, (2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64)(x)
    x = Dense(32)(x)
    x = Dense(16)(x)
    x = Dense(8)(x)
    outputs = Dense(10, activation='sigmoid', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [100, 200, 300, 400, 500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0,5]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}

hyperparameters = create_hyperparameter()
# print(hyperparameters)
# model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) #, validation_split=0.2, epochs=2)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyperparameters, cv=5)

model = RandomizedSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)


print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)


# {'optimizer': 'rmsprop', 'drop': 0, 'batch_size': 100}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001328F581F40>
# 0.9766166806221008
# 100/100 [==============================] - 0s 4ms/step - loss: 0.0525 - acc: 0.9831
# 최종 스코어 :  0.9830999970436096