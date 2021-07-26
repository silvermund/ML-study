from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
# print(x_train[1], type(x_train[1]))

print(y_train[0], type(y_train[0]))
# 3

print(len(x_train[0]), len(x_train[1])) #87 56

# print(x_train[0].shape) #'list' object has no attribute 'shape' 

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(type(x_train)) #<class 'numpy.ndarray'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #2376
# print("뉴스기사의 최대길이 : ", max(len(x_train))) 이건 안됨
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
print(type(x_train), type(x_train[0]))
print(x_train[0])

# y 확인
print(np.unique(y_train)) #46

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(8982, 46) (2246, 46)

# 2. 모델구성

# 실습, 완성해보세요!!!
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=100))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=1)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)