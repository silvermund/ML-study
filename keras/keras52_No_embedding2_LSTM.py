from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np

# 1. 데이터
docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요', ' 재미없어요', 
        '너무 재미없다', '참 재밌네요',' 청순이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고예요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, 
# '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, 
# '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, 
# '재미없다': 23, '재밌네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], 
# [22], [2, 23], [1, 24], [25, 3, 26, 27]]

pad_x = pad_sequences(x, padding='pre', maxlen=5) #post
print(pad_x)
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]

# print(pad_x.shape) 
# print(np.unique(pad_x))

pad_x = pad_x.reshape(13, 5, 1)

# 2. 모델
model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=77, input_length=5))
model.add(LSTM(32, input_shape=(5,1)))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)