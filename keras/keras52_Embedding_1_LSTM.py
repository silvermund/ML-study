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

print(labels.shape) #(13, )

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

print(pad_x.shape) 
#(13, 5) -> 원핫인코딩 하면? -> (13, 5, 27)
#옥스포드 사전? (13, 5, 1000000) -> 6500만개 -> 자원소모가 크다

# word_size = len(token.word_index)
# print(word_size) #27
print(np.unique(pad_x))

# 2. 모델
model = Sequential()
model.add(Embedding(input_dim=28, output_dim=77, input_length=5))
# input_dim=라벨의 수,단어사전의 개수 output_dim=output 노드의 개수 input_length=문장의 수, 
# 파라미터 계산 = input_dim=27 * output_dim=77
# input_length는 벡터화에만 연관하므로 연산에 영향을 미치지 않는다.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 77)             2079
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                14080
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 16,192
# Trainable params: 16,192
# Non-trainable params: 0
# _________________________________________________________________


# model.add(Embedding(28, 77, input_length=5))
# model.add(Embedding(28, 77))


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 77)          2079
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                14080
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 16,192
# Trainable params: 16,192
# Non-trainable params: 0
# _________________________________________________________________

model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)
