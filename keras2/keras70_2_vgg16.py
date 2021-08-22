from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
model = VGG16()
# model = VGG19()

vgg16.trainable=False   # 가중치를 동결한다, 훈련을 동결한다

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))

# model.trainable=False  # 가중치를 동결한다, 훈련을 동결한다


model.summary()

print(len(model.weights))             # 26 -> 30
print(len(model.trainable_weights))   # 0 -> 4

