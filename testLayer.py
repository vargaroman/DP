from keras.models import Model, Sequential
from keras.layers import Dense
import VargaLayer as VL

model=Sequential()
model.add(VL.VargaLayer(32, input_shape = (16,)))
model.add(Dense(8, activation='softmax'))
model.summary()
