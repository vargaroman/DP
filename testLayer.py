from keras.models import Model, Sequential
from keras.layers import Dense
from VargaLayer import VargaLayer as VL
import matplotlib.pyplot as plt


plt.imshow(VL()('resized/test/yes/Y14.jpg')[0])

# model=Sequential()
# model.add(VargaLayer())
# model.add(Dense(8, activation='softmax'))
# model.summary()
