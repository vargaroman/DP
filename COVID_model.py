import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping


IMAGE_SIZE = 64
SEED = 42
BATCH_SIZE = 64

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
folder="directory/folder path"

disease_types=['COVID', 'non-COVID']
source_dir = 'datasetcovid'
data_dir = os.path.join(source_dir)

all_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(data_dir, sp)):
        all_data.append(['{}/{}'.format(sp, file), defects_id, sp])
data = pd.DataFrame(all_data, columns=['File', 'DiseaseID','Disease Type'])
data.head()

data = data.sample(frac=1, random_state=SEED)
data.index = np.arange(len(data)) # Reset indices
data.head()

def read_image(filepath):
    image = cv2.imread(os.path.join(data_dir, filepath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

data.info()

formated_data = np.zeros((data.shape[0], IMAGE_SIZE, IMAGE_SIZE))

for i, file in tqdm(enumerate(data['File'].values)):
    image = read_image(file)
    if image is not None:
       formated_data[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))

formated_data = formated_data / 255.
formated_data = formated_data.reshape((formated_data.shape[0], formated_data.shape[1],formated_data.shape[2], 1))
print('Data Shape: {}'.format(formated_data.shape))

formated_data_labels = data['DiseaseID'].values

train_data, test_data, train_labels, test_labels = train_test_split(formated_data, formated_data_labels, test_size=0.3, random_state=SEED)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

from CustomLayers.HorizontalFlipLayer import HorizontalFlipLayer
from CustomLayers.RotationLayer import RotationLayer
from CustomLayers.BrightnessLayer import BrightnessLayer
from CustomLayers.SandPLayer import SandPLayer
from CustomLayers.VerticalFlipLayer import VerticalFlipLayer

flipLayer = HorizontalFlipLayer(None, input_shape=(64, 64, 1))
verticalFlipLayer = VerticalFlipLayer(None, input_shape=(64, 64, 1))
brightnessLayer = BrightnessLayer(None, input_shape=(64, 64, 1), lower_bound=-0.2, upper_bound=0.2)
rotatioLayer = RotationLayer(None, input_shape=(64, 64, 1), upper_bound=10, lower_bound=-10)
sandpLayer = SandPLayer(None, input_shape=(64, 64, 1), noise_ratio=0.02, sandp_ratio=0.5)

model = Sequential()
# model.add(flipLayer)
# model.add(verticalFlipLayer)
# model.add(rotatioLayer)
# model.add(sandpLayer)
# model.add(brightnessLayer)
model.add(Conv2D(48, 5,input_shape=(64, 64, 1), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, strides = 2))
model.add(Conv2D(64, 5, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, strides = (2,2)))
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1024, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10
)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'], run_eagerly=True)

saved_model = "weights.best.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs = 100, batch_size = BATCH_SIZE, callbacks=[checkpoint, es])


print("Loading model....")
model = load_model('weights.best.hdf5', custom_objects={'HorizontalFlipLayer': HorizontalFlipLayer,"VerticalFlipLayer":VerticalFlipLayer, 'RotationLayer':RotationLayer, 'BrightnessLayer':BrightnessLayer, 'SandPLayer':SandPLayer})
y_pred = model.predict(test_data)
y_int = np.zeros_like(y_pred)
print(y_int.shape)
y_int[y_pred > 0.5] = 1
confusion_matrix_result = confusion_matrix(test_labels, y_int)
print(confusion_matrix_result)
print(classification_report(test_labels, y_int))
