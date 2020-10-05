import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
import tensorflow.keras.layers as tflayers
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

imgdir = os.listdir('dataset')
i = 0
if not os.listdir().__contains__('cropped'):
    os.mkdir('cropped')
for imagename in imgdir:
    if (imagename.upper().__contains__("JPG") or imagename.upper().__contains__("JPEG") or imagename.upper().__contains__("PNG")):
        i += 1
        image = cv2.imread("dataset/"+imagename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        crop_img = image[extTop[1]:extBot[1],extLeft[0]:extRight[0]]
        cv2.imwrite("cropped/"+imagename,crop_img)

if not os.listdir().__contains__('resized'):
    os.mkdir('resized')
    os.mkdir('resized/test')
    os.mkdir('resized/train')
    os.mkdir('resized/test/yes')
    os.mkdir('resized/test/no')
    os.mkdir('resized/train/yes')
    os.mkdir('resized/train/no')
else:
    shutil.rmtree('resized')
    os.mkdir('resized')
    os.mkdir('resized/test')
    os.mkdir('resized/train')
    os.mkdir('resized/test/yes')
    os.mkdir('resized/test/no')
    os.mkdir('resized/train/yes')
    os.mkdir('resized/train/no')

IMG_SIZE = 224
croppedImagesDir = os.listdir('cropped')
i = 0
for croppedImageName in croppedImagesDir:
    i +=1
    image = cv2.imread("cropped/" + croppedImageName)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    if i % 4 == 0:
        if croppedImageName.upper().__contains__("Y"):
            cv2.imwrite("resized/test/yes/" + croppedImageName, image)
        else:
            cv2.imwrite("resized/test/no/" + croppedImageName, image)
    else:
        if croppedImageName.upper().__contains__("Y"):
            cv2.imwrite("resized/train/yes/" + croppedImageName, image)
        else:
            cv2.imwrite("resized/train/no/" + croppedImageName, image)

#precprocessing
classifier = Sequential()
classifier.add(tflayers.Convolution2D(16,3,3, input_shape=(224,224,3), activation='relu'))
classifier.add(tflayers.MaxPooling2D(pool_size=(2,2)))
classifier.add(tflayers.Convolution2D(32,3,3, activation='relu'))
classifier.add(tflayers.MaxPooling2D(pool_size=(2,2)))
classifier.add(tflayers.Convolution2D(64,3,3, activation='relu'))
classifier.add(tflayers.MaxPooling2D(pool_size=(2,2)))


classifier.add(tflayers.Flatten())
classifier.add(tflayers.Dropout(0.2))
classifier.add(tflayers.Dense(1024, activation='relu'))
classifier.add(tflayers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('resized/train',
                                                 target_size=(224,224),
                                                 batch_size=4,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('resized/test',
                                                 target_size=(224,224),
                                                 batch_size=4,
                                                 class_mode='binary')
from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='accuracy',
    mode='max',
    patience=6
)
classifier.fit(
    training_set,
    steps_per_epoch=int(training_set.samples/training_set.batch_size),
    epochs=35,
    validation_data=test_set,
    validation_steps=int(test_set.samples/test_set.batch_size),
    callbacks=[es]
)

classifier.summary()

import numpy
test_steps_per_epoch = numpy.math.ceil(test_set.samples / test_set.batch_size)
predictions = classifier.predict(test_set, steps=test_steps_per_epoch)
print(predictions)
y_int = numpy.zeros_like(predictions)
y_int[predictions > 0.5] = 1

true_classes = test_set.classes
class_labels = list(test_set.class_indices)
import sklearn.metrics as metrics
report = metrics.classification_report(true_classes, y_int, zero_division=0)
print(report)

cnf_mtx = confusion_matrix(true_classes, y_int)
print(cnf_mtx)


print(test_set.classes)
print(true_classes)
print(y_int)
