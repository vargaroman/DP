import cv2
import os
import shutil
import itertools
import imutils
import sklearn.metrics as metrics
import numpy
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
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
    if (imagename.upper().__contains__("JPG") or imagename.upper().__contains__(
            "JPEG") or imagename.upper().__contains__("PNG")):
        i += 1
        image = cv2.imread("dataset/" + imagename)
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
        crop_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        cv2.imwrite("cropped/" + imagename, crop_img)

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


def trainingLoss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('')
    plt.ylabel('celková úspešnosť')
    plt.xlabel('počet epoch')
    plt.legend(['trénovanie', 'validácia'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('')
    plt.ylabel('chyba')
    plt.xlabel('počet epoch')
    plt.legend(['trénovanie', 'validácia'], loc='upper left')
    plt.show()


for croppedImageName in croppedImagesDir:
    i += 1
    # image = cv2.imread("cropped/" + croppedImageName)
    image = cv2.imread("cropped/" + croppedImageName, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(224, 224)
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

if os.listdir().__contains__('review'):
    shutil.rmtree('review', ignore_errors=True)
from VargaLayer import VargaLayer
from VargaRotationLayer import VargaRotationLayer
from VargaColorLayer import VargaColorLayer
# precprocessing
classifier = Sequential()
layer1 = VargaLayer(64,input_shape=(224,224,1))
layer2 = VargaRotationLayer(128)
# layer3 = VargaColorLayer(64)
classifier.add(layer1)
classifier.add(layer2)
# classifier.add(layer3)
weights = layer1.get_weights()
weights2 = layer2.get_weights()
# weights3 = layer3.get_weights()
print(weights)
print(weights2)
# print(weights3)

classifier.add(tflayers.Conv2D(16, (2,2)))
classifier.add(tflayers.MaxPooling2D(16,(4,4)))
# classifier.add(tflayers.Conv2D(8, (2,2)))
# classifier.add(tflayers.MaxPooling2D(16,(4,4)))
classifier.add(tflayers.Flatten())
classifier.add(tflayers.Dropout(0.3))
classifier.add(tflayers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

train_datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1. / 255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('resized/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='binary'
                                                 )
print(training_set.samples)

test_set = test_datagen.flow_from_directory('resized/test',
                                            target_size=(224, 224),
                                            batch_size=16,
                                            color_mode='grayscale',
                                            class_mode='binary'
                                            )
print(test_set.samples)
from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=10
)
history = classifier.fit(
    training_set,
    steps_per_epoch=int(training_set.samples / training_set.batch_size),
    epochs=10,
    validation_data=test_set,
    validation_steps=int(test_set.samples / test_set.batch_size),
    callbacks=[es, checkpoint]
)

best_model = load_model('weights.best.hdf5', custom_objects={'VargaLayer': VargaLayer, 'VargaRotationLayer':VargaRotationLayer})
best_model.summary()
print(best_model.layers[0].get_weights())
print(best_model.layers[1].get_weights())

test_steps_per_epoch = numpy.math.ceil(test_set.samples / test_set.batch_size)
predictions = best_model.predict(test_set, steps=test_steps_per_epoch)
y_int = numpy.zeros_like(predictions)
y_int[predictions > 0.5] = 1

trainingLoss(history)

true_classes = test_set.classes
class_labels = list(test_set.class_indices)
report = metrics.classification_report(true_classes, y_int, zero_division=0)
print(report)

cnf_mtx = confusion_matrix(true_classes, y_int)
print(cnf_mtx)
