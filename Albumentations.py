from ImageDataAugmentor.image_data_augmentor import *
import albumentations
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy


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


import tensorflow as tf
from ImageDataAugmentor.image_data_augmentor import *
import os
import imutils
import shutil


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

if not os.listdir().__contains__('IDGresizedMRI64x64'):
    os.mkdir('IDGresizedMRI64x64')
    os.mkdir('IDGresizedMRI64x64/test')
    os.mkdir('IDGresizedMRI64x64/train')
    os.mkdir('IDGresizedMRI64x64/test/yes')
    os.mkdir('IDGresizedMRI64x64/test/no')
    os.mkdir('IDGresizedMRI64x64/train/yes')
    os.mkdir('IDGresizedMRI64x64/train/no')
else:
    shutil.rmtree('IDGresizedMRI64x64')
    os.mkdir('IDGresizedMRI64x64')
    os.mkdir('IDGresizedMRI64x64/test')
    os.mkdir('IDGresizedMRI64x64/train')
    os.mkdir('IDGresizedMRI64x64/test/yes')
    os.mkdir('IDGresizedMRI64x64/test/no')
    os.mkdir('IDGresizedMRI64x64/train/yes')
    os.mkdir('IDGresizedMRI64x64/train/no')

IMG_SIZE = 64
croppedImagesDir = os.listdir('cropped')
i = 0

for croppedImageName in croppedImagesDir:
    i += 1
    # image = cv2.imread("cropped/" + croppedImageName)
    image = cv2.imread("cropped/" + croppedImageName, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(64, 64)
    if i % 4 == 0:
        if croppedImageName.upper().__contains__("Y"):
            cv2.imwrite("IDGresizedMRI64x64/test/yes/" + croppedImageName, image)
        else:
            cv2.imwrite("IDGresizedMRI64x64/test/no/" + croppedImageName, image)
    else:
        if croppedImageName.upper().__contains__("Y"):
            cv2.imwrite("IDGresizedMRI64x64/train/yes/" + croppedImageName, image)
        else:
            cv2.imwrite("IDGresizedMRI64x64/train/no/" + croppedImageName, image)

if os.listdir().__contains__('review'):
    shutil.rmtree('review', ignore_errors=True)


AUGMENTATIONS = albumentations.Compose([
    albumentations.HorizontalFlip(p=1),
    albumentations.Rotate((-10, 10), p=1, border_mode=cv2.BORDER_WRAP),
    albumentations.RandomBrightnessContrast((0.2,-0.2),(0,0), p=1)
])
image_data_gen = ImageDataAugmentor(augment = AUGMENTATIONS, seed=42)
train_image_gen = image_data_gen.flow_from_directory(directory='IDGresizedMRI64x64/train',target_size=(64,64),class_mode='binary',color_mode='grayscale',batch_size=8)
test_image_gen = image_data_gen.flow_from_directory(directory='IDGresizedMRI64x64/test',target_size=(64,64),class_mode='binary',color_mode='grayscale')

model = Sequential()
model.add(Conv2D(48, 5,input_shape=(64, 64, 1), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, strides = 2))
model.add(Conv2D(64, 5, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, strides = (2,2)))
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1024, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'], run_eagerly=True)

saved_model = "weights.best.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10
)

history = model.fit(
    train_image_gen,
    steps_per_epoch=int(train_image_gen.samples / train_image_gen.batch_size),
    epochs=50,
    validation_data=test_image_gen,
    validation_steps=int(test_image_gen.samples / test_image_gen.batch_size),
    callbacks=[es, checkpoint]
)

best_model = load_model('weights.best.hdf5')
best_model.summary()

test_steps_per_epoch = numpy.math.ceil(test_image_gen.samples / test_image_gen.batch_size)
predictions = best_model.predict(test_image_gen, steps=test_steps_per_epoch)
y_int = numpy.zeros_like(predictions)
y_int[predictions > 0.5] = 1

# trainingLoss(history)

true_classes = test_image_gen.classes
class_labels = list(test_image_gen.class_indices)
report = classification_report(true_classes, y_int, zero_division=0)
print(report)

cnf_mtx = confusion_matrix(true_classes, y_int)
print(cnf_mtx)


################################################################

image_data_gen = ImageDataAugmentor(augment = AUGMENTATIONS)
train_image_gen_covid = image_data_gen.flow_from_directory(directory='resizedcovid224x224/train',target_size=(64,64),class_mode='binary',color_mode='grayscale',batch_size=8)
test_image_gen_covid = image_data_gen.flow_from_directory(directory='resizedcovid224x224/test',target_size=(64,64),class_mode='binary',color_mode='grayscale')

model2 = Sequential()
model2.add(Conv2D(48, 5,input_shape=(64, 64, 1), padding = 'same', activation = 'relu'))
model2.add(MaxPool2D(pool_size = 3, strides = 2))
model2.add(Conv2D(64, 5, padding = 'same', activation = 'relu'))
model2.add(MaxPool2D(pool_size = 3, strides = (2,2)))
model2.add(Conv2D(32, 3, padding = 'same', activation = 'relu'))
model2.add(Dropout(0.1))
model2.add(Dense(1024, activation = 'relu'))
model2.add(Flatten())
model2.add(Dense(1, activation = 'sigmoid'))


saved_model2 = "weights.best2.hdf5"
checkpoint2 = ModelCheckpoint(saved_model2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'], run_eagerly=True)

history = model2.fit(
    train_image_gen_covid,
    steps_per_epoch=int(train_image_gen_covid.samples / train_image_gen_covid.batch_size),
    epochs=50,
    validation_data=test_image_gen_covid,
    validation_steps=int(test_image_gen_covid.samples / test_image_gen_covid.batch_size),
    callbacks=[es, checkpoint2]
)

best_model2 = load_model('weights.best2.hdf5')
best_model2.summary()

test_steps_per_epoch = numpy.math.ceil(test_image_gen_covid.samples / test_image_gen_covid.batch_size)
predictions = best_model2.predict(test_image_gen_covid, steps=test_steps_per_epoch)
y_int = numpy.zeros_like(predictions)
y_int[predictions > 0.5] = 1

# trainingLoss(history)

true_classes = test_image_gen_covid.classes
class_labels = list(test_image_gen_covid.class_indices)
report = classification_report(true_classes, y_int, zero_division=0)
print(report)

cnf_mtx = confusion_matrix(true_classes, y_int)
print(cnf_mtx)
