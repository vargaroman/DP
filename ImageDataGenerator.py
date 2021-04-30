from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from CustomLayers.HorizontalFlipLayer import HorizontalFlipLayer
from CustomLayers.VerticalFlipLayer import VerticalFlipLayer
from CustomLayers.RotationLayer import RotationLayer
from CustomLayers.BrightnessLayer import BrightnessLayer
from CustomLayers.SandPLayer import SandPLayer
from keras.models import load_model
import numpy
import os
import shutil
import cv2

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
    plt.ylabel('chyba.jpg')
    plt.xlabel('počet epoch')
    plt.legend(['trénovanie', 'validácia'], loc='upper left')
    plt.show()


if not os.listdir().__contains__('resizedcovid224x224'):
    os.mkdir('resizedcovid224x224')
    os.mkdir('resizedcovid224x224/test')
    os.mkdir('resizedcovid224x224/train')
    os.mkdir('resizedcovid224x224/train/non-COVID')
    os.mkdir('resizedcovid224x224/train/COVID')
    os.mkdir('resizedcovid224x224/test/non-COVID')
    os.mkdir('resizedcovid224x224/test/COVID')
else:
    shutil.rmtree('resizedcovid224x224')
    os.mkdir('resizedcovid224x224')
    os.mkdir('resizedcovid224x224/test')
    os.mkdir('resizedcovid224x224/train')
    os.mkdir('resizedcovid224x224/train/non-COVID')
    os.mkdir('resizedcovid224x224/train/COVID')
    os.mkdir('resizedcovid224x224/test/non-COVID')
    os.mkdir('resizedcovid224x224/test/COVID')

datasetCovidDir = os.listdir('datasetcovid/COVID')
datasetnonCovidDir = os.listdir('datasetcovid/non-COVID')

i = 0
IMG_SIZE = 224

for imageName in datasetCovidDir:
    i+=1
    image = cv2.imread("datasetcovid/COVID/" + imageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(IMG_SIZE, IMG_SIZE)
    if i%3:
        cv2.imwrite("resizedcovid224x224/test/COVID/" + imageName, image)
    else:
        cv2.imwrite("resizedcovid224x224/train/COVID/" + imageName, image)

i=0
for imageName in datasetnonCovidDir:
    i+=1
    image = cv2.imread("datasetcovid/non-COVID/" + imageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(IMG_SIZE, IMG_SIZE)
    if i%3:
        cv2.imwrite("resizedcovid224x224/test/non-COVID/" + imageName, image)
    else:
        cv2.imwrite("resizedcovid224x224/train/non-COVID/" + imageName, image)

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

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('resizedcovid224x224/train',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 color_mode='grayscale',
                                                 class_mode='binary'
                                                 )
print(training_set.samples)

test_set = test_datagen.flow_from_directory('resizedcovid224x224/test',
                                            target_size=(64, 64),
                                            batch_size=4,
                                            color_mode='grayscale',
                                            class_mode='binary'
                                            )

best_model = load_model('weights.best.hdf5', custom_objects={'HorizontalFlipLayer': HorizontalFlipLayer,"VerticalFlipLayer":VerticalFlipLayer, 'RotationLayer':RotationLayer, 'BrightnessLayer':BrightnessLayer, 'SandPLayer': SandPLayer})
best_model.summary()

test_steps_per_epoch = numpy.math.ceil(test_set.samples / test_set.batch_size)
predictions = best_model.predict(test_set, steps=test_steps_per_epoch)

y_int = numpy.zeros_like(predictions)

y_int[predictions > 0.5] = 1


true_classes = test_set.classes
class_labels = list(test_set.class_indices)
report = metrics.classification_report(true_classes, y_int, zero_division=0)
print(report)

cnf_mtx = confusion_matrix(true_classes, y_int)
print('second_confusion')
print(cnf_mtx)
