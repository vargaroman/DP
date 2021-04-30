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

if not os.listdir().__contains__('resizedMRI64x64'):
    os.mkdir('resizedMRI64x64')
    os.mkdir('resizedMRI64x64/no')
    os.mkdir('resizedMRI64x64/yes')
else:
    shutil.rmtree('resizedMRI64x64')
    os.mkdir('resizedMRI64x64')
    os.mkdir('resizedMRI64x64/yes')
    os.mkdir('resizedMRI64x64/no')





IMG_SIZE = 64
croppedImagesDir = os.listdir('cropped')
i = 0

for croppedImageName in croppedImagesDir:
    image = cv2.imread("cropped/" + croppedImageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(IMG_SIZE, IMG_SIZE)
    if croppedImageName.upper().__contains__("Y"):
        cv2.imwrite("resizedMRI64x64/yes/" + croppedImageName, image)
    else:
        cv2.imwrite("resizedMRI64x64/no/" + croppedImageName, image)

if os.listdir().__contains__('review'):
    shutil.rmtree('review', ignore_errors=True)

IMG_SIZE = 224


if not os.listdir().__contains__('resizedMRI224x224'):
    os.mkdir('resizedMRI224x224')
    os.mkdir('resizedMRI224x224/no')
    os.mkdir('resizedMRI224x224/yes')
else:
    shutil.rmtree('resizedMRI224x224')
    os.mkdir('resizedMRI224x224')
    os.mkdir('resizedMRI224x224/yes')
    os.mkdir('resizedMRI224x224/no')

for croppedImageName in croppedImagesDir:
    image = cv2.imread("cropped/" + croppedImageName, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(IMG_SIZE, IMG_SIZE)
    if croppedImageName.upper().__contains__("Y"):
        cv2.imwrite("resizedMRI224x224/yes/" + croppedImageName, image)
    else:
        cv2.imwrite("resizedMRI224x224/no/" + croppedImageName, image)
