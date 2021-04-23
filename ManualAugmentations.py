import albumentations as A
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
transform = A.Compose(
    [A.CLAHE(),
     # A.Flip(),
     A.RandomBrightness(limit=(-0.2,0.2)),
     # A.Rotate((-10,10))
    ])

def visualize(image):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(np.clip(image,0,1))
    plt.show()
i = 0

for image_name in os.listdir('datasetcovid/non-COVID/') :
    print(image_name)
    image = cv2.imread('datasetcovid/non-COVID/'+image_name)
    augmented_image = transform(image=image)
    cv2.imwrite('manually_augmented/non-COVID/'+image_name, augmented_image['image'])
    #visualize(augmented_image['image'])
for image_name in os.listdir('datasetcovid/COVID/') :
    image = cv2.imread('datasetcovid/COVID/'+image_name)
    augmented_image = transform(image=image)
    cv2.imwrite('manually_augmented/COVID/'+image_name, augmented_image['image'])
    #visualize(augmented_image['image'])

for image_name in os.listdir('resizedMRI224x224/yes/') :
    print(image_name)
    image = cv2.imread('resizedMRI224x224/yes/'+image_name)
    augmented_image = transform(image=image)
    cv2.imwrite('MRI_dataset/yes/'+image_name, augmented_image['image'])
    #visualize(augmented_image['image'])
for image_name in os.listdir('resizedMRI224x224/no/') :
    image = cv2.imread('resizedMRI224x224/no/'+image_name)
    augmented_image = transform(image=image)
    cv2.imwrite('MRI_dataset/no/'+image_name, augmented_image['image'])
    #visualize(augmented_image['image'])
