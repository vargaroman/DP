from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from keras.models import load_model
from skimage.color import rgb2gray
import VargaLayer
from VargaLayer import VargaLayer
from VargaRotationLayer import VargaRotationLayer

# load the model
model = load_model('weights.best.hdf5', custom_objects={'VargaLayer': VargaLayer, 'VargaRotationLayer':VargaRotationLayer})
# redefine model to output right after the first hidden layer
ixs = [0, 1, 3, 4, 6, 8, 10]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img("resized/test/yes/Y2.jpg", target_size=(224, 224))
print(img)
# convert the image to an array
img = img_to_array(img)
img = rgb2gray(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# img = expand_dims(img, axis=3)
# prepare the image (e.g. scale pixel values for the cnn)
# img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
    # plot all 64 maps
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, 0], cmap='gray')
            ix += 1
    pyplot.show()
