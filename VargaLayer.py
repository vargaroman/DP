import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class VargaLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(VargaLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                        shape=(input_shape[1], self.output_dim),
                        initializer='normal', trainable=True)
        super(VargaLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.dot(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
