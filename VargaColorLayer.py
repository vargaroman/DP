from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import tensorflow_addons as tfa

class VargaColorLayer(Layer):
    def __init__(self, num_outputs, name='VargaColorLayer', factor=0.5, **kwargs):
        super(VargaColorLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.factor = factor

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[1], initializer="random_normal", trainable=True
        )
        super(VargaColorLayer, self).build(input_shape)

    def color(self, x: tf.Tensor) -> tf.Tensor:
        contrast = tf.image.adjust_contrast(x, 10)
        return contrast

    def call(self, inputs, **kwargs):
        return self.color(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(VargaColorLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))


