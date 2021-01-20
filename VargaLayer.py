from keras import backend as K
from keras.layers import Layer
import tensorflow as tf


# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/data_augmentation.ipynb#scrollTo=nMxEhIVXmAH0

class VargaLayer(Layer):
    def __init__(self, num_outputs, name='VargaLayer', factor=0.5, **kwargs):
        super(VargaLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.factor = factor

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.num_outputs),
        #                               initializer='normal', trainable=True,
        #                               regularizer=tf.keras.regularizers.l1_l2())
        super(VargaLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        flipped = self.horizontal_flip(x=inputs, p=self.factor)
        return flipped

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(VargaLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))

    def horizontal_flip(self, x, p):
        randomNumber = tf.random.uniform([])
        if randomNumber < p:
            x = K.reverse(x, axes=2)
        else:
            x
        return x
