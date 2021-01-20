from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math

#https://stackoverflow.com/questions/50724762/create-a-transformation-matrix-out-of-scalar-angle-tensors

class VargaRotationLayer(Layer):
    def __init__(self, num_outputs, name = 'VargaRotationLayer', **kwargs):
        super(VargaRotationLayer, self).__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=[1], dtype="float32"),
            trainable=True,
        )
        self.add
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(VargaRotationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        rotated = tfa.image.rotate(inputs, (self.w*1000)*math.pi/180, interpolation='BILINEAR')
        rotated = tf.reshape(rotated, shape=[-1,224,224,1])
        return rotated



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(VargaRotationLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
