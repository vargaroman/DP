from keras.layers import Layer
import tensorflow as tf
import tensorflow_addons as tfa
import math
import random

class RotationLayer(Layer):
    def __init__(self, num_outputs,lower_bound = None, upper_bound = None, name = 'RotationLayer', **kwargs):
        super(RotationLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def build(self, input_shape):
        super(RotationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.lower_bound is not None and self.upper_bound is not None:
            angle = (random.randint(self.lower_bound,self.upper_bound))*math.pi/180
            rotated = tfa.image.rotate(inputs, angles=round(angle),fill_mode='wrap')
        elif self.lower_bound is not None:
            angle = (random.randint(self.lower_bound,360))*math.pi/180
            rotated = tfa.image.rotate(inputs, angles=round(angle),fill_mode='wrap')
        elif self.upper_bound is not None:
            angle = (random.randint(-360,self.upper_bound))*math.pi/180
            rotated = tfa.image.rotate(inputs, angles=round(angle),fill_mode='wrap')
        else:
            angle = (random.randint(-360,360))*math.pi/180
            rotated = tfa.image.rotate(inputs, angles=round(angle),fill_mode='wrap')
        rotated = tf.reshape(rotated, shape=[-1, 64, 64, 1])
        return rotated

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(RotationLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
