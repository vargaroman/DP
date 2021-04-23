from keras.layers import Layer
import tensorflow as tf
import numpy as np

class BrightnessLayer(Layer):
    def __init__(self, num_outputs, name='BrightnessLayer',lower_bound = -1, upper_bound = 1, **kwargs):
        super(BrightnessLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def build(self, input_shape):
        super(BrightnessLayer, self).build(input_shape)

    def call(self, inputs , **kwargs):
        brightness = np.random.uniform(self.lower_bound, self.upper_bound)
        return tf.image.adjust_brightness(inputs, brightness)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

    def get_config(self):
        base = super(BrightnessLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))


