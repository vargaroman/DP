from keras import backend as K
from keras.layers import Layer
import tensorflow

class VargaRotationLayer(Layer):
    def __init__(self, num_outputs, name = 'VargaLayer', **kwargs):
        super(VargaRotationLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.num_outputs),
                                      initializer='normal', trainable=True)

        super(VargaRotationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        rotated = tensorflow.image.rot90(inputs)
        return rotated

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(VargaRotationLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
