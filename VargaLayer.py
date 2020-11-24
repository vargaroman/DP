from keras import backend as K
from keras.layers import Layer

class VargaLayer(Layer):
    def __init__(self, num_outputs, name = 'VargaLayer', **kwargs):
        super(VargaLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.num_outputs),
                                      initializer='normal', trainable=True)

        super(VargaLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        flipped = K.reverse(inputs,axes=2)
        return flipped

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(VargaLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
