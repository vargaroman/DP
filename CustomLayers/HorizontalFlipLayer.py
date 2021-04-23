from keras.layers import Layer
import keras.backend as K

class HorizontalFlipLayer(Layer):
    def __init__(self, num_outputs, name='HorizontalFlipLayer', **kwargs):
        super(HorizontalFlipLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        super(HorizontalFlipLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.reverse(inputs, axes=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(HorizontalFlipLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
