from keras import backend as K
from keras.layers import Layer

class SandPLayer(Layer):
    def __init__(self, num_outputs, name = 'SandP', noise_ratio = 0.01, sandp_ratio = 0.5, **kwargs):
        super(SandPLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.noise_ratio = noise_ratio
        self.sandp_ratio = sandp_ratio
    def build(self, input_shape):
        super(SandPLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        def noised():
            shp = K.shape(inputs)[1:]
            mask_select = K.random_binomial(shape=shp, p=self.noise_ratio)
            mask_noise = K.random_binomial(shape=shp, p=self.sandp_ratio)
            out = inputs * (1 - mask_select) + mask_noise * mask_select
            return out
        return K.in_train_phase(noised(), inputs, training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base = super(SandPLayer, self).get_config()
        base['num_outputs'] = self.num_outputs
        return dict(list(base.items()))
