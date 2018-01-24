# <-*- encoding:utf8 -*->
from keras.engine.topology import Layer
from keras import backend as K, initializations
import numpy as np
import theano.tensor as T

__author__ = 'https://github.com/jacoxu/ASAM'


class Attention(Layer):
    def __init__(self, time_step, spec_dim, embed_dim, mode='dot', init='glorot_uniform',
                 nonlinearity='sigmoid', **kwargs):
        self.time_step = time_step
        self.spec_dim = spec_dim
        self.embed_dim = embed_dim

        self.mode = mode
        self.init = initializations.get(init)
        self.nonlinearity = nonlinearity
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)

        align_hidden = self.embed_dim
        if self.mode == 'align':
            self.W_align = self.add_weight(shape=(self.embed_dim, align_hidden), initializer=self.init,
                                           name='{}_W_align'.format(self.name))
            self.U_align = self.add_weight(shape=(self.embed_dim, align_hidden), initializer=self.init,
                                           name='{}_U_align'.format(self.name))
            self.v_align = self.add_weight(shape=(align_hidden, 1), initializer=self.init,
                                           name='{}_v_align'.format(self.name))
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError('Attention must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # (None(batch), MaxLen(time), spec_dim, embed_dim)
        mix_embed_l = inputs[0]
        # (None(batch), embed_dim)
        spk_embed_l = inputs[1]
        energy = None
        if self.mode == 'dot':
            # (batch, time, spec_dim, embed_dim) batch_dot(3,1) (batch, embed_dim) = (batch, time, spec_dim)
            energy = K.batch_dot(mix_embed_l, spk_embed_l, axes=(3, 1))
        elif self.mode == 'align':
            # (batch, time, spec_dim, embed_dim) dot (embed_dim, align_hidden)
            # -> (batch, time, spec_dim, align_hidden)
            hUa = K.dot(mix_embed_l, self.U_align)
            # (batch, embed_dim) dot (embed_dim, align_hidden)
            # -> (batch, align_hidden)
            sWa = K.dot(spk_embed_l, self.W_align)
            sWa = sWa.dimshuffle(0, 'x', 'x', 1)
            # -> (batch, time, spec_dim, align_hidden)
            tanh_sWahUa = K.tanh(sWa+hUa)
            # -> (batch, time, spec_dim, align_hidden) dot (align_hidden, 1)
            # -> (batch, time, spec_dim, 1)
            energy = K.dot(tanh_sWahUa, self.v_align)
            # -> (batch, time, spec_dim)
            energy = K.reshape(energy, (-1, self.time_step, self.spec_dim))
        else:
            raise ValueError('Unknown merge mode.')

        if self.nonlinearity == 'sigmoid':
            alpha = K.sigmoid(energy)
        elif self.nonlinearity == 'linear':
            alpha = energy
        else:
            raise Exception('Unknown nonlinearity mode for attention:'+self.nonlinearity)
        # (batch, time, spec_dim)
        return alpha

    def get_output_shape_for(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # (batch, time, spec_dim)
        return input_shape[0][:-1]


class RemoveMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(RemoveMasking, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x


class MeanPool(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, time, 'x')
            mask = mask.dimshuffle(0, 1, 'x')
            # to make the masked values in x be equal to zero
            x = x * mask
            x = K.sum(x, axis=1) / (K.sum(mask, axis=1) + np.spacing(1))
        else:
            x = K.mean(x, axis=1)
        return x

    def get_output_shape_for(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]


class SpkLifeLongMemory(Layer):
    def __init__(self, mem_size, vec_dim, unk_spk='NO', **kwargs):
        self.mem_size = mem_size
        self.vec_dim = vec_dim
        self.unk_spk = unk_spk
        self.init = initializations.get('zero')
        super(SpkLifeLongMemory, self).__init__(**kwargs)

    def build(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # Create a life-long memory with zero initialization
        self.life_long_mem = self.add_weight(shape=(self.mem_size, self.vec_dim), initializer=self.init,
                                             name='{}_Memory'.format(self.name), trainable=False)

        super(SpkLifeLongMemory, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError('SpkLifeLongMemory must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # (None(batch), 1), index of speaker
        target_spk_l = inputs[0]
        target_spk_l = K.reshape(target_spk_l, (target_spk_l.shape[0], ))
        if K.dtype(target_spk_l) != 'int32':
            target_spk_l = K.cast(target_spk_l, 'int32')
        # (None(batch), embed_dim)
        spk_vector_l = inputs[1]
        # Start to update life-long memory based on the learned speech vector
        # First do normalization
        spk_vector_eps = K.switch(K.equal(spk_vector_l, 0.), np.spacing(1), spk_vector_l)  # avoid zero
        spk_vector_eps = K.sqrt(K.sum(spk_vector_eps**2, axis=1))
        spk_vector_eps = spk_vector_eps.dimshuffle((0, 'x'))
        spk_vector = T.true_div(spk_vector_l, K.repeat_elements(spk_vector_eps, self.vec_dim, axis=1))
        # Store speech vector into life-long memory according to the speaker identity.
        life_long_mem = T.inc_subtensor(self.life_long_mem[target_spk_l, :], spk_vector)
        # Normalization for memory
        life_long_mem_eps = K.switch(K.equal(life_long_mem, 0.), np.spacing(1), life_long_mem)  # avoid 0
        life_long_mem_eps = K.sqrt(K.sum(life_long_mem_eps**2, axis=1))
        life_long_mem_eps = life_long_mem_eps.dimshuffle((0, 'x'))
        life_long_mem = T.true_div(life_long_mem, K.repeat_elements(life_long_mem_eps, self.vec_dim, axis=1))

        # (None(batch), spk_size, embed_dim)
        return life_long_mem

    def get_output_shape_for(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # (None(batch), spk_size, embed_dim)
        return input_shape[1][0], self.mem_size, input_shape[1][-1]


class SelectSpkMemory(Layer):
    def __init__(self, **kwargs):
        super(SelectSpkMemory, self).__init__(**kwargs)

    def build(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError('SelectSpkMemory must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # (None(batch), 1), speaker identity
        target_spk_l = inputs[0]
        target_spk_l = K.reshape(target_spk_l, (target_spk_l.shape[0], ))
        if K.dtype(target_spk_l) != 'int32':
            target_spk_l = K.cast(target_spk_l, 'int32')
        # (None(batch), spk_size, embed_dim), life-long memory
        life_long_mem = inputs[1]
        # Extract the acoustic feature from memory
        spk_memory = K.gather(life_long_mem, target_spk_l)
        # (None(batch), embed_dim)
        return spk_memory

    def get_output_shape_for(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # (None(batch), spk_size, embed_dim)
        return input_shape[1][0], input_shape[1][-1]


def update_memory(auditory_model, target_spk, spk_memory):
    weights = auditory_model.get_layer('SpkLifeLongMemory').get_weights()
    assert len(weights) == 1
    spk_life_long_memory = weights[0]
    target_spk = target_spk.reshape(target_spk.shape[0])
    spk_life_long_memory[target_spk] = spk_memory
    # Update memory.
    auditory_model.get_layer('SpkLifeLongMemory').set_weights([spk_life_long_memory])


class MaskingGt(Layer):
    """Masks a sequence by using a mask value to skip timesteps.
    """

    def __init__(self, mask_value=0., **kwargs):
        self.supports_masking = True
        self.mask_value = mask_value
        super(MaskingGt, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask=None):
        return K.any(K.greater(x, self.mask_value), axis=-1)

    def call(self, x, mask=None):
        boolean_mask = K.any(K.greater(x, self.mask_value),
                             axis=-1, keepdims=True)
        return x * K.cast(boolean_mask, K.floatx())

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(MaskingGt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
