# A cell wrapper for tensorflow RNN cell on multiple GPUS.
#
#Author: Sen Jia
#

class CELLWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, cell, device):
        self._cell = cell
        self._device = device
     
    @property
    def state_size(self):
        return self._cell.state_size
     
    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, scope)
