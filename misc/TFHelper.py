# A helper class file for TensorFlow to build a neural network architecture.
# Common functions are included as a wrapper function, e.g. conv() for 2d convolution,
# relu() for relu activation.
#
# To initialise a neuran network, following arguments are needed:
# 1. an input node,[batch, image_size, image_size, channels] 
# 2. number of classes
# 3. SEED (optional, for random number generation)
#   
# All copyrights are reserved.
# Author: Sen Jia
#
import abc

import tensorflow as tf


class TFHelper(abc.ABC):
  """TF helper class."""

  def __init__(self, images, num_classes, seed=None):
    """class constructor.
    Args:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      num_clases: number of classes.
      seed: used for random number generation.
    """
    self._images = images
    self._num_classes = num_classes 
    self._initialized = False
    self._seed = seed 

  
  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self._build_model()
    self._initialized = True

  def _build_model(self):
    self.forward(self._images)

  @abstractmethod
  def forward(self,x,train=False):
      """ 
      Implement this method to define network architecture.
      Argument:
          x - input data. [batch, size, size ,channels]
          train - boolean variable to determine if this is a training step, may involve with dropout.
      Return:
          logits - forward result.
      """
      return

  @staticmethod
  def _stride_arr(stride):
      """Map a stride scalar to the stride array for tf.nn.conv2d."""
      return [1, stride, stride, 1]

  @staticmethod
  def _dtype():
      return tf.float32

  @staticmethod
  def _conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
           TFHelper._dtype(),
           initializer=tf.contrib.layers.xavier_initializer()
           )
      bias = tf.get_variable("bias",shape=[out_filters],initializer=tf.constant_initializer(1))
      out = tf.nn.conv2d(x, kernel, strides, padding='SAME')
      return tf.nn.bias_add(out,bias) 
  
  @staticmethod
  def _fully_connected(x, out_dim):
    """FullyConnected layer for final output."""
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  @staticmethod
  def _relu(x):
    """Relu."""
    return tf.nn.relu(x)

  @staticmethod
  def _max_pool(x,ksize,stride):
      """ max pooling layer """
      return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding='SAME')
