"""
This is a softmax python implementation within caffe. This code is based on 
"https://github.com/BVLC/caffe/issues/4023" by biprajiman
with modification of the numeric stability practical issues (cs231n)
Author: Sen Jia
"""
import sys
sys.path.append("/xxx/xxx/caffe/python/")
import caffe
import numpy as np


class SoftmaxWithLoss(caffe.Layer):
    """
    Compute the output of softmax and its loss 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
	# check input dimensions match
	if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("Inputs must have the same dimension.")
    	# difference is shape of inputs
    	self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    	# loss output is scalar
    	top[0].reshape(1)

    def forward(self, bottom, top):
	scores = bottom[0].data
	scores -= np.max(scores)
    	exp_scores = np.exp(scores)
    	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
	label_probs=probs[range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16)]
	correct_logprobs = -np.log(label_probs)
	data_loss = np.sum(correct_logprobs)/bottom[0].num
	self.diff[...] = probs
    	top[0].data[...] = data_loss
	

    def backward(self, top, propagate_down, bottom):
	delta = self.diff
	for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16)] -= 1
	    bottom[i].diff[...] = delta/bottom[0].num
