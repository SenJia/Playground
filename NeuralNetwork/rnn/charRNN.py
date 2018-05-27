
# This code is based on the character-level RNN model written by Andrej Karpathy.
# Minor changes are added just for learning and tweaking the algorihtm of RNN, e.g.
# The optimization method is traditional SGD with step learning rate, multiplying 0.1 every 5 epochs, no momentum used.
# This charRNN only looks one step back, the derivative of Whx and Whh are based on x(t) and h(t-1).
#
# A more general implementaion of vanilla RNN will be updated soon, in which the Back Propagation Through Time will trace back to the very beginning.
# Author: Sen Jia

import sys
import numpy as np

class TxtReader(object):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self._data = f.read() 
        self._chars = list(set(self._data))
        self._data_size, self._vocab_size = len(self._data), len(self._chars)
        print ('data has %d characters, %d unique.' % (self._data_size, self._vocab_size))
        self._char_to_ix = { ch:i for i,ch in enumerate(self._chars) }
        self._ix_to_char = { i:ch for i,ch in enumerate(self._chars) }
    
    @property
    def data(self):
        return list(self._data)
    
    @property
    def chars(self):
        return self._chars

    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def data_size(self):
        return self._data_size

    def char2index(self, char):
        return self._char_to_ix[char]
    
    def index2char(self, index):
        return self._ix_to_char[index]

class RNNModel(object):
    # hyperparameters
    def __init__(self, hidden, length, TxtReader, lr=1e-1, epoch=20):
        self._txt_reader = TxtReader
        self._hidden_size = hidden # size of hidden layer of neurons
        self._seq_length = length # number of steps to unroll the RNN for
        self._learning_rate = lr 

        self._pointer = 0
        self._iter_index = 0
        self._epoch_index = 0
        self._num_epoch = epoch

        # model parameters
        self._Wxh = np.random.uniform(-np.sqrt(1/self._txt_reader.vocab_size), np.sqrt(1/self._txt_reader.vocab_size), (self._hidden_size, self._txt_reader.vocab_size)) # input to hidden
        self._Whh = np.random.uniform(-np.sqrt(1/self._hidden_size), np.sqrt(1/self._hidden_size), (self._hidden_size, self._hidden_size)) # hidden to hidden
        self._Why = np.random.uniform(-np.sqrt(1/self._txt_reader.vocab_size), np.sqrt(1/self._txt_reader.vocab_size), (self._txt_reader.vocab_size, self._hidden_size)) # hidden to output
        self._bh = np.zeros((self._hidden_size, 1)) # hidden bias
        self._by = np.zeros((self._txt_reader.vocab_size, 1)) # output bias

    def train(self):
        smooth_loss = -np.log(1.0/self._txt_reader.vocab_size) * self._seq_length # loss at iteration 0
        epoch_loss = 0
        hprev = np.zeros((self._hidden_size,1)) 
        while self._epoch_index < self._num_epoch:
            inputs = [self._txt_reader.char2index(ch) for ch in self._txt_reader.data[self._pointer:self._pointer+self._seq_length]]
            targets = [self._txt_reader.char2index(ch) for ch in self._txt_reader.data[self._pointer+1:self._pointer+self._seq_length+1]]
            if len(inputs) < self._seq_length:
                self._pointer = 0
                self._epoch_index += 1
                epoch_loss = 0
                hprev = np.zeros((self._hidden_size,1)) # reset RNN memory[201~
                inputs = [self._txt_reader.char2index(ch) for ch in self._txt_reader.data[self._pointer:self._pointer+self._seq_length]]
                targets = [self._txt_reader.char2index(ch) for ch in self._txt_reader.data[self._pointer+1:self._pointer+self._seq_length+1]]


            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            epoch_loss += loss/self._seq_length
      
            for param, dparam in zip([self._Wxh, self._Whh, self._Why, self._bh, self._by], 
                                          [dWxh, dWhh, dWhy, dbh, dby]):
                param -= self._learning_rate * dparam

            if self._pointer == 0:
                print ("Epoch", self._epoch_index, "Loss:", epoch_loss)
                if self._epoch_index % 5 == 0:
                    self._learning_rate *= 0.1

            self._pointer += self._seq_length # shift the pointer to the next batch.

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        self._hidden is (hidden_size x 1) array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self._txt_reader.vocab_size,1)) # One-hot, encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self._Wxh, xs[t]) + np.dot(self._Whh, hs[t-1]) + self._bh) # compute chidden state
            ys[t] = np.dot(self._Why, hs[t]) + self._by # logits 
            ys[t] -= ys[t].max() # for numerical stability
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self._Wxh), np.zeros_like(self._Whh), np.zeros_like(self._Why)
        dbh, dby = np.zeros_like(self._bh), np.zeros_like(self._by)

        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self._Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self._Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def main(file_path):
    txt_reader = TxtReader(file_path)
    rnn = RNNModel(hidden=100, length=25, TxtReader=txt_reader)
    rnn.train()

if __name__ == "__main__" : main(sys.argv[1])
