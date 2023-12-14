import numpy as np
import random
import dataset

'''
Just like everything written in Jupyter notebooks, this code is extremely cursed.
However, unlike a lot of code that is extremely cursed, it's not actually my fault.
I have done my best to neatly order everything in a way that makes sense, but I'm
bound to have missed something.
'''

class RNN:
  # A Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
    self.Wxh = np.random.randn(hidden_size, input_size) / 1000
    self.Why = np.random.randn(output_size, hidden_size) / 1000

    # Biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))


  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one-hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h

    # Compute the output
    y = self.Why @ h + self.by

    return y, h

  def backprop(self, d_y, learn_rate=2e-2):
    '''
    Perform a backward pass of the RNN.
    - d_y (dL/dy) has shape (output_size, 1).
    - learn_rate is a float.
    '''
    n = len(self.last_inputs)

    # Calculate dL/dWhy and dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    # Calculate dL/dh for the last h.
    d_h = self.Why.T @ d_y

    # Backpropagate through time.
    for t in reversed(range(n)):
      # An intermediate value: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp @ self.last_hs[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Update weights and biases using gradient descent.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by

'''
Below is all the stuff that's not part of the RNN. They should probably go in main() or
something, right? Except for the stuff that should go in the Trainer, of course. We
need one of those, right? Hmm.
'''

class Trainer:
    def __init__(self, train_data, test_data):
        # Create the vocabulary.
        train_vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
        self.train_size = len(train_vocab)
        # Assign indices to each word.
        self.train_to_idx = { w: i for i, w in enumerate(train_vocab) }
        self.idx_to_train = { i: w for i, w in enumerate(train_vocab) }

        # Create the vocabulary.
        test_vocab = list(set([w for text in test_data.keys() for w in text.split(' ')]))
        self.test_size = len(test_vocab)
        # Assign indices to each word.
        self.test_to_idx = { w: i for i, w in enumerate(test_vocab) }
        self.idx_to_test = { i: w for i, w in enumerate(test_vocab) }

        self.model = RNN(len(self.train_vocab), 2)

    def train(self):
        for epoch in range(1000):
            train_loss, train_acc = self.processData(dataset.train_data)

            if epoch % 100 == 99:
                print('--- Epoch %d' % (epoch + 1))
                print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

                test_loss, test_acc = self.processData(dataset.test_data, backprop=False)
                print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
    
    def processData(self, data, backprop=True):
        '''
        Returns the RNN's loss and accuracy for the given data.
        - data is a dictionary mapping text to True or False.
        - backprop determines if the backward phase should be run.
        '''
        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        for x, y in items:
            inputs = self.createInputs(x)
            target = int(y)

            # Forward
            out, _ = self.model.forward(inputs)
            probs = self.softmax(out)

            # Calculate loss / accuracy
            loss -= np.log(probs[target])
            num_correct += int(np.argmax(probs) == target)

            if backprop:
                # Build dL/dy
                d_L_d_y = probs
                d_L_d_y[target] -= 1

                # Backward
                self.model.backprop(d_L_d_y)

        return loss / len(data), num_correct / len(data)
    
    def createInputs(self, text, vocab_size, word_to_idx):
        '''
        Returns an array of one-hot vectors representing the words
        in the input text string.
        - text is a string
        - Each one-hot vector has shape (vocab_size, 1)
        '''
        inputs = []
        for w in text.split(' '):
            v = np.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            inputs.append(v)
        return inputs

    def softmax(xs):
        # Applies the Softmax Function to the input array.
        return np.exp(xs) / sum(np.exp(xs))
       

'''
i have no idea what to do with this
'''

def main():
  #Initialize our RNN!
    trainer = Trainer(dataset.train_data, dataset.test_data)

    # uhhhh
    # TODO

    # inputs = createInputs('i am very good')
    # out, h = rnn.forward(inputs)
    # probs = softmax(out)










