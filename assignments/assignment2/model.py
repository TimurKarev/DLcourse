import numpy as np
import linear_classifer as lc

from layers import FullyConnectedLayer, ReLULayer, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        
        # TODO Create necessary layers
        self.fully_layers = []
        self.relu_layers = []
        inp = n_input
        out = n_output
        for l in range(hidden_layer_size):
            #print ("inp = %, out = % ", inp, out)
            self.fully_layers.append(FullyConnectedLayer(inp, out))
            #inp, out = out, inp
            inp = 10
            self.relu_layers.append(ReLULayer())

        self.fully_layers.append(FullyConnectedLayer(inp, n_output))
        #raise Exception("Not implemented!")
        """
        self.reg = reg
        self.input_fc = FullyConnectedLayer(n_input,hidden_layer_size)
        self.input_re = ReLULayer()
        self.hidden_fc = FullyConnectedLayer(hidden_layer_size,n_output)


    def forward_pass(self, X):

        self.input_fc.clearGrad()
        self.hidden_fc.clearGrad()
        #raise Exception("Not implemented!")

        x = X.copy()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        i_fc = self.input_fc.forward(x)
        i_re = self.input_re.forward(i_fc)

        h_fc = self.hidden_fc.forward(i_re)

        return h_fc
 
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        h_fc = self.forward_pass(X)
        simp_loss, dfinal_mat = lc.softmax_with_cross_entropy(h_fc, y)

        d_hf = self.hidden_fc.backward(dfinal_mat)
        d_ir = self.input_re.backward(d_hf)
        d_if = self.input_fc.backward(d_ir)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        param = self.params()
        i_loss, iL2_Wgrad = l2_regularization(param['iW'].value, self.reg)
        h_loss, hL2_Wgrad = l2_regularization(param['hW'].value, self.reg)
        loss = simp_loss + i_loss + h_loss

        param['iW'].grad += iL2_Wgrad
        param['hW'].grad += hL2_Wgrad
        #raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        y_pred = np.zeros(X.shape[0], np.int)
        predictions = self.forward_pass(X)
        pred = lc.softmax(predictions)
        #print (pred.shape)
        y_pred = np.argmax(pred, axis=1)

        #raise Exception("Not implemented!")
        return y_pred

    def params(self):
        result = {}
        # TODO Implement aggregating all of the params
        d = self.input_fc.params()
        result['iW'] = d['W']
        result['iB'] = d['B']
        d = self.hidden_fc.params()
        result['hW'] = d['W']
        result['hB'] = d['B']

        #raise Exception("Not implemented!")

        return result
