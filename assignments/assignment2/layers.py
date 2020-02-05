import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    loss = reg_strength * np.sum(np.sum(np.square(W)))
    #print(W, '\n\n', l2_reg_loss)
    grad = 2*reg_strength * W
    
    return loss, grad

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X.copy()
        x_out = X.copy()
        x_out[x_out<0] = 0
        #raise Exception("Not implemented!")
        return x_out

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input

        """
        dout = d_out.copy()
        x = self.X.copy()
        d_result = dout * (x >= 0 )
        #print ("dX=", d_X)
        #d_result = np.dot(d_out.T, d_X)
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("lemented!")
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.learning_rate = 0.001

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        L = np.add(np.dot(X,self.W.value),self.B.value)
        #print("L = ", L.shape)
        #raise Exception("Not implemented!")
        return L
  
    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad = np.dot(self.X.T,d_out)
        self.B.grad = np.array([np.sum(d_out, axis=0)])

        dX = np.dot(d_out, self.W.value.T)
        
        #learning_rate = 0.001        
        #self.W.value -= (learning_rate * self.W.grad)
        #self.B.value -= (learning_rate * self.B.grad)

        #raise Exception("Not implemented!")
        #d_input = dX
        return dX #d_input
    
    def clearGrad(self):
        self.W.grad = 0
        self.B.grad = 0

    def params(self):
        return {'W': self.W, 'B': self.B}
