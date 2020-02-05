import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    l = len(predictions.shape)
    p = predictions.copy()
    if (l == 1):
        p -= np.max(p)
        e = np.exp(p)
        znam = e.sum()
        probs = e / znam
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        return probs
    elif (l > 1):
        m = np.max(p,axis=1)
        m = m[:,None]
        p -= m
        e = np.exp(p)
        znam = np.sum(e, axis=1)
        znam = znam[:,None]
        probs = e / znam
        return np.asarray(probs)
    raise Exception("Something wrong with argument")
        
        


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    l = len(probs.shape)
    norm = 1
    gt_bool = np.zeros(probs.shape, np.float64)
    
    if (l == 1):
        norm = 1 
        #gt_bool = np.zeros(probs.shape, np.float64)
        gt_bool[target_index] = 1
        #probs_log = np.log(probs)
        #return -1 * np.sum(gt_bool * probs_log)
    elif (l>1):
        norm = len(probs)
        ti = target_index.copy()
        ti = ti[:,None]
        for i in range(0, len(gt_bool)):
            gt_bool[i][ti[i]] = 1
    else:
        raise Exception("Something wrong with argument")


    probs_log = np.log(probs)
    return (-1 * np.sum(gt_bool * probs_log)) / norm




        


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    l = len(predictions.shape)
    if l < 1 :
        raise Exception("Something wrong with argument")

    soft_max = np.asarray(softmax(predictions))
    loss = cross_entropy_loss(soft_max, target_index)

    if l==1:
        soft_max[target_index] -= 1
        dprediction = soft_max
        print(dprediction)
        return loss, dprediction
    else:
        #m = np.max(soft_max, axis=1)
        #print (len(predictions))
        #print('\n', 'target index', target_index, '\n', soft_max)
        soft_max[np.arange(len(predictions)), target_index] -= 1
        dprediction = soft_max / len(predictions)
        #loss /= len(predictions) 
        #print(loss, dprediction, '\n')
        return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    loss = reg_strength * np.sum(np.sum(np.square(W)))
    #print(W, '\n\n', l2_reg_loss)
    grad = 2*reg_strength * W
    

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #print (predictions, predictions.shape)
    
    #prob = softmax(predictions)
    loss, dscores = softmax_with_cross_entropy(predictions,target_index)
    #print (dW, "\n shape dW", dW.shape, W.shape )
    dW = np.dot(X.T,dscores)

    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # TODO implement generating batches from indices
            loss = 0
            for i in range(len(batches_indices)):
                batch = X[batches_indices[i]]
                # Compute loss and gradients
                s_loss, gW = linear_softmax(batch,self.W,y[batches_indices[i]])
                l2_loss, l2_gW = l2_regularization(self.W,reg)
                # Apply gradient to weights using learning rate
                # Don't forget to add both cross-entropy loss
                # and regularization!
                #if (loss < (s_loss + l2_loss) and loss != 0):
                #      raise Exception("Loss Error", loss, l2_loss + s_loss)
                loss = s_loss + l2_loss
                self.W -= learning_rate*(l2_gW + gW) 
                #raise Exception("Not implemented!")

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
      
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        pred = softmax(predictions)
        y_pred = np.argmax(pred,axis=1)
        
        #print (pred.shape, y_pred.shape)

        #raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
