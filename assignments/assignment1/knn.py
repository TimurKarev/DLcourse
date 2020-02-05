import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=1):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                #dists[i_test,i_train] = np.linalg.norm(self.train_X[i_train] - X[i_test]) 
                dists[i_test,i_train] = np.linalg.norm(X[i_test] - self.train_X[i_train], ord=1)
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.linalg.norm(X[i_test] - self.train_X,ord=1, axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!

        X3 = X[:,np.newaxis,:]
        dists = np.sum(np.abs(X3 - self.train_X),axis=2)
        return dists
        

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        score = np.zeros((2,2),np.float64)

        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples32233
            #pred[i] = self.train_y[np.argmin(dists[i])]

            l_dist = dists[i]
            score.fill(0)
           # print(self.k)
            for j in range (0,self.k):
                p = np.argmin(l_dist)
                v = self.train_y[p]
                d = l_dist[p]
                l_dist = np.delete(l_dist,p)
                score[int(v)][0] += 1 
                score[int(v)][1] += d

            #print(score)
            if (score[0][0] == score[1][0]):
                if (score[0][1] == score[1][1]):
                    self.k += 1
                    pred = predict_labels_binary(dists=dists)
                    #pred[i] = False
                    self.k -= 1
                else:
                    pred[i] = score[1][1] > score[0][1]
            else:
                pred[i] = score[1][0] > score[0][0]
            
            #print(str(pred[i]) + "\n")

        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        #num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        score = np.zeros(10,np.int) 

        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            l_dist = dists[i]
            score.fill(0)
            for j in range(0, self.k):
                p = np.argmin(l_dist)
                v = self.train_y[p]
                l_dist = np.delete(l_dist,p)
                score[int(v)] += 1 
                #print(v)
                ##score[int(v)][1] += d
            
            pred[i] = np.argmax(score)
            #print(score)
            #print ("Pred[%d] = %d" %(i, pred[i]))

        return pred
