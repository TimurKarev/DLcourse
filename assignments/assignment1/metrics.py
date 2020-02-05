def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp =0
    fp =0
    fn =0
    tn = 0
    for i in range(0,ground_truth.shape[0]):
        if(prediction[i]==True and ground_truth[i]==True):
             tp = tp+1
            # print ("Iter {}  TP".format(i))

        if(prediction[i]==False and ground_truth[i]==False): 
            #print ("Iter {}  FP".format(i))
            fp = fp+1 

        if(prediction[i]==False and ground_truth[i]==True): 
            #print ("Iter {}  FN".format(i))
            fn= fn+1 

        if(prediction[i]==True and ground_truth[i]==False): 
            #print ("Iter {}  TN".format(i))
            tn= tn+1 

        #print ("p={} t={} \n".format(prediction[i], ground_truth[i]))
    
    #print ("tp = {} fp={} fn={} tn={}  , len = {}".format(tp,fp,fn,tn, ground_truth.shape))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+fp)/(tp+fp+tn+fn)
    f1 = (2*precision*recall)/(precision+recall)

    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    tp = 0
    tot_samples = len(prediction)
    for i in range(0,tot_samples):
        if (prediction[i] == ground_truth[i]):
            tp += 1;
    

    
    return tp/tot_samples
