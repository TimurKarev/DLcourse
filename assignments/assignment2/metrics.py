def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    #print (prediction.shape, ground_truth.shape)

    # TODO: Implement computing accuracy
    tp = 0
    tot_samples = len(prediction)
    for i in range(0,tot_samples):
        if (prediction[i] == ground_truth[i]):
            tp += 1;



    return tp/tot_samples