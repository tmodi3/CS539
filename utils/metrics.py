from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    """
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='macro'):
    """
    Calculate the precision of predictions.
    """
    return precision_score(y_true, y_pred, average=average)

def calculate_recall(y_true, y_pred, average='macro'):
    """
    Calculate the recall of predictions.
    """
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='macro'):
    """
    Calculate the F1 score of predictions.
    """
    return f1_score(y_true, y_pred, average=average)
