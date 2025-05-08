from src.decision_tree import predict_tree

def oob_score(tree, X_oob, y_oob):
    ...

def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)
