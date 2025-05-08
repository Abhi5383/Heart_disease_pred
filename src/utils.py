import numpy as np
import random

def entropy(p):
    if p in [0, 1]:
        return 0
    return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

def information_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if parent else 0
    p_left = left_child.count(1) / len(left_child) if left_child else 0
    p_right = right_child.count(1) / len(right_child) if right_child else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child)/len(parent)*IG_l - len(right_child)/len(parent)*IG_r

def draw_bootstrap(X_train, y_train):
    indices = np.random.choice(range(len(X_train)), len(X_train), replace=True)
    oob_indices = [i for i in range(len(X_train)) if i not in indices]
    return X_train.iloc[indices].values, y_train[indices], X_train.iloc[oob_indices].values, y_train[oob_indices]
