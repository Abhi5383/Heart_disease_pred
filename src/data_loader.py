import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    features = ['cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    df = df.sample(frac=1, random_state=217)  # Shuffle
    nb_train = int(0.9 * len(df))
    X_train = df[features][:nb_train]
    y_train = df['target'][:nb_train].values
    X_test = df[features][nb_train:]
    y_test = df['target'][nb_train:].values
    return X_train, y_train, X_test, y_test
