from src.data_loader import load_data
from src.random_forest_custom import random_forest, predict_rf
from src.sklearn_rf import find_best_random_state
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = load_data('data/aiml_project.csv')

print("Training custom Random Forest...")
trees = random_forest(X_train, y_train, n_estimators=100, max_features=3, max_depth=20, min_samples_split=2)
y_pred_custom = predict_rf(trees, X_test)
print(f"Custom RF Accuracy: {accuracy_score(y_test, y_pred_custom) * 100:.2f} %")

print("Training sklearn Random Forest...")
best_seed, best_score = find_best_random_state(X_train, y_train, X_test, y_test)
print(f"Best sklearn RF Accuracy: {best_score*100:.2f} % at seed {best_seed}")
