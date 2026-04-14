import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred_log = model.predict(X_test)

    # Convert back to real scale
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)


    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    mae= mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)



    return mse, r2, mae

def print_evaluation_results(mse, r2, mae):
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")