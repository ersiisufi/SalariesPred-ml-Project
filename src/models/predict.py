import pandas as pd
import numpy as np
from joblib import load

def load_model(path= "models/model.joblib"):
    """Load the trained model from disk."""
    return load(path)

def predict(model, input_data: pd.DataFrame):
    """Make predictions using the loaded model."""

    y_pred_log = model.predict(input_data)

    y_pred = np.exp(y_pred_log)
    return y_pred

