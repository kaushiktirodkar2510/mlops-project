import joblib
import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to trained model
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"


def load_model():
    """Load the trained car price prediction model."""
    return joblib.load(MODEL_PATH)


def predict_price(input_data):
    """Predict the used car price from user input."""

    model = load_model()

    # Convert input dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)

    return prediction[0]