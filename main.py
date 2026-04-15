from src.data.load_data import load_raw_data, save_processed_data
from src.data.preprocess import preprocess_data
# from src.features.build_features import build_features 
from src.models.train import train_model, save_model
from src.models.evaluate import evaluate_model, print_evaluation_results
from src.models.predict import load_model, predict



RAW_DATA_PATH = "data/raw/salaries.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def main():
    # 1. Load raw data
    df = load_raw_data(RAW_DATA_PATH)

    # 2. Basic preprocessing (ONLY cleaning, no feature engineering)
    df = preprocess_data(df)

    # 3. Train model (pipeline handles features + preprocessing)
    model, X_test, y_test = train_model(df)

    # 4. Evaluate model (on real salary scale)
    mse, r2, mae = evaluate_model(model, X_test, y_test)
    print_evaluation_results(mse, r2, mae)

    # 5. Save trained model
    save_model(model)

    # 6. Load model again (simulate real-world usage)
    loaded_model = load_model()

    # 7. Simulate new unseen data
    sample = df.drop('salary', axis=1).iloc[:5]

    # 8. Predict salaries
    predictions = predict(loaded_model, sample)

    print("\nSample Predictions (USD):")
    print(predictions)


if __name__ == "__main__":
    main()
