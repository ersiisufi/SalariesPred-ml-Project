from src.data.load_data import load_raw_data, save_processed_data
from src.data.preprocess import preprocess_data
# from src.features.build_features import build_features 
from src.models.train import train_model, save_model
from src.models.evaluate import evaluate_model, print_evaluation_results


RAW_DATA_PATH = "data/raw/salaries.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def main():

    # Load raw dataset
    df = load_raw_data(RAW_DATA_PATH)

    #Preprocess Data
    df_clean = preprocess_data(df)

    # Save processed Data
    # save_processed_data(df_clean, PROCESSED_DATA_PATH)



    model, X_test, y_test = train_model(df)
    result = evaluate_model(model, X_test, y_test)
    print_evaluation_results(*result)

    save_model(model)

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()