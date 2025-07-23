
import argparse
from src.preprocessing.main import preprocess_data
from src.features.main import extract_features
from src.models.main import train_and_evaluate
from src.explainability.main import explain_model

def main(args):
    """
    Main function to run the experiment.
    """
    # 1. Preprocess the data
    print("Starting data preprocessing...")
    train_data, test_data, train_labels, test_labels = preprocess_data(args.data_path, max_records=args.max_records)
    print("Data preprocessing complete.")

    # 2. Extract features
    print("Extracting features...")
    train_features, test_features = extract_features(train_data, test_data, method=args.feature_extractor)
    print("Feature extraction complete.")

    # 3. Train and evaluate the model
    print("Training and evaluating the model...")
    model = train_and_evaluate(train_features, train_labels, test_features, test_labels, model_type=args.model)
    print("Model training and evaluation complete.")

    # 4. Explain the model
    if args.explain:
        print("Explaining the model...")
        explain_model(model, test_features, test_labels)
        print("Model explanation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Deep Forest ECG experiment.')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset.')
    parser.add_argument('--feature_extractor', type=str, default='MFCC', choices=['MFCC', 'DWT', 'HHT', 'SSCWT'], help='Feature extraction method.')
    parser.add_argument('--model', type=str, default='gcForest', choices=['gcForest', 'CascadeForest'], help='Model to train.')
    parser.add_argument('--explain', action='store_true', help='Whether to run SHAP explainability.')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to process for testing purposes.')
    args = parser.parse_args()
    main(args)
