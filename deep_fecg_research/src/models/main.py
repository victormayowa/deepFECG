from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from deepforest import CascadeForestClassifier
from .gcforest import gcForest
import numpy as np


def train_and_evaluate(train_features, train_labels, test_features, test_labels, model_type='CascadeForest'):
    """
    Trains and evaluates the specified Deep Forest model with hyperparameter tuning.
    Args:
        train_features (np.ndarray): The training features.
        train_labels (np.ndarray): The training labels.
        test_features (np.ndarray): The testing features.
        test_labels (np.ndarray): The testing labels.
        model_type (str): The type of model to train ('gcForest' or 'CascadeForest').

    Returns:
        object: The trained model.
    """
    print(f"DEBUG: main.py - Calling train_and_evaluate with model_type={model_type}")
    print(f"Training and evaluating {model_type} model...")

    if model_type == 'CascadeForest':
        model = _train_cascade_forest_with_tuning(train_features, train_labels)
    elif model_type == 'gcForest':
        model = _train_gcforest_with_tuning(train_features, train_labels)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print("Evaluating the best model on the test set...")
    predictions = model.predict(test_features)
    
    # For roc_auc_score, we need probability estimates
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(test_features)
    else:
        probas = model.predict(test_features)

    _print_evaluation_metrics(test_labels, predictions, probas)

    return model


def _train_cascade_forest_with_tuning(train_features, train_labels):
    """
    Trains a CascadeForestClassifier model using GridSearchCV.
    """
    print("Training CascadeForestClassifier with hyperparameter tuning...")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [100, 200],
        'n_trees': [300, 500],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5]
    }
    
    cf = CascadeForestClassifier(use_predictor=True, n_jobs=1, random_state=42, partial_mode=False)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=cf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(train_features, train_labels)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def _train_gcforest_with_tuning(train_features, train_labels):
    """
    Trains a GCForest model with multi-grained scanning and hyperparameter tuning.
    """
    print("Training GCForest with multi-grained scanning and hyperparameter tuning...")

    # Determine shape_1X for sequential data (ECG)
    # Assuming train_features is 2D: (n_samples, n_features)
    # shape_1X should be (1, n_features) for sequence slicing
    feature_dim = train_features.shape[1]
    shape_1X_val = (1, feature_dim)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_mgsRFtree': [30, 50],  # Number of trees in MGS Random Forests
        'window': [[int(feature_dim * 0.1)], [int(feature_dim * 0.2)], [int(feature_dim * 0.3)]], # Example window sizes, adjust based on ECG characteristics
        'stride': [1], # Keep stride at 1 for now
        'n_cascadeRF': [2, 3], # Number of Random Forests in a cascade layer
        'n_cascadeRFtree': [101, 201], # Number of trees in cascade Random Forests
        'min_samples_mgs': [0.1, 0.05], # Min samples for split in MGS
        'min_samples_cascade': [0.05, 0.02], # Min samples for split in Cascade
        'cascade_layer': [np.inf], # Allow cascade to grow until tolerance
        'tolerance': [0.001], # Accuracy tolerance for cascade growth
    }

    # Instantiate gcForest with fixed parameters and shape_1X
    # n_jobs=-1 to use all available cores
    gcf = gcForest(shape_1X=shape_1X_val, n_jobs=-1)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced n_splits for faster tuning

    grid_search = GridSearchCV(estimator=gcf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(train_features, train_labels)

    print(f"Best parameters found for GCForest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def _print_evaluation_metrics(labels, predictions, probas):
    """
    Prints a comprehensive set of evaluation metrics.
    """
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # ROC AUC score for multi-class
    if probas.shape[1] == len(np.unique(labels)):
        roc_auc = roc_auc_score(labels, probas, multi_class='ovr', average='weighted')
        print(f"ROC AUC Score: {roc_auc:.4f}")
    else:
        print("ROC AUC Score could not be computed for this model.")

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
