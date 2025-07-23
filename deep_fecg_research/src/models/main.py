from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from deepforest import CascadeForestClassifier
from gcforest.gcforest import GCForest

def train_and_evaluate(train_features, train_labels, test_features, test_labels, model_type='gcForest'):
    """
    Trains and evaluates the specified Deep Forest model.

    Args:
        train_features (np.ndarray): The training features.
        train_labels (np.ndarray): The training labels.
        test_features (np.ndarray): The testing features.
        test_labels (np.ndarray): The testing labels.
        model_type (str): The type of model to train ('gcForest' or 'CascadeForest').

    Returns:
        object: The trained model.
    """
    print(f"Training {model_type} model...")

    if model_type == 'gcForest':
        model = _train_gcforest(train_features, train_labels)
    elif model_type == 'CascadeForest':
        model = _train_cascade_forest(train_features, train_labels)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print("Evaluating the model...")
    predictions = model.predict(test_features)

    # Calculate and print performance metrics
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return model

def _train_gcforest(train_features, train_labels):
    print("Training gcForest...")
    # Initialize GCForest with some default parameters
    # These parameters can be tuned further for optimal performance
    gc = GCForest(shape_1X=train_features.shape[1],
                  n_mgs=1,
                  n_estimators_as_forest=[100],
                  min_samples_leaf=1,
                  max_depth=None,
                  n_tolerant_retry=10,
                  n_jobs=-1) # Use all available cores
    gc.fit(train_features, train_labels)
    return gc

def _train_cascade_forest(train_features, train_labels):
    print("Training CascadeForestClassifier...")
    # Initialize CascadeForestClassifier with some default parameters
    # These parameters can be tuned further for optimal performance
    cf = CascadeForestClassifier(n_estimators=100,
                                 n_trees=500,
                                 use_predictor=True,
                                 min_samples_leaf=1,
                                 max_depth=None,
                                 n_jobs=-1) # Use all available cores
    cf.fit(train_features, train_labels)
    return cf
