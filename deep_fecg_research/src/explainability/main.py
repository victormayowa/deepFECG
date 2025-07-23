import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, test_features, test_labels):
    """
    Explains the model's predictions using SHAP.

    Args:
        model (object): The trained model (gcForest or CascadeForestClassifier).
        test_features (np.ndarray): The testing features.
        test_labels (np.ndarray): The testing labels.
    """
    print("Calculating SHAP values...")

    # For tree-based models like gcForest and CascadeForestClassifier, TreeExplainer is efficient.
    # If the model is a scikit-learn compatible tree ensemble, TreeExplainer should work.
    # If not, KernelExplainer is a more general but slower alternative.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)
    except Exception as e:
        print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer (may be slow)...")
        # KernelExplainer requires a background dataset for estimation
        # Using a subset of test_features as background for performance
        background_data = shap.sample(test_features, 100) # Sample 100 instances
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(test_features)

    print("Generating SHAP summary plot...")
    # If the model is multi-output (multi-class classification), shap_values will be a list of arrays.
    # For summary_plot, we often plot for one class or the absolute mean of all classes.
    if isinstance(shap_values, list):
        # For multi-class, plot the SHAP values for the first class (or average/sum them)
        shap.summary_plot(shap_values[0], test_features, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, test_features, plot_type="bar", show=False)

    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()
    print("SHAP summary plot saved as shap_summary_plot.png")

    # You can also implement logic to analyze misclassified instances
    # and generate local SHAP plots (e.g., shap.force_plot).