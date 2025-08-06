

import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, test_features, test_labels, feature_names, class_names):
    """
    Explains the model's predictions using SHAP, with enhanced visualizations.
    """
    print("Calculating SHAP values...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)
    except Exception as e:
        print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer...")
        background_data = shap.sample(test_features, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(test_features)

    # Global Feature Importance (Summary Plot)
    print("Generating SHAP summary plots for each class...")
    if isinstance(shap_values, list):
        for i, class_shap_values in enumerate(shap_values):
            plt.figure()
            shap.summary_plot(class_shap_values, test_features, feature_names=feature_names, show=False)
            plt.title(f"SHAP Feature Importance for {class_names[i]}")
            plt.tight_layout()
            plt.savefig(f"shap_summary_{class_names[i]}.png")
            plt.close()
    else:
        plt.figure()
        shap.summary_plot(shap_values, test_features, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        plt.close()

    # Local Explanations for Misclassified Instances
    print("Analyzing misclassified instances...")
    predictions = model.predict(test_features)
    misclassified_indices = np.where(predictions != test_labels)[0]

    if len(misclassified_indices) > 0:
        # Analyze the first 5 misclassified instances
        for i in misclassified_indices[:5]:
            print(f"\nAnalyzing misclassified instance {i}: Predicted={class_names[predictions[i]]}, Actual={class_names[test_labels[i]]}")
            
            # Generate a force plot for this instance
            if isinstance(shap_values, list):
                # For multi-class, explain the prediction for the predicted class
                shap.force_plot(explainer.expected_value[predictions[i]], shap_values[predictions[i]][i,:], test_features[i,:], feature_names=feature_names, matplotlib=True, show=False)
            else:
                shap.force_plot(explainer.expected_value, shap_values[i,:], test_features[i,:], feature_names=feature_names, matplotlib=True, show=False)
            
            plt.title(f"SHAP Explanation for Misclassified Instance {i}")
            plt.tight_layout()
            plt.savefig(f"shap_force_plot_misclassified_{i}.png")
            plt.close()
            print(f"Force plot saved as shap_force_plot_misclassified_{i}.png")

