
# Deep FECG Research

This project is a research platform for applying Deep Forest models to Fetal Electrocardiogram (FECG) data. It provides a pipeline for data preprocessing, feature extraction, model training and evaluation, and model explainability.

## Purpose of the Study & Objectives

The primary purpose of this study is to systematically investigate and comparatively evaluate the efficacy and interpretability of feature-engineered Deep Forest models for ECG arrhythmia classification. This research endeavours to develop a robust and transparent AI system capable of assisting cardiologists in accurate and trustworthy arrhythmia diagnosis, thereby directly addressing the current limitations associated with "black-box" deep learning approaches in clinical settings.

The overall aim is to enhance the interpretability and clinical utility of automated ECG arrhythmia classification systems while maintaining or improving upon current state-of-the-art diagnostic performance.

The specific objectives of this study are:

*   To systematically extract and conduct a comparative analysis of the performance of diverse signal processing features, including MFCC, DWT/SWT, HHT, and SSCWT, for effective ECG arrhythmia representation.
*   To implement and rigorously optimise Deep Forest models, specifically gcForest and CascadeForestClassifier, for multi-class ECG arrhythmia classification using the benchmark MIT-BIH Arrhythmia Database.
*   To perform a comprehensive comparative analysis of the classification performance of feature-engineered Deep Forest models against state-of-the-art deep learning models, evaluating key metrics such as accuracy, sensitivity, specificity, and F1-score.
*   To apply and evaluate SHAP (SHapley Additive exPlanations) to interpret the decision-making processes of the Deep Forest models, assessing the alignment of the generated model explanations with established cardiological knowledge and clinical guidelines.
*   To demonstrate how the inherent and post-hoc interpretability provided by feature-engineered Deep Forests can significantly enhance clinical trust and facilitate the practical adoption of AI in routine arrhythmia diagnosis.

## Installation

To run this project, you will need to have Python 3.9 installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/victormayowa/deep-fecg-research.git
    cd deep-fecg-research
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3.9 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for the project is `main.py`. You can run the experiment with different configurations using the command-line arguments.

```bash
python main.py --data_path /path/to/your/data --feature_extractor MFCC --model gcForest --explain
```

### Arguments

*   `--data_path`: Path to the dataset (default: `./data`).
*   `--feature_extractor`: Feature extraction method to use. Choices: `MFCC`, `DWT`, `HHT`, `SSCWT` (default: `MFCC`).
*   `--model`: Model to train. Choices: `gcForest`, `CascadeForest` (default: `gcForest`).
*   `--explain`: Whether to run SHAP explainability.

## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

*   numpy
*   scikit-learn
*   librosa
*   ssqueezepy
*   deep-forest
*   gcforest
*   shap
*   wfdb
*   matplotlib
*   seaborn
*   jupyter

## Project Structure

```
.
├── data/
├── notebooks/
├── src/
│   ├── explainability/
│   │   └── main.py
│   ├── features/
│   │   └── main.py
│   ├── models/
│   │   └── main.py
│   └── preprocessing/
│       └── main.py
├── tests/
├── main.py
├── README.md
└── requirements.txt
```
