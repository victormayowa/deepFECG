
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# AAMI-compliant class mappings
AAMI_CLASSES = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Non-ectopic
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular ectopic
    'V': 2, 'E': 2,                         # Ventricular ectopic
    'F': 3,                                 # Fusion
    '/': 4, 'f': 4, 'Q': 4,                 # Paced/Unknown
}

def get_aami_class(symbol):
    """Maps an annotation symbol to its AAMI class."""
    return AAMI_CLASSES.get(symbol)

def apply_bandpass_filter(signal, fs=360):
    """Applies a band-pass filter to the signal."""
    lowcut = 0.5
    highcut = 45.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    """Normalizes the signal to a range of -1 to 1."""
    return (signal - np.mean(signal)) / np.std(signal)

def segment_heartbeats(signal, annotations, fs=360):
    """
    Segments the signal into individual heartbeats around the R-peak.
    A window of 200 samples (100 before and 99 after the R-peak) is used.
    """
    heartbeats = []
    labels = []
    window_before = 100
    window_after = 100 # 99 after + 1 for the peak itself

    for i, symbol in enumerate(annotations.symbol):
        aami_class = get_aami_class(symbol)
        if aami_class is not None:
            peak_sample = annotations.sample[i]
            start = peak_sample - window_before
            end = peak_sample + window_after
            if start >= 0 and end < len(signal):
                segment = signal[start:end]
                if len(segment) == 200: # Ensure consistent segment length
                    heartbeats.append(segment)
                    labels.append(aami_class)

    return np.array(heartbeats), np.array(labels)

def preprocess_data(data_path, test_size=0.2, max_records=None):
    """
    Loads and preprocesses the ECG data from the MIT-BIH Arrhythmia Database
    using an inter-patient data splitting strategy.
    """
    print(f"Loading data from {data_path}...")
    
    record_names = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.hea')]
    record_names.sort()

    # Inter-patient split
    train_records, test_records = train_test_split(record_names, test_size=test_size, random_state=42)

    def process_records(records):
        all_heartbeats = []
        all_labels = []
        for record_name in records:
            if max_records and len(all_heartbeats) >= max_records:
                break
            record_full_path = os.path.join(data_path, record_name)
            try:
                record = wfdb.rdrecord(record_full_path)
                annotations = wfdb.rdann(record_full_path, 'atr')
                
                if 'MLII' in record.sig_name:
                    signal_index = record.sig_name.index('MLII')
                else:
                    signal_index = 0
                signal = record.p_signal[:, signal_index]

                filtered_signal = apply_bandpass_filter(signal, fs=record.fs)
                normalized_signal = normalize_signal(filtered_signal)
                heartbeats, labels = segment_heartbeats(normalized_signal, annotations, fs=record.fs)
                
                all_heartbeats.append(heartbeats)
                all_labels.append(labels)
            except Exception as e:
                print(f"Error processing record {record_name}: {e}")
        
        if not all_heartbeats:
            return np.array([]), np.array([])
            
        return np.concatenate(all_heartbeats), np.concatenate(all_labels)

    print("Processing training data...")
    X_train, y_train = process_records(train_records)
    
    print("Processing testing data...")
    X_test, y_test = process_records(test_records)

    print("Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    # Reshape data for SMOTE
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)
    # Reshape back to original
    X_train_smote = X_train_smote.reshape(X_train_smote.shape[0], X_train.shape[1])


    return X_train_smote, X_test, y_train_smote, y_test
