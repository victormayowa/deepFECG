
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
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

def segment_heartbeats(signal, annotations, fs=360, window_size=360):
    """
    Segments the signal into individual heartbeats.
    """
    heartbeats = []
    labels = []
    window_before = window_size // 2
    window_after = window_size - window_before

    for i, symbol in enumerate(annotations.symbol):
        aami_class = get_aami_class(symbol)
        if aami_class is not None:
            peak_sample = annotations.sample[i]
            start = peak_sample - window_before
            end = peak_sample + window_after
            if start >= 0 and end < len(signal):
                heartbeats.append(signal[start:end])
                labels.append(aami_class)

    return np.array(heartbeats), np.array(labels)

def preprocess_data(data_path, window_size=360, max_records=None):
    """
    Loads and preprocesses the ECG data from the MIT-BIH Arrhythmia Database.
    """
    print(f"Loading data from {data_path}...")
    
    # Get a list of all record names by listing .hea files
    record_names = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.hea')]
    record_names.sort() # Ensure consistent order

    all_heartbeats = []
    all_labels = []

    for i, record_name in enumerate(record_names):
        if max_records and i >= max_records:
            print(f"Reached max_records limit of {max_records}. Stopping data loading.")
            break
        print(f"Processing record: {record_name}")
        record_full_path = os.path.join(data_path, record_name)
        try:
            record = wfdb.rdrecord(record_full_path)
            annotations = wfdb.rdann(record_full_path, 'atr')

            # Use the first channel (MLII) if available, otherwise the first channel
            if 'MLII' in record.sig_name:
                signal_index = record.sig_name.index('MLII')
            else:
                signal_index = 0 # Default to first channel
            signal = record.p_signal[:, signal_index]

            # Apply band-pass filter
            filtered_signal = apply_bandpass_filter(signal, fs=record.fs)

            # Segment heartbeats
            heartbeats, labels = segment_heartbeats(
                filtered_signal, annotations, fs=record.fs, window_size=window_size
            )

            all_heartbeats.append(heartbeats)
            all_labels.append(labels)

        except Exception as e:
            print(f"Error processing record {record_name}: {e}")

    if not all_heartbeats:
        raise ValueError("No heartbeats processed. Check data_path and file integrity.")

    X = np.concatenate(all_heartbeats)
    y = np.concatenate(all_labels)

    print("Splitting data into training and testing sets...")
    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
