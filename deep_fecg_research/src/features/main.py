import numpy as np
import librosa
import pywt
from ssqueezepy import ssq_cwt

def extract_features(train_data, test_data, method='MFCC'):
    """
    Extracts features from the preprocessed ECG data.

    Args:
        train_data (np.ndarray): Training data (heartbeats).
        test_data (np.ndarray): Testing data (heartbeats).
        method (str): Feature extraction method (MFCC, DWT, HHT, SSCWT).

    Returns:
        tuple: A tuple containing train_features, test_features.
    """
    print(f"Extracting features using {method} method...")

    if method == 'MFCC':
        train_features = _extract_mfcc(train_data)
        test_features = _extract_mfcc(test_data)
    elif method == 'DWT':
        train_features = _extract_dwt(train_data)
        test_features = _extract_dwt(test_data)
    elif method == 'HHT':
        train_features = _extract_hht(train_data)
        test_features = _extract_hht(test_data)
    elif method == 'SSCWT':
        train_features = _extract_sscwt(train_data)
        test_features = _extract_sscwt(test_data)
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

    return train_features, test_features

def _extract_mfcc(data, sr=360, n_mfcc=13):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs).
    """
    mfccs = []
    for heartbeat in data:
        # Ensure heartbeat is float type for librosa
        heartbeat = heartbeat.astype(float)
        mfcc = librosa.feature.mfcc(y=heartbeat, sr=sr, n_mfcc=n_mfcc)
        mfccs.append(np.mean(mfcc.T, axis=0)) # Take mean across time frames
    return np.array(mfccs)

def _extract_dwt(data, wavelet='db4', level=4):
    """
    Extracts Discrete Wavelet Transform (DWT) features.
    """
    dwt_features = []
    for heartbeat in data:
        coeffs = pywt.wavedec(heartbeat, wavelet, level=level)
        # Flatten coefficients and concatenate them
        features = np.concatenate([np.array(c).flatten() for c in coeffs])
        dwt_features.append(features)
    # Pad features to the maximum length if they are not uniform
    max_len = max(len(f) for f in dwt_features)
    padded_features = np.array([np.pad(f, (0, max_len - len(f)), 'constant') for f in dwt_features])
    return padded_features

def _extract_hht(data):
    """
    Extracts Hilbert-Huang Transform (HHT) features.
    Note: HHT implementation is complex and often requires external libraries
    or a custom implementation of EMD. This is a placeholder.
    For a full implementation, consider libraries like `emd`.
    """
    print("Warning: HHT feature extraction is a placeholder and returns dummy data.")
    # Dummy implementation: return mean and std of the signal as basic features
    hht_features = []
    for heartbeat in data:
        hht_features.append([np.mean(heartbeat), np.std(heartbeat)])
    return np.array(hht_features)

def _extract_sscwt(data, fs=360):
    """
    Extracts Synchrosqueezed Continuous Wavelet Transform (SSCWT) features.
    Note: SSCWT can produce high-dimensional output. This is a placeholder
    and returns a simplified representation.
    """
    print("Warning: SSCWT feature extraction is a placeholder and returns dummy data.")
    sscwt_features = []
    for heartbeat in data:
        # ssq_cwt returns (Tx, Wx, ssq_freqs, scales, wavel_scales)
        # We'll take the mean of the absolute value of the transform as a simple feature
        Tx, Wx, ssq_freqs, scales, wavel_scales = ssq_cwt(heartbeat, 'morlet', fs=fs)
        sscwt_features.append(np.mean(np.abs(Tx)))
    return np.array(sscwt_features).reshape(-1, 1)