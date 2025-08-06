
import numpy as np
import librosa
import pywt
import emd
from ssqueezepy import ssq_cwt
from scipy.stats import entropy

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
        heartbeat = heartbeat.astype(float)
        mfcc = librosa.feature.mfcc(y=heartbeat, sr=sr, n_mfcc=n_mfcc)
        mfccs.append(np.mean(mfcc.T, axis=0))
    return np.array(mfccs)

def _extract_dwt(data, wavelet='db2', level=4):
    """
    Extracts statistical features from DWT coefficients.
    """
    dwt_features = []
    for heartbeat in data:
        coeffs = pywt.wavedec(heartbeat, wavelet, level=level)
        features = []
        for c in coeffs:
            features.extend([
                np.mean(c), 
                np.var(c), 
                np.std(c), 
                entropy(np.abs(c)), 
                np.sum(np.square(c))
            ])
        dwt_features.append(features)
    return np.array(dwt_features)

def _extract_hht(data):
    """
    Extracts Hilbert-Huang Transform (HHT) features using EMD.
    """
    hht_features = []
    for heartbeat in data:
        imfs = emd.sift.sift(heartbeat)
        instantaneous_freq, instantaneous_amp = emd.spectra.frequency_transform(imfs, 360, 'hilbert')
        features = [
            np.mean(instantaneous_freq), 
            np.mean(instantaneous_amp), 
            np.var(instantaneous_freq),
            np.var(instantaneous_amp)
        ]
        hht_features.append(features)
    return np.array(hht_features)

def _extract_sscwt(data, fs=360):
    """
    Extracts features from the Synchrosqueezed CWT.
    """
    sscwt_features = []
    for heartbeat in data:
        Tx, _, _, _, _ = ssq_cwt(heartbeat, 'morlet', fs=fs)
        # Extract energy from different frequency bands
        freq_bands = [0, 5, 15, 25, 45]
        energy_features = []
        for i in range(len(freq_bands) - 1):
            band_energy = np.sum(np.abs(Tx[freq_bands[i]:freq_bands[i+1], :])**2)
            energy_features.append(band_energy)
        sscwt_features.append(energy_features)
    return np.array(sscwt_features)
