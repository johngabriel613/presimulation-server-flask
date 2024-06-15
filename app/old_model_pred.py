import joblib
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
cnn_model = joblib.load("cnn.pkl")
scaler = joblib.load("scaler.pkl") 

def old_model_pred(file_path):
    low_pass_cutoff = 195
    
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Load and preprocess the audio file
    x, sr = librosa.load(file_path, sr=22050, duration=10)

    # Filter the audio using a low-pass filter
    x_filtered = butter_lowpass_filter(x, low_pass_cutoff, sr)

    # Extract features
    mfcc_features = np.mean(librosa.feature.mfcc(y=x_filtered, sr=sr, n_mfcc=128), axis=1)
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(x_filtered))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=x_filtered, sr=sr))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=x_filtered, sr=sr))

    features = np.array([*mfcc_features, zero_crossings, spectral_rolloff, chroma_stft])
    features = features.reshape(1, -1)  # Reshape for a single sample

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = cnn_model.predict(scaled_features)
    confidence_scores = cnn_model.predict_proba(scaled_features)

    # Map the numerical label back to the original label
    label_mapping = {4: "normal", 3: "murmur", 0: "artifact", 2: "extrastole", 1: "extrahls"}
    old_model_pred_class = label_mapping[prediction[0]]
    old_model_confidence = confidence_scores[0][prediction[0]]

    return old_model_pred_class, old_model_confidence
