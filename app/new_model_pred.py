from tensorflow import keras
from keras.models import load_model
import numpy as np
import librosa

lstm_model = load_model("./lstm_cnn.h5")

def new_model_pred(file_path):
    classes = ["artifact", "murmur", "normal", "extrastole", "extrahls"]
    duration=10
    sr=22050
    input_length = sr * duration

    X, sr = librosa.load(file_path, sr=sr, duration=duration)
    dur = librosa.get_duration(y=X, sr=sr)

    # Pad audio file if necessary
    if round(dur) < duration:
        X = librosa.util.fix_length(X, size=input_length)

    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=52, n_fft=512, hop_length=2048).T, axis=0)
    feature = np.array(mfccs).reshape(1, 52, 1)

    # Predict using the LSTM model
    preds = lstm_model.predict(feature)
    new_model_pred_class = classes[np.argmax(preds)]
    new_model_confidence = np.amax(preds)

    return new_model_pred_class, new_model_confidence

