import streamlit as st
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(['Bahaouddyn', 'Belvanie', 'Brel', 'Clement', 'Danielle', 'Emeric', 'Harlette', 'Ines', 'Nahomie', 'Ngoran', 'Sasha'])

# Chargement du modèle
model = load_model('/home/nunmua/INF/M I/SEM 2/ML_II/TP/speaker_detection_gru.h5')

def convert_to_wav(filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
        temp_file.write(filename.read())
        temp_filename = temp_file.name

    y, s = librosa.load(temp_filename, sr=16000)
    yt, _ = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=64)
    new_filename = os.path.splitext(temp_filename)[0] + '.wav'
    sf.write(new_filename, yt, s)
    os.remove(temp_filename)

    return new_filename

def extract_mfcc_features(filename, n_mfcc=13):
    audio_path = convert_to_wav(filename)
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return np.array(mfcc_mean)

def prediction(audio_file):
    mfcc = extract_mfcc_features(audio_file)
    data = mfcc.reshape(1, 13)
    data_reshape = np.reshape(data, (data.shape[0], data.shape[1], 1))
    pred = model.predict(data_reshape)
    pred = np.argmax(pred, axis=1)
    pred_1d = pred.flatten()
    pred_decoded = encoder.inverse_transform(pred_1d)
    return pred_decoded

st.title("RECONNAISSANCE DU LOCUTEUR")
st.write("Veuillez télécharger un fichier audio pour identifier le locuteur")

audio_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "aac", "m4a", "ogg"])

if audio_file is not None:
    st.audio(audio_file)
    predictions = prediction(audio_file)
    st.write(f"Le locuteur prédit est {predictions[0]}")
