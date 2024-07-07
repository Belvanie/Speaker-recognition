import streamlit as st
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import av
import traceback

encoder = LabelEncoder()
encoder.fit(['Bahaouddyn', 'Belvanie', 'Brel', 'Clement', 'Danielle', 'Emeric', 'Harlette', 'Ines', 'Nahomie', 'Ngoran', 'Sasha'])

# Chargement du modèle
model = load_model('speaker_detection_gru.h5')

def convert_to_wav(filename):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name

        audio = AudioSegment.from_file_using_temporary_files(filename)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format='wav')

        return temp_filename
    except Exception as e:
        st.error(f"Erreur de conversion en WAV: {e}")
        return None

def extract_mfcc_features(filename, n_mfcc=13):
    audio_path = convert_to_wav(filename)
    if audio_path is None:
        return None
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return np.array(mfcc_mean)

def prediction(audio_file):
    mfcc = extract_mfcc_features(audio_file)
    if mfcc is None:
        return ["Erreur de traitement de l'audio"]
    try:
        data = mfcc.reshape(1, 13)
        data_reshape = np.reshape(data, (data.shape[0], data.shape[1], 1))
        pred = model.predict(data_reshape)
        pred = np.argmax(pred, axis=1)
        pred_1d = pred.flatten()
        pred_decoded = encoder.inverse_transform(pred_1d)
        return pred_decoded
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return ["Erreur de prédiction"]

st.title("RECONNAISSANCE DU LOCUTEUR")
st.write("Veuillez télécharger un fichier audio pour identifier le locuteur")

audio_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "aac", "m4a", "ogg"])

if audio_file is not None:
    st.audio(audio_file)
    predictions = prediction(audio_file)
    st.write(f"Le locuteur prédit est {predictions[0]}")

# Enregistrement et traitement de l'audio en direct
st.write("Ou enregistrez votre voix :")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={  # Ajout de rtc_configuration pour spécifier les configurations WebRTC
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
    audio_receiver_size=512,
)

if webrtc_ctx.audio_receiver:
    audio_frames = []

    while True:
        try:
            audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
        except av.error.FFmpegError:
            break
        audio_frames.append(audio_frame)

    if len(audio_frames) > 0:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            sf.write(tmp_file.name, np.concatenate([frame.to_ndarray() for frame in audio_frames]), samplerate=16000)
            audio_path = tmp_file.name

        st.audio(audio_path)
        predictions = prediction(audio_path)
        st.write(f"Le locuteur prédit est {predictions[0]}")
