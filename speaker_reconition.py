import streamlit as st
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import av

# Encoder pour les labels de locuteurs
encoder = LabelEncoder()
encoder.fit(['Bahaouddyn', 'Belvanie', 'Brel', 'Clement', 'Danielle', 'Emeric', 'Harlette', 'Ines', 'Nahomie', 'Ngoran', 'Sasha'])

# Chargement du modele
model = load_model('speaker_detection_gru.h5')

# Convertion des fichiers .m4a, .acc, .ogg en .wav
def convert_to_wav(filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(filename.read())
        temp_filename = temp_file.name

    y, s = librosa.load(temp_filename, sr=16000) 
    yt, _ = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=64) 
    new_filename = os.path.splitext(temp_filename)[0] + '.wav'  
    sf.write(new_filename, yt, s)
    os.remove(temp_filename)
    return new_filename

# Fonction pour l'extraction des caracteristiques
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

# Interface utilisateur
st.title("RECONNAISSANCE DU LOCUTEUR")
st.write("Veuillez télécharger un fichier audio ou enregistrer votre voix pour identifier le locuteur")

# Ajouter une option pour télécharger un fichier audio
audio_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "aac", "m4a", "ogg"])

# Traitement et prédiction pour le fichier téléchargé
if audio_file is not None:
    st.audio(audio_file)
    predictions = prediction(audio_file)
    st.write(f"Le locuteur prédit est {predictions[0]}")

# Enregistrement et traitement de l'audio en direct
st.write("Ou enregistrez votre voix :")
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={
            "audio": True,
            "video": False,
        },
    ),
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, np.concatenate([frame.to_ndarray() for frame in audio_frames]), samplerate=16000)
            audio_path = tmp_file.name

        st.audio(audio_path)
        predictions = prediction(audio_path)
        st.write(f"Le locuteur prédit est {predictions[0]}")