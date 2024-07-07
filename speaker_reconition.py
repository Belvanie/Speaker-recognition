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

encoder = LabelEncoder()
encoder.fit(['Bahaouddyn', 'Belvanie', 'Brel', 'Clement', 'Danielle', 'Emeric', 'Harlette', 'Ines', 'Nahomie', 'Ngoran', 'Sasha'])
# Chargement du modele
model = load_model('speaker_detection_gru.h5')

# Chemin vers le dossier contenant nos fichiers
#fichier = '/home/nunmua/INF/M I/SEM 2/ML_II/TP/belvanie_test_2.m4a'

# Convertion des fichiers .m4a, .acc, .ogg en .wav
def convert_to_wav(filename):
    # Utiliser un fichier temporaire pour enregistrer le contenu de l'objet UploadedFile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
        temp_file.write(filename.read())
        temp_filename = temp_file.name

    # Charger le fichier temporaire avec librosa
    y, s = librosa.load(temp_filename, sr=16000)  # Charge le fichier et resample à 16000 Hz, ce qui nous permet de normaliser nos donnees
    yt, index = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=64) # top_db est le seuil en dB sous lequel le signal est considéré comme du silence
    
    # Créer un nouveau nom de fichier pour le fichier .wav
    new_filename = os.path.splitext(temp_filename)[0] + '.wav'  # Change l'extension du fichier
    
    # Écrire le fichier au format .wav
    sf.write(new_filename, yt, s)  # Écrit le fichier au format .wav
    
    # Supprimer le fichier temporaire .m4a
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

    # Afficher la prédiction
    #print(pred_decoded)
    return pred_decoded

#predictions = prediction(fichier)

# Interface utilisateur
st.title("RECONNAISSANCE DU LOCUTEUR")
st.write("Veuillez télécharger un fichier audio ou enregistrer votre voix pour identifier le locuteur")

# Ajouter une option pour télécharger un fichier audio
audio_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "aac", "m4a", "ogg"])

# Traitement et prédiction pour le fichier téléchargé
if audio_file is not None:
    st.audio(audio_file)#, format='audio/wav')
    predictions = prediction(audio_file)
    st.write(f"Le locuteur prédit est {predictions[0]}")
