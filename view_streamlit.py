import streamlit as st
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import librosa.display
import soundfile as sf
import io


def get_mel_spec(audio, sr):
    audio = librosa.util.fix_length(audio, size=154350)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    s_db = librosa.power_to_db(mel_spec, ref=np.max)
    raw_s_db = s_db.T.tolist()
    return raw_s_db


def get_raw_mfc(audio, sr):
    hop_length = np.floor(0.010 * sr).astype(int)  # 10ms
    # win_length = np.floor(0.020 * sr).astype(int)  # 20ms
    n_mfcc, n_mels, n_ftt = 13, 13, 2048
    audio = librosa.util.fix_length(audio, size=154350)
    raw_mfc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_ftt, hop_length=hop_length)
    raw_mfc = raw_mfc.T.tolist()
    return raw_mfc


def load_model_cnn(path):
    model = load_model(path)
    model.summary()
    return model


st.header('AI Diagnosis using Cough Audio')
uploaded_files = st.file_uploader("Choose a Cough file (*.wav, .ogg, .webm)", accept_multiple_files=True)
list_users = {
        'data': [],
        'id': [],
        'result': [],
    }
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.audio(bytes_data, format='audio/ogg')
    data, rate = sf.read(io.BytesIO(bytes_data))

    raw_feature = np.array([get_raw_mfc(data, 22050)])
    raw_feature = raw_feature.reshape(raw_feature.shape[0], -1, 13, 1)
    print(raw_feature.shape)

    list_users['data'].append(raw_feature)
    list_users['id'].append(uploaded_file.name.split(".")[0])
    list_users['result'].append("None")

submit_click = False
with st.form("my_form"):
    submit = st.form_submit_button(label='Submit', help="Submit List Coughs")
    if submit is True:
        submit_click = True
        # print("ID: ", list_users["id"])
        my_model = load_model_cnn('model/cnn_sigmoid_v6MFCCsF13.h5')

        result = []
        for i in range(0, len(list_users["data"])):
            res = my_model.predict(list_users["data"][i])
            list_users["result"][i] = "Positive diagnosis is " + str(round(res[0][0] * 100, 2)) + "%"
        pass

df = pd.DataFrame(list_users, columns=["id", "result"])
if submit_click is True:
    st.dataframe(df)
