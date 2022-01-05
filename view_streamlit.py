import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def lib_audio_data_wav(file):
    return librosa.load(file + '.wav')


def get_mel_spec(audio, sr):
    plt.ioff()
    fig = plt.figure()
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    s_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(s_db)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig=fig)
    return img


def load_model(path):
    my_model = tf.keras.models.load_models(path)
    my_model.summary()
    return my_model


def view_stream(img, new_model):
    st.title("Cough Detection by Do Manh Quang")

    res = new_model.predict(img)

    st.write('result: ', res)
    pass


if __name__ == '__main__':
    path_model = ''
    path_ex = ''
    sr, y = lib_audio_data_wav(path_ex)
    model = load_model(path_model)
    img = get_mel_spec(y, sr)
    view_stream(img, new_model=model)
    pass
